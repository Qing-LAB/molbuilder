# The env framework — one registry, one runner, one outcome

**Role:** contract
**Domain:** ops
**Companions:** [`installation.md`](?doc=ops/installation.md) — the GUIDE a
person installing molbuilder reads; this page owns the machinery underneath it.
[`deployment.md`](?doc=ops/deployment.md) § 1.0a–1.0c — the supervisor
discipline the runner's verify-before-signal rules were copied from.
[`engines/template.md`](?doc=engines/template.md) — the catalogue, which is a
different registry for a different thing (calculation parameters, not env
packages) and must not be confused with this one.

## The short version

| rule | where |
|---|---|
| **Two state machines, and they answer different questions.** `EnvState` about the env before we touch it; `Outcome` about each step after we run it | § 2 |
| **The registry is the only place that says what an env is.** Recipes are data | § 3 |
| **A fact about one package lives ON that package.** No parallel `optional_*` lists — membership-by-name silently mis-declares | § 3.2 |
| **A bare string is the ordinary case**, a record only when the package needs more than a name | § 3.1 |
| **The sequence is forced by dependency**, not by list order, and the document says which orderings may never move | § 4.1 |
| **Every situation is a FIELD on a record**, never a branch in the runner | § 4.2 |
| **Optionality means different things to a solver and to a per-package installer** — a non-fatal step for pip, a degraded attempt for conda | § 4.3 |
| **One door runs every step** — `run_step` — so no phase can grow its own dispatch again | § 5 |
| **Five outcomes, one transition rule**, and every step in a result carries one | § 5.1 |
| **A step's success criteria ride on the STEP**, so the runner has no per-phase special cases | § 5.2 |
| **The verdict is DERIVED from the steps**, never tracked beside them | § 5.3 |
| **The audit reads disk, never the network**, and answers on identity *and* provenance | § 6 |
| **Repair re-reads the RECORD**, never a flattened copy of an instruction | § 7 |

```
recipes.py ── Recipe ──┬── CondaPackage(spec, optional, reason)
  (the registry)       └── PipPackage(name, source, extras, optional,
   § 3                                force, fallback_to_index, reason)
                                    │
          ┌─────────────────────────┴──────────────────┐
   create_step_for · pip_step_for · verify_step_for   audit_packages
     conda_argv       pip_argv                        (reads disk only)
        (record  ->  InstallStep; the ONLY translators)      │     § 6
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
                                              ("re-run --clean")
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

Three cheap observations, combined into five states by `probe_env_state`. It is
a pure read, taken before any subprocess work, so a half-broken env is
diagnosed in a second instead of failing ten minutes into a `conda create`.

| registry | directory | `conda-meta/` | state | what the installer does |
|---|---|---|---|---|
| no | no | — | **FRESH** | dispatch `conda create` |
| yes | yes | yes | **PRESENT** | `SKIPPED` — resume into the existing env |
| no | yes | yes | **ORPHAN** | `FAILED` — conda would refuse; name `--clean` |
| yes | no | — | **GHOST** | `FAILED` — a registry entry with no directory |
| — | yes | no | **BROKEN** | `FAILED` — a directory that is not an env |

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

Naming the states is not decoration; it is the fix for a bug class. The runner
used to decide this with nested conditions — is the code zero, is it `None`
because the process never launched, are there alternatives, is the step
optional — and every edit lost a branch. Twice: a launch failure skipped the
optional check and aborted an install it should have survived, and a recovered
step was reported under the command that had failed.

Two predicates read these states, and they answer **different questions**:

| | asks about | answer for `DEGRADED` |
|---|---|---|
| `is_success` | did the env get what this STEP was for? | **False** — the package is honestly absent |
| `stops_the_install` | must everything after this be abandoned? | **False** — which is what `optional` buys |

`DEGRADED` is the only state where the two disagree, and that disagreement is
the entire point of optionality. Using the wrong predicate turns one
unavailable GPU wheel into a failed CPU env.

## 3. The registry

`molbuilder/envs/recipes.py` holds one `Recipe` per environment, and a recipe is
**data**. Nothing derives a package list at run time; what the file says is what
gets installed. `molbuilder envs` reads the registry and does not know how to
guess.

### 3.1 One rule for both package kinds

A **bare string is the ordinary case** and is normalised into a record. A record
is written out only when the package needs something a name cannot say:

```python
conda_packages=("python=3.12", "pip", "numpy"),
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

There used to be an `optional_conda_packages` list beside `conda_packages`, and
an `optional_pip_packages` beside `pip_packages`. Both are gone, for two
reasons that took real bugs to learn:

* **Membership matched by name fails silently.** An entry that matched nothing
  did not raise — it left its package REQUIRED, and the first anyone heard was
  the audit reporting it missing with nothing to explain why.
* **It only reached the audit.** Every pip package went into one combined
  install command, so a single unavailable GPU wheel aborted an otherwise
  healthy CPU env *regardless* of how optional the recipe called it. The word
  described the report and not the behaviour.

`optional` on the record governs **both**: the audit reports absence as
informational, and the installer gives the package its own non-fatal step.

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

## 4. The pipeline

### 4.1 The sequence, and which orderings may never move

A recipe's steps are not run in list order because a list has an order — most
of the sequence is **forced by dependency**, and those orderings are rules:

| # | stage | why it sits here |
|---|---|---|
| 1 | `conda create` | **Forced.** Nothing installs into an env that does not exist. It is also the one step needing no env prefix. |
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
    return [create_step_for(recipe),    # ...this is what --dry-run prints
            *pip_steps_for(recipe),
            *extra_steps_for(recipe),
            verify_step_for(recipe)]


def run_step(step, prefix):             # THE ONE DOOR
    for n, argv in enumerate((step.argv, *step.fallbacks)):
        rc, output = dispatch(bypass_conda_run(argv, prefix))
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
labels the output. That is the test of whether the unification is real: verify
used to need a loop of its own, and what made it special was policy read off
the *recipe* at execution time. Moved onto the step, the difference vanishes.

It branches on `StepRole`, never on `label`. The label is display text; four
sites used to key control flow on it, so renaming one for clarity silently
disabled the create-skip and every install then re-attempted `conda create` on
an existing env.

### 4.3 How a field covers each situation

Each row is something an env actually needs. **None of them is a branch in the
runner** — every one is a field on a record that a translator turns into a
step the one door can run.

| the situation | expressed as | what happens | outcome when it goes wrong |
|---|---|---|---|
| an ordinary package | a bare string in the list | batched into one `pip install` | `FAILED` |
| a package whose version cannot prove it is the declared build | `force=True` | `--force-reinstall --no-deps`, its own step, every time | `FAILED` |
| a pip package the env is usable without | `optional=True` | its own step, `fatal=False` | `DEGRADED`, install continues |
| a conda package the env is usable without | `optional=True` | **a degraded attempt on the create step** — see below | `RECOVERED` |
| a package not on PyPI | `source="git+https://…"` | PEP 508 `name @ url` | `FAILED` |
| …whose source may be unreachable | `fallback_to_index=True` | the indexed build as a declared alternative, **flagless** | `RECOVERED` |
| a tool that reports through its output and exits non-zero | `ignore_exit_code=True` | the substring becomes the verdict | `FAILED` if absent |
| a tool that exits 0 while the thing we asked for is missing | `expect_contains="…"` | checked in addition to the exit code | `FAILED`, naming the substring |
| the env already exists | — (`EnvState`) | create is not dispatched | `SKIPPED`, and no exit code |
| the env is wreckage | — (`EnvState`) | nothing is dispatched | `FAILED`, naming `--clean` |
| a binary that must be compiled | `build_spec=BuildSpec(…)` | `builds.py`'s own executor, results adapted | `FAILED` per phase |
| "tell me, do not do it" | `--dry-run` | `plan_install` only; nothing dispatches | no outcome at all |

That last row is why an outcome is `Optional`: a step with no outcome is a step
that is still a *plan*. Everywhere else, every step carries one.

**Optionality means something different to a solver.** pip installs one package
at a time, so an optional pip package is its own non-fatal STEP. conda solves
everything at once — an optional conda package cannot be its own step without
paying a *second solve*, and a second solve may legitimately change versions for
packages the first one already placed. So conda optionality is an **attempt**,
not a step property: `create_step_for` tries the full solve, and on failure the
same solve without the optional specs. The env lands `RECOVERED` — created,
minus something the recipe said it could live without.

**One degradation step, not a search.** With *n* optional specs, "find the
largest subset that still solves" is 2ⁿ solves at minutes each, so the rule is
all of them or none. Nothing is lost silently: the audit then names exactly
which optional packages are absent, with the recipe's `reason` beside each, and
`repair --include-optional` installs them one at a time — paying the second
solve there because the operator asked for it. A recipe with no optional conda
package gets no fallback, so its plan is byte-identical to one written before
this existed.

**`fallback_to_index` carries no flags, deliberately.** Its job is to make sure
the package EXISTS, not to replace what is there — with a force flag, an
unreachable source during an unrelated install would overwrite a good git tree
with a worse indexed one.

## 5. Execution

`plan_install` builds `InstallStep`s and runs nothing. `run_step` runs exactly
one step. **Every step goes through it** — every install phase, `repair`, and
`doctor`'s verify probe — which is what stops them from growing private
dispatch again. They had: `repair` issued bare pip commands that knew nothing of
a package's alternatives, the verify phase kept its own loop, and `doctor` kept a
third copy. Each copy re-derived the same four things — prefix resolution, the
`conda run` bypass, the launch-failure branch, the output trim — and `doctor`'s
had already drifted to a different output limit.

A record becomes a step in exactly one place per kind — `create_step_for` for
the env itself, `pip_step_for` for a `PipPackage`, `verify_step_for` for a
recipe's verify fields, `conda_step_for` for `repair`'s batched install. One
place writes each command line: `pip_argv` and `conda_argv`. So force flags,
optionality, fallbacks and verify criteria are each read off one record by one
reader, and the channel flags cannot sit before the specs in one caller and
after them in another, which is what happened while `conda_argv` did not exist.

### 5.1 What the five outcomes mean in a result

The states and the transition rule are § 2.2 — this is what they imply for the
`InstallResult` a caller reads.

`rc is None` is not a special case in that rule. It is simply *not accepted*.

**Every step in an `InstallResult` carries an outcome** — including the ones
decided *without* being dispatched (`_undispatched`): a skipped create, an env
too broken to install into, an unresolvable prefix. Those keep `returncode =
None`, because a step that did not run has no exit code; the skip used to report
`0`, which made "I did not do this" indistinguishable from "I did this and it
worked". The only step with no outcome is one that is still a *plan*, which is
what `--dry-run` prints.

`run_step` returns the argv that **actually ran**, so a recovered step can never
be reported under the attempt that failed.

### 5.2 What counts as success rides on the step

An exit code alone is not enough. **Every** recipe adds a substring its verify
output must contain (`expect_contains`) — a tool can exit `0` while the thing we
asked about is missing — and one also ignores the exit code entirely
(`ignore_exit_code`, for `tleap`, which exits non-zero from a perfectly healthy
start, so its banner IS the verification). Both ride on the **step** and are
checked by `InstallStep.accepts`.

They used to be read off the *recipe* inside the verify loop, which is precisely
why verify needed a loop of its own. On the step, verify is an ordinary step and
the runner needs no knowledge of which phase it is serving. A process that never
launched is never accepted — ignoring an exit code is not ignoring a missing
process.

### 5.3 The install's verdict is derived

`succeeded` is `not any(step.outcome.stops_the_install ...)`, computed from the
steps at the end. It used to be a `bool` tracked alongside them, and the two
drifted: two *failures* announced themselves to the user as "SKIPPED" because the
word printed and the verdict recorded came from different places.

Note **which** predicate. `is_success` asks about one STEP — and is `False` for
`DEGRADED`, because an optional package that could not be installed is honestly
absent. `stops_the_install` asks about the INSTALL. `DEGRADED` is exactly where
the two answers differ, and that difference is the whole point of `optional`.

### 5.4 What is deliberately outside the door

`validate.py`'s post-install probes (SIESTA's `ctest`, ELPA's `make check`)
dispatch on their own. They are not install steps and do not become them: they
answer "is this build scientifically sane", they need a working directory and
their own output parsing, and they report a `ProbeResult`. The rule is that every
`InstallStep` goes through `run_step`, not that no subprocess may exist.

### 5.5 Source builds keep their own executor

`builds.run_build_spec` is not a sequence of `InstallStep`s and does not become
one: it has sentinel-based resume, which no install step has. What the installer
does is **adapt** its results — each build phase becomes a step whose outcome
comes from `Outcome.decide`, so build steps stop arriving with no outcome at
all — while `builds.py` keeps its own verdict, which the installer reads rather
than re-derives.

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
from and carries that package's `reason`. The conda half used to walk spec
strings, which forced a name-set of the optional ones — the very mechanism
§ 3.2 abolished for failing silently — left `name` empty so `repair` could not
map a conda issue back to anything, and meant `CondaPackage.reason` could never
reach a user. `optional` suffixes every conda kind, not only absence: an
optional package resolving to the wrong build must not fail the health check for
an env the recipe calls usable without it.

`doctor` also runs the recipe's **verify** step, and it runs the same step the
installer does — built by `verify_step_for`, dispatched by `run_step`, judged by
`accepts`. That is the only way the health report can agree with what `install`
just said. It trims its excerpt tighter (2 KiB against the installer's 4) because
it prints one block per env and there are a dozen envs; that is a display choice,
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
package is dropped from the create solve instead. `repair` is the place that
pays a second solve, because the operator asked for one.

## 8. The shell shim

`scripts/install-env.sh` solves one chicken-and-egg problem — you cannot run
`molbuilder envs` until the host env exists — and forwards everything else.

Because it owns that one install, it has to honour one flag it otherwise just
forwards: **`--dry-run` means nothing gets installed, including by the shim.**
Creating the host env is the largest install in the flow, it used to happen
before the Python layer that honours the flag, and on a fresh machine — the only
case where a bootstrap dry run is interesting — "print the plan; do not install"
performed several GB of conda create. Planning genuinely requires the env, so
`bootstrap --dry-run` now refuses and names the command that makes it work.
`--gcc` is still the only flag the shim *consumes*; this one it reads and also
passes on.

**The flag reference is Python's.** The shim's help names only what Python
cannot know — `--gcc`, which it consumes because `recipes.py` reads
`MOLBUILDER_GCC` at import time, and the two flags it reads and forwards — and
points at `<subcommand> --help` for the rest. It used to re-specify every flag
and had drifted in four places, including two flags the Python layer *prints as
the command to run*: a second copy of a surface nothing can keep in sync is
worse than no copy. (Which is why `--help` implies `--yes`: otherwise the
pointer stops at the env-manager prompt and, with no TTY, prints nothing.)

It duplicates the host package lists, and **a bash array can hold a name
and nothing else**. A host package that ever needs a source or a force flag
cannot be expressed there; the drift-guard test compares the arrays against
`conda_specs` / `pip_specs` and will fail the moment one grows a URL. That
failure is a decision to make, not a test to relax.

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

What is NOT asymmetric any more, and must not drift back: both kinds are a
record per package, both normalise from a bare string, both carry `optional` and
`reason`, both are audited by iterating records, and both reach `repair` through
a lookup by name.
