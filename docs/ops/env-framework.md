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
| **The registry is the only place that says what an env is.** Recipes are data; nothing computes a package list | § 2 |
| **A fact about one package lives ON that package.** No parallel `optional_*` lists — membership-by-name silently mis-declares | § 2.2 |
| **A bare string is the ordinary case**, a record only when the package needs more than a name | § 2.1 |
| **One door runs every command** — `run_step` — so `install` and `repair` cannot drift | § 3 |
| **Five outcomes, one transition rule.** Not nested conditions over return codes | § 3.1 |
| **The audit reads disk, never the network**, and answers on identity *and* provenance | § 4 |
| **Repair re-reads the RECORD**, never a flattened copy of an instruction | § 5 |

```
recipes.py ── Recipe ──┬── CondaPackage(spec, optional, reason)
  (the registry)       └── PipPackage(name, source, extras, optional,
                                      force, fallback_to_index, reason)
                                    │
                    ┌───────────────┴────────────────┐
              plan_install                      audit_packages
              (steps, no side effects)          (reads disk only)
                    │                                │
                    └────────► run_step ◄────────────┘
                               (the one door)              repair
                                    │
                              Outcome: OK · RECOVERED
                                       DEGRADED · FAILED · SKIPPED
```

---

## 1. What this document owns

How an environment gets described, created, checked and repaired: the shape of
the registry, the execution model, and the contract between them. It does not
own which packages any particular env needs — that is the recipe — nor the
source-build machinery (`builds.py`), which has its own resume semantics.

## 2. The registry

`molbuilder/envs/recipes.py` holds one `Recipe` per environment, and a recipe is
**data**. Nothing derives a package list at run time; what the file says is what
gets installed. `molbuilder envs` reads the registry and does not know how to
guess.

### 2.1 One rule for both package kinds

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

### 2.2 Optionality is a field, never a second list

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

### 2.3 What a pip record can say that a string cannot

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

## 3. Execution

`plan_install` builds `InstallStep`s and runs nothing. `run_step` runs exactly
one step. **Every dispatched command goes through it** — the installer and
`repair` both — which is what stops the two from drifting; they were separate
procedures until 2026-09-11 and had already diverged, repair issuing bare pip
commands that knew nothing of a package's alternatives.

`pip_argv` is the one place a pip command line is written. `pip_step_for` is the
one place a `PipPackage` becomes a step, so force flags, optionality and
fallbacks are read off one record by one reader.

### 3.1 Five outcomes, one rule

```
OK · RECOVERED · DEGRADED · FAILED · SKIPPED
```

> Try each argv in turn. The first success is **OK**, or **RECOVERED** if it was
> not the first attempt. If none succeed the step is **DEGRADED** when optional
> and **FAILED** when not.

That is the whole transition table, and naming the states is not decoration —
it is the fix for a bug class. The runner used to decide this with nested
conditions (is the code zero, is it `None` because the process never launched,
are there alternatives, is the step optional) and every edit lost a branch:
a launch failure once skipped the optional check and aborted an install it
should have survived, and a recovered step was reported under the command that
had failed.

`rc is None` is not a special case here. It is simply *not zero*.

`run_step` returns the argv that **actually ran**, so a recovered step can never
be reported under the attempt that failed.

## 4. The audit

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

## 5. Repair

`repair` acts on audit issues, and for each one it **maps back to the record**
by name and runs the step the installer would. It does not build commands and
does not carry a flattened copy of the instruction — an issue says what is
WRONG; the recipe says what to do about it.

This is why `PackageAuditIssue` carries `name` and not `install_flags`: a second
copy of an instruction can only drift from the first.

## 6. The shell shim

`scripts/install-env.sh` solves one chicken-and-egg problem — you cannot run
`molbuilder envs` until the host env exists — and forwards everything else. It
therefore duplicates the host package lists, and **a bash array can hold a name
and nothing else**. A host package that ever needs a source or a force flag
cannot be expressed there; the drift-guard test compares the arrays against
`conda_specs` / `pip_specs` and will fail the moment one grows a URL. That
failure is a decision to make, not a test to relax.

## 7. What is still asymmetric

`CondaPackage` carries `spec` (conda's own grammar, with its version and build
pin) while `PipPackage` carries a parsed `name` plus `source` and `extras`. That
is deliberate: conda's spec language is rich and its own, and the audit parses
it; pip's had to be taken apart because identity and source could not share one
string. The two are *modelled* alike — a record per package, optionality as a
field — and *spelled* differently where their package managers differ.
