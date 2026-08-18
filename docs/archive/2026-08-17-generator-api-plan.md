# Making the doors take the object — the work behind A8 and A9

**Role:** plan
**Domain:** execution

**Companions — the contracts this plan implements, and where the two disagree
those win:** [`execution/architecture.md`](?doc=execution/architecture.md)
**§ 3.1** (an object travels whole) and **§ 7, rules A8 · A9** — the rule this
plan exists to satisfy; [`execution/generator.md`](?doc=execution/generator.md)
**§ 6.2** (the five steps, and what each may assume) ·
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 6.1 (the
artifact registry) and § 6.2 (config ↔ scheduler names, and the two-names note).

> **The rules are in the contracts, not here.** This file holds only the
> measurement that earned them and the order the work is done in. When the last
> step lands it goes to `archive/` — a plan that outlives its work becomes a
> second, staler copy of a contract.

---

## 1. What was already decided, and is not reopened

**Three of the four pillars of a config-driven generator exist and are built.**
Naming them is the point: a plan that re-derives settled design costs more than
the defect it fixes.

| pillar | where it lives | state |
|---|---|---|
| **data** — one structure per fact | `Item` (`template.md` § 5) · `Task` (`stages.md` § 6) · `ParameterSet` (`generator.md` § 5) · `Resources` · `JobSet` | **built** |
| **file** — which artifact holds which fact | `job-contracts.md` § 6.1 — fourteen rows, each with file, schema string and owning module | **built** |
| **vocabulary** — one name per concept per layer | `job-contracts.md` § 6.2, with the translation boundary named (`resolve.py`, floor 3) | **built** |
| **API** — one door per artifact, taking the whole structure | `architecture.md` § 3.1 + A8/A9 | **contracted 2026-08-17; unbuilt** |

The data model is not the problem and is not being changed. Every fact this plan
moves already has a home.

---

## 2. The measurement that earned the rule

One intent — 16 ranks × 8 cores — rendered through each of the two call sites of
`write_run_wrapper`, kwargs copied verbatim from the source:

| caller | `.run.sh` OMP default | `.sbatch` `-c` |
|---|---|---|
| `jobset/prep.py:166` | **1** ❌ | 8 ✅ |
| `web/blueprints/build.py:257` | 8 ✅ | **absent** ❌ |

Eleven loose keyword arguments; ten passed by one caller and five by the other.
**Each is correct about one artifact and wrong about the other**, and neither
produces a correct pair.

**It had already fired once.** `max_memory_mb` was lost the same way on
2026-08-11 — the docstring on `Resources` records it, ending *"carried on the
allocation, it cannot be forgotten by one of them."* That fix moved the field
onto `Resources` and left the calling convention alone, so the sentence was
never bought and the class stayed open.

**Why no test caught it.** Every wrapper test calls the renderer with explicit
kwargs, so both artifacts have only ever been compared to a test's intent —
never to each other. That absence is what A9 names.

---

## 3. The order to do it in — **all five landed 2026-08-17**

Each guard was **watched failing before its fix**, which is the only thing that
makes it a guard: A8 named `jobset/prep.py:166` as the single offence in the
package, and all four of A9's cases failed. After the change both pass, and the
mutation (`omp_threads` derived from nothing) puts A9 red again.


1. **`test_architecture_rules` gains A8** — a generator door's parameter names,
   intersected with the fields of any § 3 object it already takes, must be
   empty. **It must fail on today's `write_run_wrapper`** before anything moves.
2. **`test_runwrap_pair` — A9's checker.** One `Resources` in; `.run.sh` and
   `.sbatch` out; ranks, cores and the GPU ask compared **across the pair**.
   **It must fail on both call sites.** A guard that is green on arrival has
   proved nothing.
3. **`write_run_wrapper(script_path, *, resources: Resources, env=None,
   emit_sbatch=True)`** — the signature change, `omp_threads` deleted as a
   parameter, both call sites converted.
4. **`build.py` assembles a `Resources`.** The Build tab passes three loose
   values today and builds no allocation at all; giving it one is what makes its
   missing fields (`time`, `mem`, `gres`) *visible* rather than absent.
5. **Re-run the HPC probe** — CPU and GPU, `bench` and `run` — and read the
   generated scripts, not the exit codes.

**Steps 3–4 are one commit**, or the boundary is inconsistent in the middle of
it. *(Steps 1–2 were planned as a separate commit so the failing guards would be
on the record first. They ship in the same commit instead: a commit whose test
suite is red is a worse record than a commit message that says which guards were
watched failing, and 71 test call sites had to move in the same breath as the
signature.)*

### 3.1 What the conversion touched, and what it found

The signature change moved **71 call sites across nine test files**, done as an
AST-located source splice rather than by hand or by `sed` — the calls appear in
two forms (`write_run_wrapper(...)` and `runwrap.write_run_wrapper(...)`), and a
text substitution had already put an import inside a parenthesised multi-line
import block and produced a `SyntaxError`.

**One failure was uncovered, first reported as pre-existing, and was not.**
`test_runwrap.py::test_pyscf_cold_actually_moves_optimized_xyz_aside` — `--cold`
did not move `<job>_optimized.xyz` aside. **Fixed 2026-08-17**; the cause and
the rule are `job-contracts.md` § 4.1's anchoring paragraph.

> **The mis-diagnosis is recorded because the method that produced it looks
> sound.** The failing code is `_cold_restart_aside_block` at `runwrap.py:315`;
> this change's hunks are at 35, 39 and 3367+, so *"my diff does not touch
> it"* was true — and useless. The cause was `{label}.xyz` joining
> `identity.OUR_FILE_PATTERNS` in `c10e1acd`, **committed earlier the same
> day**, so it was never going to appear in a working-tree diff.
>
> **A diff shows what you have not yet committed, not what you have not
> caused.** When the question is *"is this mine?"*, the thing to search is the
> failing symbol's inputs — here, the one enumeration the block derives its
> exception list from — and `git log -S` over the session's own commits, not
> `git diff`.

---

## 4. Deliberately not in scope, with the reason

**The wrapper's string assembly.** `render_run_wrapper` is ~1780 lines emitting
bash through ~295 f-strings — a real maintenance risk, and a fair reading of
*"handcrafted text injection"*. **It is not where either defect entered**: it has
one entry point and one caller, and both bugs arrived above it, at the boundary
this plan closes. Folding it in would turn a two-file change into a rewrite of
the launch floor. Recorded as its own candidate.

**The duplicated GPU-detection algorithm.** *"Does this deck request the GPU"*
runs twice — in Python at prep (`_fdf_requests_gpu`, for the sbatch header) and
in awk at launch (inside the wrapper, after a person may have edited the deck).
**Two implementations are required here**, because one runs on a login node and
the other on a compute node hours later; the truthy set is already a shared
constant (`_GPU_TRUTHY`), so the *fact* has one home. The honest guard is a test
rendering both against one deck set — not a merge that cannot happen.

**`bench` in `molbuilder/task@1`.** `stages.md` § 6.8 puts a sweep in the
description; `generator.md` § 4.3 says a sweep is **never** a field of one; and
`stages.md` cites `generator.md` nowhere. Nothing reads `Task.bench` today, so
the Task-setup tab's machine rows are write-only. **A contract-versus-contract
conflict is decided by the user, not resolved inside a refactor** — it waits.
