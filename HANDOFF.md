# Handoff — committed; nothing has actually been run

## SCOPE

**Structure-optimization tab, SIESTA and PySCF.** `molbuilder/transport/` and
`molbuilder/spectra/` are deferred and byte-identical to `HEAD`. Do not touch,
analyse, or fold them into a finding.

| | |
|---|---|
| suite | **6888 passed, 0 failed** — `python tools/testrun.py run none2e`, ~22 min |
| reference decks | **36 cases, digest `149ac0714089ab85`**, harness `<scratch>/refgen.py` |
| uncommitted | nothing — this session's 58 files are the commit this file rides in |
| server | the user runs it; do not start one. `https://qlabsrv.physics.asu.edu:8888` |

---

## What remains, in priority order

### 1. Commit — DONE. One commit in the house shape; this file rides in it.

### 2. The end-to-end run — the original ask, never done.

**Everything so far validates generated scripts. Not one calculation has been
run.** That is the whole remaining half.

Through the browser, not a driver script calling internal functions:

> Structure optimization tab → *Send to Task setup* → Task setup → save →
> `jobset prep` → `jobset submit`, **one job at a time, never in parallel**.

Ready for it:

* a `claude-e2e` project exists on the server with the canonical subdirs.
  **Nothing has been written into the user's own projects.**
* machine config resolves to `source ~/miniconda3/etc/profile.d/conda.sh` +
  `conda activate`; no mamba on PATH.
* generated wrappers target `molbuilder-siesta` and `molbuilder-pySCF`, and
  both envs have their engines installed.
* a bare non-interactive `bash <script>.run.sh` **refuses by design** — that is
  the launch-door gate, and `MB_LAUNCHED_BY=manual` is its documented override.
  Not a bug; do not "fix" it.

Model systems already exercised at the *generation* level (water, BDT, BDT⁻,
O₂ triplet, an Au(111) slab) across both engines and both directory shapes.
All prepped clean. None has been executed.

### 3. Collapse the instance tests. This is my mess.

35 test functions generate **332 cases that re-run one rule with different
nouns** — `test_siesta_default_values_render_in_fdf` runs 20 times to prove one
walk works. This is why every problem this session was found by inspecting
outputs: the tests inspect outputs.

The rule for what to keep, proportionate rather than absolute:

| | needs |
|---|---|
| **Declared data** — the layout table, item lists, record values | nothing of its own; the walk is tested once |
| **A step that selects** — simple if/else over a value: `line`, `note_lead`, `section_title`, `validate_subject`, `provenance_defaults`, `bench_marks`, `check_rules` | nothing of its own; **the rule is enough** |
| **A step that emits CODE** — `Block.render`, and what it reaches: `runtime_info.emit_threading_setup_lines` · `emit_gpu_probe_lines` · `emit_pyscf_post_import_lines` · `emit_runtime_info_capture_lines` · `trajectory_log.MolwatchEmitter` | **its own test** — it introduces statements, scopes and names nothing models |

Worked through: `line` returns one deck line, and the rule *"every line the
parameters step wrote is in the file"* already covers it whatever it returns.
`bench_marks` returns values for a block the framework writes. `check_rules`
returns issues the framework reports uniformly, and what they SAY is engine
science with its own validator tests. **`Block.render` is the only hook handing
back text nothing models** — which is exactly where the `scf` collision came
from.

Start with the instance tests I added today (spin, charge and k-grid appearing
in a deck); they are the wrong shape by this rule.

### 4. Optional — give the PySCF deck's namespace an owner.

A SIESTA deck is a keyword list; names cannot collide. **A PySCF deck is a
Python program**, and five independent pieces of molbuilder write 1003 lines
into its one namespace: `pyscf/input.py`, four `runtime_info.emit_*` functions,
and the whole `MolwatchEmitter` class injected verbatim via
`inspect.getsource`. Nothing coordinates the names; they coexist by a prefix
convention — molbuilder's imports are `_mw_np`, `_mb_socket`, `_os`, while the
unprefixed `gto` / `scf` / `dft` / `mol` / `mf` belong to the engine and the
reader.

**The one place that broke it is fixed** — the molwatch callback bound a bare
`scf`, rebinding the pyscf module to a list of dicts in 32 of 64
configurations. Whether to make collision *impossible* rather than *detected*
is a judgement: for, because being correct currently requires an emitter to
know what the other four are called; against, because one place out of the
whole surface broke it. If it is done, it is one place that owns the reserved
names — not a framework.

`tests/test_emitted_program_namespace.py` currently detects collisions across
the configuration space. That is a smoke alarm standing in for a design.

---

## What was actually wrong this session, and is fixed

* **Every spin-polarized SIESTA deck with a fixed spin was refused.** One
  setting legitimately emits a *pair* (`Spin.Fix` + `Spin.Total`) and the check
  gate compared the two-line emission as one blob against single lines.
* **The reference harness was pinning a `TypeError` as an expected deck.** Its
  spin case passed `spin_polarized`, which `SiestaConfig` does not have, so it
  guarded 28 decks while reporting 32 — spin and GPU were never rendered by it.
  Now 34 real decks plus 2 deliberate refusals.
* **`geometry.h_ratio` told a gold slab it was missing hydrogens.** Now
  conditioned on "if this is a molecular system" rather than gated on a
  molecule/crystal test that cannot be made reliably.
* **One name collision in the emitted PySCF program** (`scf` → `_mw_scf`).
* Three dead imports.

## What was landed as new capability

* **The pipeline provenance log** — `molbuilder/pipeline_log.py`, one writer,
  two callers. `jobset prep --pipeline-log`, off by default, **byte-identical
  output either way**. Records what each step received, decided and produced,
  with a source per value. In flat layout each rung gets its own file; a
  sweep's goes to the `bench/` container.
* **The hook boundary** — `issues.calling` over all 16 engine callables.
  Annotates with `add_note` rather than replacing, so the exception type and
  message survive and an engine's deliberate refusal is not buried; always
  re-raises. `issues.notes_of` carries the attribution into `prep`'s
  user-facing error, because `str(exc)` drops notes.
* Contract sections for both, in `docs/execution/script-preparation.md`.

## Things I raised that turned out NOT to be defects

Recorded so nobody spends time on them again. I reported all three as problems
before checking; each took one measurement to settle.

1. **19 PySCF settings written as free text.** I set a non-default value for
   each and rendered the deck: **all 19 reach the generated code correctly.**
   Nothing is broken. It was a preference about how they are written.
2. **Five settings recording the wrong keyword.** The reader that uses those
   records has two callers and **both pass `engine="siesta"`**, whose records
   are all correct. PySCF's readback uses an explicit table. Nothing reads them.
3. **PySCF's settings check emits no hard errors.** True, but there is no
   example of a calculation that should be refused and isn't. An observation.

---

## How to work here

- **Design so the bad state cannot occur, then state the rule. Testing is
  last.** Mutation testing is not evidence about the system — it only says
  whether a test notices damage, which is poking an output one level up.
- **Read the code; do not poke it.** Every problem this session was found by
  inspecting an output. That is the symptom the test cleanup addresses.
- **Check before reporting.** Three of my findings were preferences rather than
  defects, and three more were artifacts of my own measuring: a grep capped at
  three lines that hid the k-grid, a substring check that called a live import
  dead, a harness that pinned a `TypeError` as a deck.
- **Never call a defect "harmless".** The `scf` collision broke nothing on the
  day it was written, which is precisely why it survived.
- **Warn, never decide the user's science.** Stating a fact, or an
  inconsistency between two things the user set, is the job. Judging their
  choice is not — metallicity is a *result*, so nothing may warn that a k-grid
  is too coarse for a metal. A warning that states something FALSE is
  different: that is a defect.
- **The tests here are ours.** Cleaning up redundant ones is part of the work.
- Copy `molbuilder/ tests/ pyproject.toml` (~40 MB) for scratch runs, never the
  whole tree. Never edit source while a test batch is running.
