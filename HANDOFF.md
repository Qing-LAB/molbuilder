# Handoff — committed, and the workflow has now actually been run

## SCOPE

**Structure-optimization tab, SIESTA and PySCF.** `molbuilder/transport/` and
`molbuilder/spectra/` are deferred and byte-identical to `HEAD`. Do not touch,
analyse, or fold them into a finding.

| | |
|---|---|
| suite | **6882 passed, 0 failed** — `python tools/testrun.py run none2e`, ~22 min |
| reference decks | **36 cases, digest `149ac0714089ab85`**, harness `<scratch>/refgen.py` |
| committed | `00d9d8e7` (the 58 files) on `feature/generator-jobset-ui`; nothing pushed |
| E2E | **done 2026-08-19** — both engines ran real water calculations through the browser workflow; see § 2 |
| server | the user runs it; do not start one. `https://qlabsrv.physics.asu.edu:8888` |

---

## What remains, in priority order

### 1. Commit — DONE. One commit in the house shape; this file rides in it.

### 2. The end-to-end run — DONE, and it earned four findings.

Executed 2026-08-19 through the real workflow, nothing hand-driven:
Molbuilder (SMILES `O` → RDKit water, vacuum set 8 Å in Modify → Cell) →
Structure optimization → *Send to Task setup* → Task setup (shape, note,
save) → `jobset prep run coarse --pipeline-log` → `jobset submit run coarse`,
one job per invocation.  All inside `projects/claude-e2e/`; the user's own
projects untouched.

| leg | engine · shape | result |
|---|---|---|
| `optimization/water` | siesta · flat | relaxed in 8 CG steps, max force 0.0022 < 0.02 eV/Å, E = −481.005 eV, O–H 0.974 Å, ∠ 104.3° |
| same, attempt 2 | manual door | `MB_LAUNCHED_BY=manual` → `-run1.out`, warm continue from `.XV`/`.DM`: **1 SCF iteration, 5.5 s** |
| gate check | — | bare non-interactive `bash *.run.sh` refused, exit 2 |
| `optimization/water-pyscf` | pyscf · hierarchical | geomeTRIC converged, E = −76.3589 Ha, O–H 0.967 Å, ∠ 103.1°, molwatch wrote 4 live steps |

The pipeline log was the validation instrument: every `⊕` fact checked traced
to deck text; the render walk, the W10 derived context, the declined-items
line, the W5 recorded nothings, and the 24-line check gate all appear as the
contract says.  Results tab: live plots and trajectories for both runs.

**The four findings — classified (user, 2026-08-19), fixed the same day.**
Natures: 1 design hole, 2 framework bug, 3 explicitly-deferred design item,
4 framework bug (and the "missing lines" half of my original report was my
own truncated read -- the lines were present; the READER dropped them).

1. **Fixed — the `.source` reservation** (`job-contracts.md` § 6.3).  The
   description's structure pair is `<label>.source.xyz` +
   `<label>.source.molstruct.json`; identities are validated dot-free, so
   no engine output can take a dotted name in any shape.  Writers: the
   hand-over and `jobset describe` (which now also records the marked name
   in the travelling `task.json`).  `identity.OUR_FILE_PATTERNS` rows
   updated — a bare `<label>.xyz` at a flat root is the ENGINE's now and
   is reported as run state; `--cold` names it as at-stake.  Old folders
   keep working (readers follow `task.json`).
2. **Fixed — the emitted anchor.**  `_mb_outfile` anchors with
   ``absolute()``: `resolve()` was a flat-era reflex (a6908f1f,
   2026-05-28, when the deck was never a link) that walked hierarchical
   outputs out of `run-<n>/`.  Attempt state now lands in the attempt;
   warm carry stays with prep's sanctioned route (`pyscf/warm-files.toml`).
   Verified live: run-1 holds chk/molwatch/xyz, and `jobset status` lists
   its warm files.  The same residue exists in `spectra/pyscf_script.py`
   — deferred scope, fix it in the spectra pass.
3. **Fixed — status reads the engine-neutral conclusion.**
   `running-a-job.md` § 4 now derives state over RESULT files: every
   `.out` plus each molwatch log whose footer concludes the run; a
   seed (no footer) contributes nothing, so SIESTA behaviour is
   bit-identical.  The dir parser claims molwatch dirs (the B3 deferral
   of 2026-06-19, shipped for status).  Verified live: the PySCF attempt
   reads finished · job_completed.
4. **Fixed — the convergence-key grammar.**  The molwatch reader's key
   regex was letter-first while real stage tokens are digit-first
   (`01_coarse`), so every staged header parsed to an EMPTY target dict
   and the Results card said "not found" over eight present lines.  One
   grammar fragment (`_CONV_KEY`) now feeds both the extraction regex and
   the dispatch rule; the nested-form tests round-trip REAL tokens
   instead of the imagined `stageN` spelling.  The JS card needed
   nothing (read in full: shape-detection is type-based, key-agnostic).

**The follow-up sweep the fixes earned (user, 2026-08-19: "number-starting
name vs stageN was a historical residue -- be sure nothing is left behind, in
comments, code, or documents").**  Swept the whole tree for the retired
spellings (`stageN`, `-stage<N>`, letter-first key grammars, pre-stage stem
strippers) and for the retired trajectory glob.  Live code found and fixed:

* `parse/engines/pyscf.py` — THREE private pre-stage stem-strippers (molwatch
  sibling, Step-0 energy log, SCF-history log) each missed every staged run's
  files; one inverse of the writer grammar now (`_resolve_job_token`), plus
  the module's own letter-first flat-only convergence regex replaced by the
  format owner's one reader (`molwatch.parse_convergence_line`).
* `web/static/lib/inspectors/trajectory.js` — the absorb rule stripped the
  retired `-stage<N>` overlay, so every staged relaxation was five menu
  entries again; now strips the real token and matches both stems.
* `web/blueprints/watch.py` — discovery-chain steps 3–4 spelled only the
  unstaged grammar (`*_geom_optim.xyz` cannot match a staged trajectory);
  masked whenever a molwatch seed exists, live for `write_molwatch_log=False`.
* Tests that round-tripped the imagined spellings (`stage1` keys, `-stage3`
  masters) now round-trip real tokens — they were green against grammar no
  generated file has used since the token rename.
* Teaching text swept to today's grammar (emitter examples, format.py's
  `stage_name` doc, core.js's nested example, runwrap/checkpoint examples,
  the emitted deck's Outputs rows, `engines/pyscf.md` § 4 now specs the
  header/footer lines and the digit-first key grammar, `job-contracts.md`'s
  chain text).  Left alone on purpose: frozen fixtures
  (`hemeC-stage2-run3-finished-42fr.out`), dated empirical citations, user
  tag examples (`stage1-good`), and project-layout's quoted-and-corrected
  review trail — history keeps its era's names; teaching text does not.

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
