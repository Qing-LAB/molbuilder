# The plan — one file

**Role:** plan — **the only one.** Every open item from the nine plan
documents that preceded it lives here; those nine are archived under
`docs/archive/2026-09-01-*.md` as records of what was decided and built.
**Domain:** all
**Started:** 2026-09-01, by consolidation

> *(user, 2026-09-01: "We don't need ten plan files scattered. We want one
> plan folder or one plan file and stay with that file.")*

**Nothing here has been executed.** This file is the merge, fact-checked
against code on 2026-09-01; the next step is agreeing what it should contain
and in what order.

---

## 0. Rule R3, restated

`docs/README.md` used to say **R3: `roadmap.md` is THE one plan.** That is now
this file *(user, 2026-09-01: "the old roadmap and audit should be archived")*,
and R3 reads:

> **`plans/plan.md` is the one plan. Every open item lives there, and nowhere
> else. A document that finds work records the evidence and sends the item
> here.**

That roadmap and the two audit reports are archived under
`archive/2026-09-01-*`. They were 2044 lines between them, and the fact-check
below is why the merge was worth doing rather than the filing.

---

## 1. What the fact-check found

Nine plan documents, read in full and checked against the code on 2026-09-01.
Three headers were **flatly false** and are corrected in the archived copies:

| document | claimed | actually |
|---|---|---|
| `modify-redesign-plan` | *"items 3 and 4 designed"* | all five built — § 3 carried its own **Built 2026-08-30** marker three sections lower |
| `css-system-plan` | *"proposed, not started"* | steps A and B done **2026-08-02**, four weeks earlier, ticked in its own § 4 table |
| `config-access-plan` | header *"steps 1–4 built"*, body ***"No code yet"*** | one document, two answers; steps 1–4 built and step 5 mostly |

**Two were in no index at all** — `consolidated-cleanup-plan` and
`config-access-plan` existed on disk and appeared nowhere in `toc.json`, so
they were invisible in the Documents tab. Fixed 2026-09-01.

**One item I got wrong first, corrected by reading the function:** bench § 2.2
looked built on a grep for `cpu_mean_pct`; `parse_util_bound` takes only the
*verdict*, and its docstring says the numbers on that line **"are deliberately
NOT read here any more."** It is open.

---

## 2. Open — engine, execution, science

| # | item | from | state |
|---|---|---|---|
| **E1** | **Benchmark iteration count, settable per calculation.** No field exists on `task.json` or `Resources` | `bench-and-junction` § 2.1 | not started |
| **E2** | **Read the monitor's own mean — VERIFIED 2026-09-03, and it is a design call, not a cleanup.** The premise is right: `util_accum.add` runs on EVERY tick so `[UTIL-SUMMARY]`'s mean is exact, while `util.csv` is change-gated (a row only when a metric moves ≥ 10% or a 300 s keepalive elapses), so `parse_util_csv`'s time-weighted mean is a reconstruction over a subset. **The false sentence is fixed**: the docstring said the summary is *"a digest of the same samples util.csv records raw"* — the csv is not raw, it is gated. But the other two reasons still hold, and one is decisive: the summary is written only in the monitor's terminal branch, so it is **absent exactly when a trial was killed** — the case a benchmark most needs to read. So this is *prefer the summary, fall back to the csv*, weighed against the two-readers-of-one-fact lesson that removed it in the first place. Wants your call | `bench-and-junction` § 2.2 | needs a call |
| ~~**E3**~~ | **DONE 2026-09-03 — named, with the tell that detects it.** `wall_s` is the monitored window, not job wall time, and the docstring claimed the opposite (*"the monitor runs for the life of the job"*). Both ends are anchored only when the monitor reaches its terminal branch — a KILLED trial ends at the last change-gated row, up to a 300 s keepalive short and further if the metrics had gone flat. The same branch writes `[UTIL-SUMMARY]`, so **its absence is the flag** that `wall_s` is a lower bound; the two fail together and the reader gets that for free | `bench-and-junction` § 3B | done |
| ~~**E4**~~ | **DONE 2026-09-03.** The *node name and CPU model* half was already there — the monitor's `[MACHINE]` line is written first, unconditionally, on every path. What was GPU-only is the wrapper's own TOPOLOGY reading (`phys_cores` / `n_sockets` / `cores_per_socket`), and that is the one `bench/result.py` reads into `node_phys_cores` — the field written so a sweep landing on different node types can say so (Au-BDT-Au ran 2x24 while the record said 2x32). A CPU sweep could never fill it. The probe now REPORTS what it measures and is hoisted out of the GPU block, so every path carries the line exactly once and the GPU echo keeps only `mps_available`. Pinned by `test_the_node_it_ran_on_is_recorded_on_EVERY_path`, mutation-tested | `bench-and-junction` § 3D | done |
| **E5** | **Two facts need a Sol run**, not code: a real `[MACHINE]` line from a live job, and a re-probe to settle `lightwork`'s cap | `machine-identity` § 6 | blocked on a machine |
| **E6** | **`default_contact_distance` has no consumer.** Measured physics (Au–S 2.40, Pt–N 2.05, Ag–S 2.50) with nothing reading it since `add_electrode_slab` went. Wiring it in as `--electrode @contact=`'s default would make it live again — a decision, not a cleanup | found 2026-09-01 | open |
| **E7** | **D7's cluster half** — the same prep→submit→watch loop through SLURM on Sol. Everything else about D7 shipped; this needs the machine, and pairs with **E5** in one session | roadmap § 1 | blocked on a machine |
| ~~**E8**~~ | **ALREADY DONE — verified 2026-09-03, not changed.** `_CHECKPOINT_DEFAULTS` carries `pyscf: ["*.chk", "*.cube"]` (landed with the classification move, c818274c) and it resolves: a 1-byte `job.chk` is archived by the hint, a 20 MB `job.log` by measurement. The globs are a SPEED hint, never a filter — anything they miss is still measured against the size limit, which is why `is_big` says an unknown engine is *"merely slower"*. So there was nothing here that could be wrong, only slow | roadmap § 1 | done |
| ~~**E9**~~ | **BOTH HALVES CLOSED 2026-09-03.** **C8**: the third `BlockSize` state went on 2026-08-15 and its residue is deleted — a comment asserting three states beside one asserting two, and a PROVENANCE arm that would have LIED (with `0` the emitter writes `BlockSize 0`, because 0 is not None). **C12, by your ruling**: the atom-count clamp is *removed*, not re-based on the species count. It was unscientific — *"whether mpi is too big for a system is none of your business... we had that problem because of a psml related issue, not a size issue"* — so all four sites are gone (the CPU auto path, the GPU rank default, the user-set WARNING, and `auto_ranks`, which was lowering the `.sbatch` header's `--ntasks` too). What replaces it is a NOTICE on the objective number: `n_orbitals / mpi_np`, wanted > 1, checked at run time against the rank count actually resolved, showing both numbers and saying *"your CPUs are not going to be fully used... this is a notice, not a limit"*. The post-run `propor` hint now leads with the pseudopotential, which is the cause it already named second | roadmap § 6 | done |
| **E10** | **Detecting from the `.out` that a run STOPPED**, and whether it stopped converged (`SCF_NOT_CONV` / `ABNORMAL_TERMINATION`) — **on a zero exit too**, since an MPI stack may not propagate an abort. Ruled to belong to `mb_monitor.py`, not the execution branch | roadmap § 4 | open |
| **E11** | **A fresh live walk of the PySCF / spectra decks.** The 2026-08-28 review exercised them only through the guard suites and says so | audit 08-28 § 5 | open |

## 3. Open — configuration and ops

| # | item | from | state |
|---|---|---|---|
| ~~**C2**~~ | **DONE 2026-09-03 — measured on a real built wheel: 90 of 141 static files.** Missing were the whole MolView module (13), the structure page (9), VibrationView (5), and the Task-setup, Documents, Transport and This-machine assets — every directory added after the hand-kept glob list was last edited. The list is replaced by one recursive pattern (141/141 after rebuild) and the setuptools floor raised to 62.3, the version that supports it. `tests/test_wheel_ships_the_front_end.py` guards both halves — every static file is packaged, and every asset a template names exists — and carries the trap that made the FIRST measurement read zero missing: `fnmatch`'s `*` crosses `/`, so `vendor/*/*` appeared to cover `lib/molview/model.js`. Mutation-tested (120 flagged on a partial list) | roadmap § 4 | done |
| ~~**C3**~~ | **DONE 2026-09-03 — deleted.** The package was 35 lines re-exporting `molbuilder.builders.backends`, kept *"for external callers"*; and `build.py` carried a comment saying in-tree code goes direct. **Both halves were false**: `nucleic.py` and `web/app.py` imported the shim by RELATIVE path (`from .backends`, `from ..backends`), which an absolute-name grep does not see — the comment describing the rule was the only thing obeying it. 19 test references, two production imports and two module docstrings now name the real package | roadmap § 4 | done |
| ~~**C1**~~ | **DONE 2026-09-03 — and there were more than three.** `runtime_config` hand-joined the PROJECT scope's filename at **seven** sites (the machine scope had a door and this one did not) and `scheduler/record` joined `environment.json` at **three**. Two of each set are inside REFUSAL MESSAGES, where a join that drifts prints a path the reader cannot find — worse than no message. Both scopes have a door now: `_project_config_file()` and `record.calculation_record()`. `monitor`'s two are still **not** residue — that module ships to a compute node with nothing else importable and owns the format (A11) | `config-access` § 5 | done |

## 4. Open — the front end

| # | item | from | state |
|---|---|---|---|
| **W1** | **The document tier (step C).** `html, body`, `header`, `button`, `footer`, `textarea` genuinely differ per page; the `*` reset is already deleted. Blocked on a browser pass over all pages | `css-system` § 4C | partly |
| **W2** | **One home per component (step D).** `.card`, `.status`, `header .tagline`. **One value to settle first:** `.card`'s padding is `var(--space-md) 18px 18px` and 18 is off the 4px grid the contract declares — moving it shifts every page by 2px | `css-system` § 4D | not started |
| **W3** | **Per-page token/namespace passes (step E)**, one page per commit: `spectra`, `structure-optimization`, `transport`, `results`, `documents` | `css-system` § 4E | partly |
| **W4** | **Guards 1 and 2 (step F)** — one home including elements; a page sheet contains only its own tier. Guard 3 landed; a fourth (no long block copied between sheets) landed 2026-09-01 | `css-system` § 4F | partly |
| **W5** | **The inspectors module's appearance still lives in `results/style.css`** — 70 mentions. Mount an inspector on another page and it renders unstyled. Three of its sheets have been repatriated; the rest is a module change, deliberately | `css-system` § 7.0 | partly |
| **W6** | **The editor module.** CodeMirror on three surfaces, owned by nothing: **three** loaders, **three** sheets theming `.CodeMirror` (40 rules — up from the 30 the plan measured), and the hard-won caps (1500-line selection, 1 MB view-only) on **one** surface. Not started, deliberately sequenced after the Task-setup workflow | `editor-module` | not started |
| **W7** | **I7 close-out** — the browser walk of Results export → cite → describe → prep. I1–I6 done | `structure-info` I7 | open |
| **W8** | **Caller-less endpoints — decide the ROLE, do not just delete.** `/api/docs/list` (a second answer to "what docs exist" beside the `/api/docs/toc` the tab calls); `/api/checkpoint/config` (the read half of a route whose write half was removed); `/api/selection/atoms` (pinned by five test files) | `structure-info` § 3 | open |
| **W9** | **Transport viewer default orientation** — a MolView camera-door contract question | `structure-info` § 3 | open |
| **W10** | **Results transmission inspector** — the record exists, the reader does not | `structure-info` § 3 · roadmap § 2 | open |
| **W12** | **The live browser walk-throughs** — checkpoint swap at narrow and wide widths, per-tab reload round-trips, a real Data/Image export, click-selection on frames ≥ 1. Carried since the archived molview-and-checkpoint plan | roadmap § 0.3 | open |
| **W13** | **384 raw px/rem literals** in the page sheets. The 2026-08-28 count was 777, so this halved without being finished; it is the tail of 7.4c | roadmap § 7.4c | partly |
| **W14** | **A web plan view and a per-stage status roll-up.** The web *describes* a staged calculation (Task setup) and *observes* runs (Results); neither shows the plan as a whole | roadmap § 6 | open |
| **W15** | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. Plus routing the CLI through the shared codec and exercising the last annotation-channel kind | roadmap § 3 | partly |
| **W11** | **The trajectory inspector's export records no contract.** Its load rides `/api/watch`, not the structure door, so an export from the trajectory view carries no `info.calculation` | `structure-info` § 3 | open |

## 4a. Open — needs a decision from you

| # | item | from | state |
|---|---|---|---|
| **N1** | **What may run in a request thread.** The GPU half is closed (every driver read is now a timed subprocess behind a cache); the general case — RDKit embeds, giant parses — still runs in request threads and can freeze every user. Wants a contract sentence, not a patch | audit 08-28 O4 | needs a rule |
| **N2** | **Ship-or-retire batch**, named in design and never built, with no recorded retirement: the checkpoint tail (`prune`, a CLI `checkpoint diff`, the `snippets/` library, wrapper-git "Path B"), and #32's MD viewer/editor | roadmap § 4 | needs a call |
| **N3** | **Three wording papercuts:** the prepped-trials list prints `run-0, run-0` without saying whose attempt each is; the science-gate warning repeats once per trial (noise at 16); a scheduler-less `ask` says *"nothing to wait for"* and then prints an `sbatch --test-only` preview | audit 08-28 O5 | open |
| **N4** | **The science-validation tail** — checks deferred with recorded rationale in `science/pseudopotentials.md`. The roadmap carried them under a heading that said *"needs a home"* for five weeks | roadmap § 5 | needs a home |

## 5. Open — documentation drift

| # | item | from | state |
|---|---|---|---|
| ~~**D1**~~ | **DONE 2026-09-03 — closed, and now guarded.** All 22 resolved: the pointer moved where a live section owns the rule — the web-api envelope's "1.6" is now § 1's *Status codes*, the job-system prep report's "2.3.3" is § 5.3, and the submission memory ceiling's "8" belongs to `scheduler.md` § 2 — and where no live section did, the rule was written first — `stages.md` § 6.8 now carries the bench lane's SIESTA-only gate and the pin set that forces it. Two were not stale pointers at all: `task.py` still carried the comment block for `bench_allocation`, a key deleted on 2026-09-01 with the disclosure that wrote it, and `_cli.py` "quoted" § 5.3 in words the document does not use. `test_docs_structure.py::test_every_section_citation_in_code_resolves` is the new guard (mutation-tested); like its doc→doc sibling it had to phrase its own examples so it does not read them as citations | `consolidated-cleanup` § 7 | done |
| **D2** | **Tests with no target, remainder.** 12 removed; ~14 flagged. The two files with zero test functions were **checked and left** — each is a signpost recording where retired coverage moved, which is a service, not residue | `consolidated-cleanup` § 9 | partly |
| ~~**D3**~~ | **DONE — and guarded.** `tests/test_no_tests_read_the_projects_tree.py` fails on a test that reads the real tree, and on a test that SKIPS ITSELF when a projects fixture is missing (the shape that turns "isolated" into "never ran"). The tree is clean after a full run | `consolidated-cleanup` § 9 | done |
| **D4** | **The README screenshots are three tabs stale** — five captured, eight ship. Nothing can enforce this (no test can count tabs in a PNG); the *owner* of the count is pinned as of 2026-09-01 | `screenshots.md` | open |

## 5a. What the roadmap fact-check found

1770 lines, read against the tree on 2026-09-01. **Most of what it carried as
open had shipped**, which is the same drift it was written to prevent — and it
says so about itself twice, at § 7.4's *"carried as open for a day after they
shipped"* and § 7.5's *"re-read against the tree and most of it was already
gone."* A third re-read was overdue.

| roadmap item | claimed | verified 2026-09-01 |
|---|---|---|
| § 7.5 **O5 residue** — `_SAFE_BASE` / `_SAFE_GPU_TYPE`, "two definitions, zero uses" | open | **gone.** Both symbols are absent from the tree |
| § 7.8 **wave 1** — the same two, plus the `task.stages and` conjunct at `_cli.py:569` | *"the live plan"* | **done.** No symbol survives |
| § 7.10 **bundle layout** — nothing rendered at the bundle root | *"plan of record"* | **built.** `project-layout.md` records *"nothing rendered sits at the root since 2026-08-24"* |
| § 7.11 the two asks reach the browser | — | **done**, and says so |
| § 3 **A4** — the disk-based selection endpoints, *"verify the live caller before deleting"* | open, caller named | **the caller is gone.** `_fetchAtoms` is absent from the MolView store; `/api/selection/atoms` is now caller-less — folded into **W8** |
| § 7.4c **anonymous values** — 777 raw px/rem literals (2026-08-28) | open | **384.** Halved, not finished — **W13** |
| § 4 **`molbuilder/backends/`** — the last no-shim violation | closed | **deleted 2026-09-03** — **C3** |
| audits' **O1 O2 O3** (conclusion marker · supervisor respawn · stack-dump hook) | closed in the audit itself | closed, consistent |

**What that leaves.** Of roughly forty items the roadmap presented as work,
the ones that survive contact with the code are the twenty-odd rows above in
§§ 2–5. The rest were closed and never struck.

---

## 5b. Open — found 2026-09-02, the run-decision round

Two code-review agents and six test-audit agents, every acted-on claim verified
by hand. **Priority is the first column and means: P0 misleads a person right
now; P1 is a check that silently is not checking; P2 is a gap in what shipped
today; P3 is cleanup, large and mechanical.**

### The front end — what a person is told to run

| P | # | item | evidence |
|---|---|---|---|
| ~~P0~~ | ~~**R1**~~ | **DONE 2026-09-02.** `--from` is composed in the browser and was wrong for `flat`. `viewer.js:1872` builds `01_coarse/run-0`; `Shape.stage_dir` returns `"."` for flat, so the taught command names a directory that does not exist. It also hard-codes `run-0`, so a stage with three attempts is taught to continue from the first. `/api/task-setup/prep-plan` already returns the correct `token` and `dir` | **this is V2 of the 2026-08-13 program, resurfaced.** Closed on the CLI, never on the browser |
| ~~P0~~ | ~~**R2**~~ | **DONE 2026-09-02, and UNVERIFIED** — no harness drives the stage panels, so the fix is read-checked only; a test needs **B3** and is tracked with **M2**. The Prep-run button was withheld from every continuing rung (`viewer.js:1935`, `if (!from)`) — and with it the A13 emitted block, from exactly the long runs A13 exists for | verified |
| ~~P2~~ | ~~**R3**~~ | **DONE and PROVEN 2026-09-02.** Page state survived a folder change, against `task-setup.md` § 2.1's *"the page holds no state of its own"*: `_extraRunRows`, `_pendingDrop`, `_queue`. `_resetPerFolderState()` now runs at the top of `loadFolder`, before the branch, and clears all of them; pinned by `test_a_row_added_on_one_folder_does_not_follow_you_to_the_next` (mutation-tested: removing the one `clear()` fails it) | verified |
| ~~P2~~ | ~~**D8**~~ | **DONE 2026-09-02.** **`task-setup.md` § 7 disagreed with the backend about where a run's `time`/`domain` go.** Its run-card row said *"a machine item becomes the launch shape, anything else a pin over the template"* — false for the two lane asks, which are neither (`_declared_execution_pins` returns them as neither pin nor axis; `declared_run_shape` takes them by name). Its `allocation` row did not mention that a run may state its own wall and queue. § 6.2b never mentioned the two rows the card actually offers, and the three row states (`chosen` / `inherited` / `unstated`) were styled but nowhere stated. All four fixed; `stages.md` § 6.8e, `architecture.md` § 5.2, `task.py`, `validation/task.py`, `_cli.py` and `viewer.js` already agreed and were left alone | verified against all six surfaces |
| **P2** | **R4** | **Three writers, one buffer, a 400 ms loss window.** `syncFromModel` writes from `_task`; `applyAsksToDoc` and `applyNotifyToDoc` patch the buffer and never touch `_task`. Type a memory value, blur, click "+ Add stage" inside the debounce and the `allocation` block is gone | verified |
| ~~P0~~ | ~~**R6**~~ | **DONE 2026-09-02.** **Naming `(this machine)` discarded the calculation's own `environment.json`.** `machine_for(bundle, target=LOCAL_TARGET)` took a private road reading only the machine scope, never walking `record_scopes` — so the explicit road and the silent one resolved different records for one box, and with no machine-scope file prep refused in *remote* words. The browser can only send the label, so **every local prep from the tab** took it. Fixed in the reader (`record.py`), rule written into `running-a-job.md` § 3.1, pinned by `test_target_machine_choice.py` + the new e2e | found by `test_task_setup_prep_e2e.py`; user: *"all environments have to be explicitly probed and stored. no environment json, error"* |
| ~~P1~~ | ~~**R7**~~ | **DONE 2026-09-02** *(user: "safety first … compatibility is not an issue. So just fix it")*. `prep` no longer probes: step 1 READS a record or refuses, naming `jobset probe --write`. Swept `project-layout.md § 2.3.1`, `workflow.md § 5`, `job-system.md` and `template.md`, which all drew the probe as part of prep. The suite's own box is now probed by a conftest fixture (`write_machine_record`), because a probed machine is a **precondition** rather than something prep arranges — and `config_root_is_never_the_developers` moved off `tmp_path`, which had been putting a config directory inside trees that tests walk | pinned + mutation-tested |
| ~~P1~~ | ~~**T4b**~~ | **DONE 2026-09-02.** The flat top-level `cert`/`key` spelling of `tls` is **removed**, not folded — refused by name with the section to write, because a config that quietly lost its certificate falls back to plain HTTP and the failure mode is *the site still loads*, unencrypted, with nobody told. Three tests describing how the two spellings got along are retired; one refusal test replaces them | pinned |
| **P3** | **R5** | **`"(this machine)"` means `LOCAL_TARGET` at the prep door and `None` at the bench-grid door.** Real asymmetry, but **the fix is not unification** — `None` is what lets the reader prefer the bundle's own snapshot, and forcing them together broke a live GPU test. The narrow gap: on an unprepped folder with named records, both fit blocks 400 and hide themselves. Fix the *surfacing*, not the value | tried and reverted 2026-09-02; the reasoning is in the code |

### Tests that are not testing

| P | # | item | evidence |
|---|---|---|---|
| ~~P1~~ | ~~**T1**~~ | **DONE 2026-09-02 — by making the claim TRUE, not by editing it away.** `§ 2.3` (SIESTA force tolerance, three tiers) is now registered in `TIER_TABLES` against `SIESTA_STAGE_PRESETS`; mutation-tested. The note also pointed at the wrong sections: § 2.4 is geomeTRIC's and has no SIESTA column, § 2.5's SIESTA column is one global rather than a ladder. Was: `tuning.md:919` claimed the SIESTA tier tables are checked. They were not — only the two PySCF tables are registered in `TIER_TABLES`. There is no SIESTA doc-parity check anywhere | verified |
| ~~P1~~ | ~~**T2**~~ | **DONE 2026-09-02.** Resolves from `__file__` like every other path in the file, and asserts the source list is non-empty so a future blind run says so. Mutation-tested **from `/tmp`**, which is where the old form proved nothing. Was: **`test_layering.py:476`** uses a relative `Path("molbuilder").rglob(...)` where every other path in the file resolves from `__file__`. Under any cwd but the repo root the set is empty and the test passes proving nothing | verified |
| ~~P1~~ | ~~**T3**~~ | **DONE 2026-09-02.** Names `dataclasses.FrozenInstanceError`; mutation-tested by un-freezing the dataclass. Was: **`test_diagnostics.py:175`** — `pytest.raises((AttributeError, Exception))`; the second member subsumes the first, so any exception passes | reported, spot-checked |
| **P1** | **T4** | **THREE OF FOUR DONE 2026-09-02.** ✅ `Config = SiestaConfig` **deleted** — alias, both `__all__` entries, the two docstring examples that taught it, and the test, together (its only callers were those). ✅ the gcc pin: the test asserted the substring `gcc_linux-64=14`, which **`14.4` satisfies as well as `14.3`** — and 14.4's gfortran miscompiles SIESTA's `kpoint_t.F90` into wrong k-points, so the one thing the pin exists to prevent was indistinguishable from success; now a property check (three packages, one version, minor present), mutation-tested through `MOLBUILDER_GCC`. ✅ the envelope test: rewritten to the property that is still true (a stray top-level key changes nothing, **ignored not refused**, because a request body is not a config file) — `struct_from_body`'s stale docstring head, which still led with the retired flat shape as *canonical*, fixed with it. ⛔ **`_FLAT_ALIASES` is NOT a code shim and I did not remove it** — `cert`/`key` is a **config-file format users have on disk**, and the loader refuses unknown keys, so deleting it stops their server booting. The no-shims rule is about renames in code; this is a migration and needs your call. Was: **Four tests actively block a correct change**: `test_review_fixes.py:237` (`assert Config is SiestaConfig`) and the three `runtime_config._FLAT_ALIASES` tests pin **backward-compat shims** against the project's no-shims rule; `test_envs_siesta_gpu_recipe.py:89` pins `gcc=14` where `installation.md:202` reverses it; `test_structure_envelope_protocol.py:87` pins a deleted legacy branch — **and that one needs the doc fixed first**, since `web-api.md` still claims `/api/modify/*` accepts the old flattened shape | verified |

### Test bloat — measured, not estimated

**163 test files added, 36 deleted since 2026-08-01.** Whole files *do* get
retired; what never happens is pruning inside a surviving file — one commit
added 19 test definitions to `test_checkpoint_states.py` and removed none,
several of them written to *replace* pieces left standing. **That single habit
is 47 of 77 findings in one partition.**

**The rule this earns, and the only one that prevents the next round:**

> **A consolidating test deletes the pieces it consolidates, in the same
> commit.**

| P | # | item | size |
|---|---|---|---|
| ~~P3~~ | ~~**B1**~~ | **WRONG, and corrected 2026-09-02 — the tests stay.** This said 58 tests pinned an archived design proposal, because the vocabulary (`fileState`, `fetchSeq`, `uiPrefs`) appears in no live document. It is the **shipped code's own structure**: `lib/trajectory/core.js` is 3,212 lines built exactly this way, `fileState` alone appears 54 times in it, and `results.md` § 4 describes the same four buckets in prose. Retiring them would have deleted the ONLY enforcement of § 4. What was true: the names had no live home and the `§ 13` citations pointed into `archive/`. Fixed by writing § 4.1 (the two-tick settle, a real behaviour with no live home), naming the buckets in § 4, adding both files to § 10's test map, and re-anchoring the citations. **The lesson is about the survey**: a vocabulary search across documents cannot tell a retired design from an undocumented one, and the two want opposite actions | investigated in code + contract, not grepped |
| **P3** | **B2** | **PARTLY DONE 2026-09-03 — and the number was the wrong instrument.** Of the four shapes named here, only one is mechanically decidable: a `Test*` class whose body is a docstring collects nothing. Five existed. **Three were empty promises** — `TestBuildSiestaHonorsSidecarFrozenAtoms`, `TestWorkspacePayloadRegionsAndFrozen`, `TestGenerateWritesToWorkspace`, each stating in the present tense that it pins something (*"Tests pin both layers"*) while holding no test, so a reader scanning for coverage reads yes. Each is replaced by a pointer at the file that DOES cover it. **Two are deliberate retirement markers** that say so and name their successor — the same call D2 already made for two zero-test files. The other three shapes do not survive measurement: `assert len(X) == 5` where the test BUILT X is a real check, and `m = re.search(...); assert m` is a precondition with the real assertions after it. **A list of ~45 that cannot be re-derived is not a finding anyone can act on** — what is left needs the file-by-file read, not a regex | 5 measured |
| **P3** | **B3** | **233 front-end tests assert substrings of `.js`/`.css`**, one pinning six spaces of column alignment. The cure exists at the deck layer (`tests/_deck.py`, written after padding broke 45 tests across eight files) and the front end never got it. **Build the equivalent first, then convert** — deleting these loses real coverage | 233 |
| **P3** | **B4** | **MEASURED 2026-09-03; the envelope half is done, the fixture half is proposed and NOT applied.** The `_envelope()` count was seven, and only **three** were re-implementations: `test_pseudos.py` and `test_task_setup_tab.py` hand-listed the envelope's fields (so a field the envelope grows would never reach them) and both now go through the one builder — which immediately surfaced a real defect: a test built a 2-atom envelope and overwrote `elements` to three, leaving `atom_names` describing the old atoms, and the route's own guard caught it the moment the canonical dict was used. The third was `test_structure_envelope_protocol.py`, carrying TWO docstrings back to back (the second was dead). The remaining four are a delegating alias and one-line `struct.to_dict()` calls — not the hand-rolled XYZ parsers the helper was written against. **`flask_server`: DONE 2026-09-03, without touching a single scope.** 18 of the 20 now call one context manager, `tests/support/live_server.py::serve()`; each module keeps its own `@pytest.fixture(...)` line, because a scope is a decision about how much state a file's tests share and a de-duplication does not get to change it for them. ~230 lines and 18 now-unused `import threading` go with it. The two left alone pass a non-default app config, which is a real difference. **`_node_esm`: 24 of 47 `*_js.py` files drive it** (the row said 7 of 48), and 13 more shell out to `node` themselves | 3 done · 16 proposed |
| ~~P3~~ | ~~**B5**~~ | **DONE.** The shadowed definition is gone, and `tests/test_no_test_is_shadowed.py` makes the next one fail instead of disappearing | 46 lines |

### Gaps in what shipped today — mine

| P | # | item |
|---|---|---|
| ~~P2~~ | ~~**M1**~~ | **DONE.** `test_task_notify.py` carries seven: the round-trip, the two-state absent-vs-empty rule, a selection alone as a policy, the by-name refusal of a non-field, the refusal naming what DOES exist, dedup-in-order, and the two vocabularies being one list |
| ~~P2~~ | ~~**M2**~~ | **DONE 2026-09-02**, and NOT the way this row expected. It said the honest test was a node harness, blocked behind **B3**. It was not: `tests/test_task_setup_prep_e2e.py` drives the real card in a real browser — four tests covering the whole chain (type · Save · Preview · Prep · read the `.sbatch`), the three row states, persistence across a tab change and a reload, and § 2.1's no-state-of-its-own. Every one mutation-tested. A source-grep harness would have proved less and cost more |
| ~~P2~~ | ~~**M3**~~ | **DONE — and the fix was the OTHER direction.** `chosen` is not dead: it is the launch shape a one-point `bench` entry states, and the transport arm was the caller failing to forward it. Both call sites pass it now; deleting the parameter would have deleted the feature |

### Documents that state something false

| P | # | item |
|---|---|---|
| ~~P0~~ | ~~**D4**~~ | **DONE 2026-09-02** — and the function's OWN docstring was stale the same way, still describing the both-keys rule for a second shape that no longer exists. `web-api.md` said `/api/modify/*` *"still accepts its old flattened columns"*. It does not — `_shared.py` raises without the envelope, and the sibling test says *"THE LEGACY SHAPE IS GONE"*. Blocks **T4** |
| ~~P1~~ | ~~**D5**~~ | **DONE 2026-09-02** — and the gap the wrong number hid is now named in the document itself: 7 of 48 `*_js.py` files drive `_node_esm`, which is **B4**'s backlog, so the paragraph no longer describes the intended state in the present tense. Was: **`process/testing.md` § 4** says the `*_js.py` tests drive `tests/_node_esm.py`; 41 of 48 do not. Same file: *"~275 test files"*, actual 419 |
| ~~P1~~ | ~~**D6**~~ | **DONE 2026-09-02.** Six labels, `buffer` included, in § 6.6 and § 13.3; no test hard-coded the count, so only the document was wrong. Was: **`web/molview.md`** § 6.6 / § 13.3 say **five** predefined labels; there are six — `buffer` is real and documented in `engines/transport.md:182`. Fix the doc, keep the test |
| ~~P2~~ | ~~**D7**~~ | **DONE — the TEST was wrong, and its exemption was protecting something else.** Both now say the same thing: no token definition outside `tokens.css`, prefixed or not. The exemption was written for a per-file namespaced palette that had not existed since 2026-06-13; what it actually protected in the 2026-09-02 tree was three tokens with NO prefix at all, one of them labelled *"inspector-only token additions"* |

### The 2026-08-13 V-program — closed, with one survivor

Task #11 carried V1–V24 from `memory/project_final_fix_program.md`. Spot-checked
against the tree on 2026-09-02: **V1, V3–V7, V19, V22, V24 verified closed**
(one `_stage_bench_dir` owner; `continue_retries` on `Resources`;
`validate_ladder` gone; the config split resolved; floor-2 gates filter on field
metadata; the named dead symbols absent; `bench/` is the unified jobset stack).
**V2 is open and is R1 above** — closed on the CLI, never on the browser, and
found again by a fresh agent three weeks later. The tier-2 and tier-3 items
(test-logic, doc sweeps) are superseded where they overlap by §§ 5b's test and
document rows, which were measured against the current tree.

---

## 6. Closed by consolidation — what was archived, and why

| document | why it is a record now |
|---|---|
| `modify-redesign-plan` | all five items built, plus § 3.4 and § 3.4a/b's removals. Nothing open |
| `transport-design` | *"graduates to a contract when built"* — it did; `engines/transport.md` is `Role: contract` |
| `machine-identity-plan` | all seven pieces built; the two remaining facts need a machine, and are **E5** above |
| `bench-and-junction-plan` | ~85% built history; its four open items are **E1–E4** |
| `structure-info-plan` | I1–I6 done; its open items are **W7–W11** |
| `config-access-plan` | steps 1–4 built, step 5 mostly; the remainder is **C1** |
| `css-system-plan` | steps A and B done; C–F are **W1–W5** |
| `editor-module-plan` | not started, and its design is worth keeping whole — **W6** points at it |
| `consolidated-cleanup-plan` | 10 of 12 items done; the rest are **D1–D3** |
| `roadmap.md` | R3's old home. 1770 lines, ~85% closed work never struck; its live items are **E7–E11, C2, C3, W12–W15, N1–N4** |
| `archive/2026-09-01-audit-2026-08-21-fullstack-review.md` | its open list became roadmap § 7.5, and § 7.5 is now empty — verified |
| `archive/2026-09-01-audit-2026-08-28-full-review.md` | O1–O3 closed in the document; O4's general case is **N1**, O5 is **N3**, its uncovered lane is **E11** |
