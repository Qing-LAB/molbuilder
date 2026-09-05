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
| ~~**E2**~~ | **DONE 2026-09-03 by your ruling — *use the exact one when it's there, fall back to the csv*.** `parse_utilisation` is the one door: the monitor's `[UTIL-SUMMARY]` means (averaged over EVERY tick) where it wrote them, `util.csv`'s time-weighted reconstruction (a ≥10%-change-gated subset) where it did not — which is what a KILLED trial leaves, the one a benchmark most needs to read. The two-readers-diverged lesson that removed the summary in 2026-08-19 is answered by putting the choice in ONE function rather than at each call site, and by `util_basis` on the record naming the source, so a reconstruction is never mistaken for an exact figure. Peak RSS, wall and peak VRAM stay the csv's — no summary carries them. The fixture makes the two sources differ by three points, so the tests cannot pass by reading either one | `bench-and-junction` § 2.2 | done |
| ~~**E3**~~ | **DONE 2026-09-03 — named, with the tell that detects it.** `wall_s` (renamed `monitored_elapsed_s` on 2026-09-05, P-T1) is the monitored window, not job wall time, and the docstring claimed the opposite (*"the monitor runs for the life of the job"*). Both ends are anchored only when the monitor reaches its terminal branch — a KILLED trial ends at the last change-gated row, up to a 300 s keepalive short and further if the metrics had gone flat. The same branch writes `[UTIL-SUMMARY]`, so **its absence is the flag** that the figure is a lower bound; the two fail together and the reader gets that for free | `bench-and-junction` § 3B | done |
| ~~**E4**~~ | **DONE 2026-09-03.** The *node name and CPU model* half was already there — the monitor's `[MACHINE]` line is written first, unconditionally, on every path. What was GPU-only is the wrapper's own TOPOLOGY reading (`phys_cores` / `n_sockets` / `cores_per_socket`), and that is the one `bench/result.py` reads into `node_phys_cores` — the field written so a sweep landing on different node types can say so (Au-BDT-Au ran 2x24 while the record said 2x32). A CPU sweep could never fill it. The probe now REPORTS what it measures and is hoisted out of the GPU block, so every path carries the line exactly once and the GPU echo keeps only `mps_available`. Pinned by `test_the_node_it_ran_on_is_recorded_on_EVERY_path`, mutation-tested | `bench-and-junction` § 3D | done |
| **E5** | **Two facts need a Sol run**, not code: a real `[MACHINE]` line from a live job, and a re-probe to settle `lightwork`'s cap | `machine-identity` § 6 | blocked on a machine |
| ~~**E6**~~ | **DONE 2026-09-03 by your ruling — a reference, not a default.** *"We abandoned the way how a junction is constructed by using bond distances… we can give this information as a reference though."* The supplier trio (`default_contact_distance`, `_load_contact_distance`, `_get_contact_distance`, 78 lines) is deleted with the question it answered — metal is added by hand, a slab is placed by its z offset. The measured table lives on in MolView's measurement readout: pick two atoms of a known pair and it shows the literature distance, **the number alone**, at any separation, labelled in the table's metal–anchor order. The anchor is half the key (Pt–N 2.05 vs Pt–S 2.30) — which the retired lookup dropped, and why the table's own `anchor` and `note` were unreachable through it | found 2026-09-01 | done |
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
| ~~**N1**~~ | **CLOSED 2026-09-03 — the rule is written and there is nothing to fix.** *"Anything over three seconds runs as a job, show progress"* (`web-api.md` § 1a), and *"we never see this as a problem"* — a small-lab server does not get concurrent heavy requests. The measurement is kept so nobody re-derives it: an RDKit embed is 4.5 s and costs one thread (it releases the GIL); a 25 MB SIESTA `.out` parse is 4.7 s and holds the GIL. One false claim was corrected on the way — `_refresh_if_changed` said dropping its lock kept other requests unblocked, and releasing a lock lets a request ENTER, not RUN | audit 08-28 O4 | done |
| ~~**N2**~~ | **ALL NINE RETIRED 2026-09-03** *(user: "retire all of them")*, each recorded where a reader would look for it rather than in a list nobody opens: `checkpoint diff`, `prune`, the `snippets/` library and wrapper-git "Path B" in `running-a-job.md` § 6.5 (whose heading is now *what is not built, and what is not going to be* — two items there ARE still open and now say so); the `beforeunload` discard guard in `web/runtime.md`; the PySCF BENCH-MARKS block in `job-contracts.md` § 3.1 + § 3.3, where it was twice called a *gap*; `--warm-restart-any` in `engines/pyscf.md` § 5; the stage-4 refinement preset in `engines/tuning.md` § 1; the MD viewer/editor in `web/trajectory.md` § 1. Each says WHY, so the next person does not re-propose it. **One turned up a false claim**: `warning-modal.js` listed `beforeunload` as its third trigger, *"registered by the tab UI module"* — no module ever registered one, so closing the tab has never warned about an unsaved canvas | roadmap § 4 | done |
| ~~**N3**~~ | **ALL THREE WERE ALREADY FIXED — the row was stale, verified 2026-09-03.** One commit, `a18398fc`, *"the three O5 wording defects, each with its guard"*, dated **2026-08-28 — the same day the audit note recording them was written**. The note and the fix crossed and the note was never struck. Measured rather than read: the prepped list prints the path from the bundle (`_rel(d)`), not the bare attempt; a scheduler-less `ask` previews no `sbatch` and points at `--mode direct`; and a 5-trial sweep runs the science gate 10 times, prints each warning ONCE and says *"20 repeat(s) across the other trials suppressed — every trial's own .validation.txt carries its full findings"*. Two tests guard them. **This is the fourth time today a recorded finding did not survive being re-derived** — the plan's rows are read, not re-measured | audit 08-28 O5 | done |
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
| **P3** | **B3** | **233 front-end tests assert substrings of `.js`/`.css`**, one pinning six spaces of column alignment. The cure exists at the deck layer (`tests/_deck.py`, written after padding broke 45 tests across eight files) and the front end never got it. **Build the equivalent first, then convert** — deleting these loses real coverage. **First pass done 2026-09-03** (see B3.1); ~200 remain | 233 |
| ↳ | **B3.1** | **DONE 2026-09-03 — the self-confessing subset.** A test whose own docstring says the real check lives elsewhere is retired, and the test it names gets written (`process/testing.md` § 3a.1, and `code-audit.md` § 5 rule 5 which was still *instructing* auditors to write these). Eleven confessed; **three were contrastive, not confessional** (`test_page_ids_unique.py`, `test_pages_no_js_errors.py`, `test_pdb_workflow_integration.py` all drive the artifact and cite string-pins only as what failed before) and one more survived reading (`TestSpectraIssuesPanelSeverityCoverage` is a CSS one-home lint whose confession describes the greps it replaced). **40 test functions removed** (38 deleted, 2 rehomed) **and 12 written** — 10 new plus the 2 rehomed. Every replacement mutation-verified. Every replacement mutation-verified. Two were *not* written and the reason is on record where each belongs: the second-load widget defect is unreachable by a single break (two publishers each rescue the other), and "a row is born at its value in force" states no rule any document carries | 40 |
| ↳ | **B3.2** | **DONE 2026-09-03 — reviewing the replacements found eight defects in them**, three substantive: a Slack channel was tested carrying a signing key (there is no such control — only a listener has one), an assertion demanded that NO part of a webhook appear when `MASK_TAIL = 4` shows the last four on purpose, and an `ok` check passed on an absent key. Two were vacuous (a vocabulary check that passes on a class with no tags; a chooser asserted to exist but not to offer anything) and three fragile (both timer tests counted the page's own intervals; a checkpoint picked by position not name; one dialog accepted where `_restore` asks twice). A green run plus one mutation had hidden all of it. Asserting the page reports no JS errors also found a **live product bug** — `pattern="[A-Za-z0-9_-]{1,64}"` never compiled under the `v` flag, so the channel-name rule was stated and never enforced (`d243e852`) | 8 |
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

## 5c. The directory door — `JobDirParser`, and its migration

*(Agreed 2026-09-04. Contract: [`model/parse.md` § 5](?doc=model/parse.md).
**Not started.** This section is the plan, and the caller list below is the
completeness check — the requirement is that nobody is left behind.)*

**The shape.** One DirParser answers everything asked *about a run
directory*; the four fields each have a named reader before a line is
written (§ 5.0's table). `active` is picked **stage, then mtime** (user
ruling) — `summarize`'s highest-`-runN` rule loses, because a run index says
nothing about which stage a file belongs to.

### The complete caller map, measured

Only **three** modules consume any of this, which is what makes the migration
checkable rather than hopeful:

| what it does today | where | becomes |
|---|---|---|
| `run_status(dir)` | `jobset/runstatus.py` ×1 | `.status` |
| `_resolve_run_directory(dir)` — the 4-rung chain | `web/blueprints/watch.py` ×1 | `.openable` + `.attempts` |
| `_list_molwatch_logs(dir)` | `web/blueprints/watch.py` ×2 | `.files["molwatch"]` |
| `_engine_of(...)` | `web/blueprints/watch.py` ×4 | `.engine` |
| `engine_of(dir)` | `web/blueprints/watch.py` ×1 | folded in as `.engine` |
| `atom_metadata_json_for_run_dir(dir)` | `web/blueprints/watch.py` ×1 | `.files` + one parse |
| `contract_of(dir)` | `parse/dirs/run_info.py` ×1 | unchanged — its own verb, over `.files["fdf"]` if the door is handy |

**Two rows were in this table and are removed, both invented by me and
both caught by re-reading the code** *(2026-09-04, second pass)*:

* `_read_system(bundle)` takes the **bundle root** — the calculation
  folder holding `task.json` — reads the DESCRIPTION first, and falls back
  to root decks. It is not a run-directory question at all.
* `_latest_run_file(d, base, suffix)` is called four times for four
  artifact KINDS (`scf-timing.log`, `monitor.log`, `util.csv`, `out`), and
  its `basename` is `Path(j.script).stem` — for a staged deck
  `<label>_<NN>_<stage>`, so **the stage is already chosen** and the only
  remaining choice is the RUN INDEX. That is not `active`'s question
  (*which result file across stages carries the status*), and mapping them
  together would have applied a stage rule where stage is not a variable.

Internals absorbed rather than migrated: `_enumerate_files`, `_build_status`,
and `contract.py`'s three declaration rungs.

### Two boundaries, decided

**`attempt_concluded` stays out** — `jobset/prep.py`, `jobset/submit.py`,
`transport/compose.py`. It answers *"may I launch here"*, its consumers are
WRITERS, and folding it in would make the launch path depend on the parse
layer for a decision that is not a reading.

**Sibling lookups stay out** — `_sidecar._siesta_fdf_path_for`,
`pyscf._resolve_job_token`, `siesta_mdnc.sibling_md_nc`. § 5a's sibling
upgrade: a parser locating one file from another *of its own format*.
Absorbing them inverts § 5's rule that a DirParser composes FileParsers.

### The behaviour changes, named in advance

**There are none to the numbers.** An earlier draft of this section said
`summarize`'s per-trial pick would change from highest-`-runN` to
stage-then-mtime; that was the invented mapping above, and the claim is
withdrawn. `active` is stage-then-mtime (user ruling) and it governs the
STATUS only, which has always used that rule.

The one visible change: **`detect()` on a directory starts resolving
again.** It has had no DirParser since 2026-09-04 and could only refuse.

### What `summarize` actually gets, which is less than first claimed

`_enumerate_files` buckets ENGINE artifacts only — `.fdf`, `.out`, `.XV`,
`.STRUCT_OUT`, `.molstruct.json`, `.ANI`, `.molwatch.log`. Three of the four
files `summarize` reads per trial are the WRAPPER's instrumentation
(`scf-timing.log`, `monitor.log`, `util.csv`) and are in no bucket.

**DECIDED AND DONE 2026-09-04** (user: *"yes give them parsers"*). The
three wrapper files are registered parsers now — `scf-timing`,
`monitor-log`, `util-csv`, returning `InstrumentResult`
([`parse.md` § 5c](?doc=model/parse.md)) — and `bench/result.py` no longer
opens a file. So the door CAN learn them: `_enumerate_files` gains three
buckets and `summarize` reads through `.files` like everyone else.

That still leaves the migration at TWO consumers and eleven sites for the
DIRECTORY door itself, because `summarize` finds its files by
`_latest_run_file` (run index, for an already stage-scoped basename) and
that is not `active`'s question — see the withdrawn row above:

* `web/blueprints/watch.py` — 9 sites
* `jobset/runstatus.py` — 1 site
* (`contract_of` unchanged)

### Order of work

1. Write `RunDirResult` + `JobDirParser`, absorbing the chain verbatim.
   Prove equivalence on the real tree BEFORE any caller moves — the
   `run_status` split did this (113/113 identical) and it is the reason that
   deletion was safe.
2. Move `runstatus` (1 site), then `watch` (9), then `summarize` (5).
3. Delete the absorbed functions and sweep their documents.
4. Re-run the caller map above and require it to come back empty.

---

## 5d. The script blocks belong to their writer — retiring `parse/scripts/`

*(Agreed 2026-09-05. Contract: [`model/parse.md` § 1 / § 6](?doc=model/parse.md)
and [`execution/job-contracts.md` § 3.1](?doc=execution/job-contracts.md).
**Not started.** This section is the plan; the caller map below is the
completeness check.)*

### The category error

`parse/` exists to read **foreign** formats — a `.out` that might be SIESTA or
PySCF, a `.XV`, a `.MD.nc`. That is why it has a registry, `can_parse`, and
detection: *"every consumer imports from here and **queries the registry
rather than knowing which parser to call**"* (§ overview).

The script-contract reserved blocks are **not foreign**. molbuilder writes
them, into a file molbuilder generated, and every caller already knows which
block it wants. There is nothing to detect. Applying a detect-and-dispatch
tier to a question with one possible answer is what made every class in it
ceremony:

```python
# ProvenanceTextParser.parse — the entire body
base = empty_script_result(cls.name)          # a 10-field object …
return replace(base, provenance=_extract_provenance_dict(text))   # … to carry one dict
```

`parse.md` § 1 says as much without drawing the conclusion: **"`TextParser`
has no detection — a text body has no path to inspect, so the caller names
the parser."** An ABC in a detection package that does not detect.

### The evidence that they are one module split in two

Measured 2026-09-05, not inferred:

1. **The readers import the format from the writer.** `parse/scripts/markers.py`
   is 40 lines of pure re-export from `script_emit`, and says why: *"so the
   read-side parsers stay in lock-step with the write-side emitters — one
   spelling of every marker, not two."*
2. **The writer imports the readers back.** `script_emit.py` is the single
   biggest consumer of the extractors — lines 606, 611, 722, 730, 791, 795,
   799, 803, 807, 811.
3. **Which deadlocks, and the deadlock was papered over.** `script_emit.py`
   carries a `_LAZY_EXTRACTORS` table whose comment reads: *"an eager
   top-level import would **deadlock** because markers.py re-exports `BLOCK_*`
   + `MARKER_RE` from this module."* The split created a circular import; the
   workaround has been shipping since.
4. **The one class with a production user justifies itself circularly.**
   `parse/dirs/atom_metadata.py` says its glob/read helper lives apart
   *because* the TextParser is memory-only under § 7 forbidden #2 — a rule
   that exists only because TextParser exists.

**The move therefore deletes a circular dependency and a lazy-import
workaround.** That is the structural gain; the line count is a side effect.

### The shape

One module owns the block format **in both directions**: it emits the blocks
and reads them back. `parse/` keeps `FileParser` and `DirParser` for foreign
formats and loses `TextParser`, which has no other implementation.

```python
from molbuilder.script_emit import read_script
read_script(text).provenance          # instead of reaching for a private fn
```

### The complete caller map, measured

Every user of every symbol in `parse/scripts/`, counted by AST + grep over
`molbuilder/` and `tests/` on 2026-09-05.

| symbol | production users | verdict |
|---|---|---|
| `_extract_atom_metadata_dict` | `script_emit` ×3, `transport/compose` ×2 | **moves** — real work |
| `_extract_bench_marks_dict` | `script_emit`, `jobset/agreement` ×2, `jobset/summarize` ×2 | **moves** |
| `_extract_provenance_dict` | `script_emit`, `parse/contract` ×2 | **moves** |
| `_extract_header_text` | `script_emit` | **moves** |
| `_extract_user_custom_inner` | `script_emit` ×3 | **moves** |
| `_extract_script_source` | `script_emit` (lazy re-export only) | **moves**, carrying the schema-version gate |
| `markers.py` | re-export of `script_emit`'s own constants | **deleted** — the constants are already home |
| `AtomMetadataTextParser` | `parse/dirs/atom_metadata.py` ×3 | **deleted**; that caller takes the function |
| `BenchMarksTextParser` | none | **deleted** |
| `HeaderTextParser` | none | **deleted** |
| `ProvenanceTextParser` | none | **deleted** |
| `UserCustomTextParser` | none | **deleted** |
| `ScriptSourceTextParser` | `parse/__init__` re-export only | **deleted** |
| `empty_script_result` | none anywhere | **deleted** |
| `ScriptResult` | `parse/__init__` re-export, `types.py` definition | **deleted** — no reader, `result_kind "script"` never checked |

**The two umbrellas fold into one.** `source.py` (71 lines, pure ceremony) and
`source_dict.py` (153 lines, the extractor **plus the only copy of the
schema-version gate**) become one `read_script`. `source_dict.py`'s docstring
has flagged this overlap as deferred since it was written, pending *"its own
careful pass over the run decoder"* — the run decoder was deleted 2026-09-04,
so the deferral has no remaining condition.

### What this costs the contract

`parse.md` goes from **three ABCs to two**. The sections that follow:

* § 1 — the ABC table loses its `TextParser` row.
* § 2 — `ScriptResult` leaves the hierarchy; the class diagram and the
  discriminator list follow.
* § 3 — `parse_text(text, parser)` loses its last caller. **Decide: delete it,
  or keep it as a public seam with no implementation?** *(Recommendation:
  delete. A registry function that can dispatch to nothing is the same
  scaffolding as `parse_dir` without a DirParser, which § 5 already carries a
  warning about.)*
* § 4 — the layout tree loses `scripts/`.
* § 6 — the "Block TextParser" row goes.
* § 7 — forbidden #2 is *"TextParsers do NO I/O"*, and the ABC it names is
  going. **The RULE stays and moves with the code**: these extractors are pure
  text functions, and one that started reading files would be a real defect,
  ABC or not. Re-aim it at the extractors in their new home — *"the block
  readers take a string; reading the file is the caller's job"* — and keep
  BOTH guards pointed there (`test_scripts.py::test_text_parsers_do_no_io`
  and `test_audit_gaps.py::test_forbidden_p2_textparsers_do_no_io`, which
  lints `parse/scripts/*.py` for I/O tokens and needs its path changed).
  Retiring an ABC is not a reason to retire the property it happened to
  carry.

`job-contracts.md` § 3.1 owns the block format and gains the read half beside
the emit matrix it already documents.

### Order of work, and what proves each step

1. **Doc first** — this section, plus the § 1 / § 2 / § 6 / § 7 edits, agreed
   before any code moves.
2. **Move the six extractors** to the writer, unchanged. Proof: the extractors'
   own tests pass against the new home with no edits to the assertions.
3. **Give it one door.** `read_script(text)` — one pass, all blocks, the
   schema-version gate included. Measured cost of reading all six on a real
   442-line deck: **0.55 ms vs 0.22 ms** for one, so a per-block door buys
   nothing.
4. **Route the three real callers** — `parse/contract.py`, `jobset/summarize.py`
   + `jobset/agreement.py`, `transport/compose.py` — off the private imports
   and onto the door. Proof: no `_extract_` name is imported across a package
   boundary anywhere (a lint, since this is the defect the move exists to fix).
5. **Delete the ceremony** — six classes, `ScriptResult`, `empty_script_result`,
   `markers.py`, and the `_LAZY_EXTRACTORS` workaround with the circularity
   that forced it. Proof: `script_emit` imports the extractors **eagerly** and
   the suite is green — the deadlock is gone, not hidden.
6. **Retire the contract text** in the order above.

**The completeness check is step 4's lint**: after the move, a private
`_extract_*` name imported from another package is a failure, not a style note.


---

## 5e. Engine additions — a person's own engine text, as an INPUT

*(Your ruling, 2026-09-05: option 2 — a distinct input to the writer, not a
catalogue extension; **engine-specific**; and **the person is told the
consequence**. Contract-first: this section is the design. **Not started.**
No code has moved.)*

### The need

A person wants engine content molbuilder does not model: a Lua script driving
a SIESTA relaxation, a `%block` for a feature with no form field yet, a PySCF
call. Today that goes in the USER-CUSTOM zone, and a UI card is wanted for it.

### Why the zone cannot carry it — measured, not argued

`render_deck` assembles the deck like this (`script_emit.py:1319`):

```python
text = (science + "\n\n" + emit_user_custom_placeholder()
        + "\n\n" + machine_record_banner()
        + "\n\n" + "\n\n".join(record) + "\n")
```

**The zone is not in the layout.** `spec.layout` never sees it; the framework
concatenates a placeholder after the walk. Everything else in a deck arrives
through the model — `Section.items` are CATALOGUE NAMES, turned into a
`Parameter` and handed to the engine's `line`, one line each, recorded in
`emitted` so `check_deck` can close the loop against the written file.

Three consequences follow from that one fact, and they are not three bugs:

1. **A user's line can duplicate a declared item's**, because nothing compares
   them — the writer never saw it.
2. **Position is fixed below the science.** Measured on a real deck: engine
   body at lines 12 / 520 / 980, zone at 1222. libfdf takes the FIRST
   occurrence, so *anything the deck already writes cannot be overridden from
   the zone*.
3. **The engine's own rules refuse it.** SIESTA's `check_rules`
   (`siesta/layout.py:354`) splits the whole file and knows nothing of the
   fence, so a duplicate raises **error** severity and `prep` refuses.

The Lua case is decided by (2) and (3) together: SIESTA engages Lua with
`MD.TypeOfRun Lua`, which the catalogue already declares. Measured — that deck
is refused, and the refusal is *correct*, because libfdf would have ignored the
line anyway. **The zone can never carry the feature it is documented for.**

### The shape

An **engine addition** is a person-supplied contribution to one engine's deck.
It is an INPUT, alongside the structure and the config — never text recovered
from a previous output.

| | |
|---|---|
| **engine-specific** | an addition is SIESTA text or PySCF text; there is no engine-neutral addition, because the content is engine syntax. It is declared for one engine and ignored by the other, the way `[item.*]`'s `engines` key already works |
| **the writer places it** | additions reach `render_deck` in the spec and are emitted during the layout walk, in the layout's position — not appended after it |
| **emitted once** | if an addition writes a keyword a declared item also writes, ONE line is written, not two |
| **recorded** | its lines join `emitted`, so `check_deck` closes the same loop over them as over every other line |

### The consequence a person is told

Silent resolution is the thing to avoid. `Parameter.writes` already answers
*"which engine keywords does this item put in the deck"* (from `expands`, else
`anchor`), so a collision is **detectable, not guessable**:

- an addition writing a keyword **no** declared item writes — accepted, placed,
  no notice;
- an addition writing a keyword a declared item **also** writes — the person is
  told, at the point of entry, what is about to happen: *your value replaces
  what `MD.TypeOfRun` would have written (`CG`)*. Their value wins, because
  they said it last and more specifically — but never without being told;
- an addition molbuilder cannot attribute to any keyword (a `%block`, free
  prose) — accepted verbatim, and the engine judges it, which is the honest
  half of today's § 3.5.

**This is the rule the current design cannot state**: today a person is either
refused (duplicate) or silently ignored (first-wins), and which one depends on
whether SIESTA's rule happens to notice.

### What this DELETES, which is the argument for it

The read-back merge stops being needed. Additions are an input, so nothing is
recovered from the file being overwritten. With it go:

- `merge_user_custom_from_target` and `write_script`'s round trip — and with
  them `check_deck`'s reason to read the file rather than the rendered string;
- `template.md` § 9.2's three structural objections to `prep` (no previous
  deck, one deck per stage, reproducibility) — all of which exist only because
  the mechanism reads the previous output;
- the transport gap: `_prep_transport` writes with a bare `write_text` and
  `molbuilder/transport/` never emits the zone at all, so transport has no
  territory today. Additions are an input, so transport gets them by having a
  writer, not by growing a zone;
- the stray-marker data loss: a marker pasted into the zone silently drops
  everything above it (measured, HTTP 200, no warning) — impossible when there
  is no zone to paste into;
- `job-contracts.md` § 3.5's "byte-for-byte" claim, which is already false
  (CRLF is normalised by `splitlines()`/`"\n".join`).

### Where it is typed, and who is responsible

*(Your proposal, 2026-09-05, taken with one correction from `tabs.md`.)*

**An addition needs no new home: it follows the path a parameter already
takes.** `tabs.md` § 1 has the ownership — Structure optimization *"collect[s]
a SIESTA/PySCF relaxation's PARAMETERS and Send[s] them to Task setup (the
deck itself is written by `prep`, on the machine that runs it)"*, and Task
setup *"read[s] a calculation folder's description — its STAGES"*.

Neither tab writes a deck. So an addition typed in either must travel as DATA
to `prep`, which is exactly what makes it an input rather than a zone — and it
lands on the relationship every parameter already has:

| where | what it does |
|---|---|
| Structure optimization | collect the addition with the calculation's other parameters |
| Task setup | override it per stage, like any item |
| `prep` | the writer places it |

So the answer to *"Task setup or Structure optimization?"* is **both, in the
roles those tabs already hold** — not a choice between them.

**The responsibility is the person's, and the format is what makes that fair.**
molbuilder does not understand the content; it places it and records it. For
"your responsibility" to be a fair deal rather than a disclaimer, three things
have to be true, and only the first is about the text:

1. **A stated format** — a clear start and end, so the addition is a bounded
   thing rather than loose text. (Not the current marker fence, which is
   file-level and is what a stray paste can break; the bound belongs to the
   addition as data.)
2. **It is separable at generation time.** `prep` can write the deck WITHOUT
   the additions and WITH them, because they are an input rather than text
   fused into the file. That gives a person the bisection directly: run it
   clean, run it with, and the difference is theirs.
3. **The consequence is stated before it is saved**, per `Parameter.writes` —
   *your value replaces what `MD.TypeOfRun` would have written (`CG`)*.

(2) is the one that turns responsibility into something a person can act on,
and it is a capability the input model gives for free. Under a zone it is
possible only by hand-stripping a section from a written file, which changes
the deck in more ways than the one being tested.

### Open, and deliberately not decided here

1. **Does the zone survive at all** for genuinely free-form text (a comment
   with no variable in it), or does that become an addition with no attributed
   keyword?
2. **What the card shows** when an addition collides — refuse, warn-and-accept,
   or show the resolved line before saving. (The consequence must be stated;
   whether it can be overridden is separate.)
3. **Ordering among additions**, when two of them write to the same section.
4. **How the without/with pair is offered** — two files, a flag on `prep`, or a
   diff shown in the tab.

### Before any code

This section is the contract. The measurements it rests on
(`script_emit.py:1319`'s concatenation, the 12/520/980-vs-1222 positions, the
reproduced duplicate-keyword refusal, `Parameter.writes`) are re-checkable, and
should be re-checked rather than trusted if this is picked up later.


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
