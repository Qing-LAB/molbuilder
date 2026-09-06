# The plan — one file

**Role:** plan — **the only one.** Every open item from the nine plan
documents that preceded it lives here; those nine are archived under
`docs/archive/2026-09-01-*.md` as records of what was decided and built.
**Domain:** all
**Started:** 2026-09-01, by consolidation

> *(user, 2026-09-01: "We don't need ten plan files scattered. We want one
> plan folder or one plan file and stay with that file.")*

**Status, 2026-09-06.** This file began as the merge, fact-checked against
code on 2026-09-01. Much of it has since been executed — struck rows carry the
date and the evidence. Two later rounds are recorded in place: **§ 5b** (the
2026-09-02 run-decision round) and **§§ 5f–5g** (2026-09-06 — the architecture
seams the merge dropped, and what re-deriving nine of this file's own rows
found: four held, five did not).

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
| ~~**E2**~~ | **DONE 2026-09-03 by your ruling — *use the exact one when it's there, fall back to the csv*.** `parse_utilisation` is the one door: the monitor's `[UTIL-SUMMARY]` means (averaged over EVERY tick) where it wrote them, `util.csv`'s time-weighted reconstruction (a ≥10%-change-gated subset) where it did not — which is what a KILLED trial leaves, the one a benchmark most needs to read. | `bench-and-junction` § 2.2 | done |
| ~~**E3**~~ | **DONE 2026-09-03 — named, with the tell that detects it.** | `bench-and-junction` § 3B | done |
| ~~**E4**~~ | **DONE 2026-09-03.** The *node name and CPU model* half was already there — the monitor's `[MACHINE]` line is written first, unconditionally, on every path. | `bench-and-junction` § 3D | done |
| **E5** | **Two facts need a Sol run**, not code: a real `[MACHINE]` line from a live job, and a re-probe to settle `lightwork`'s cap | `machine-identity` § 6 | blocked on a machine |
| ~~**E6**~~ | **DONE 2026-09-03 by your ruling — a reference, not a default.** | found 2026-09-01 | done |
| **E7** | **D7's cluster half** — the same prep→submit→watch loop through SLURM on Sol. Everything else about D7 shipped; this needs the machine, and pairs with **E5** in one session | roadmap § 1 | blocked on a machine |
| ~~**E8**~~ | **ALREADY DONE — verified 2026-09-03, not changed.** | roadmap § 1 | done |
| ~~**E9**~~ | **BOTH HALVES CLOSED 2026-09-03.** | roadmap § 6 | done |
| **E10** | **Detecting from the `.out` that a run STOPPED**, and whether it stopped converged (`SCF_NOT_CONV` / `ABNORMAL_TERMINATION`) — **on a zero exit too**, since an MPI stack may not propagate an abort. Ruled to belong to `mb_monitor.py`, not the execution branch | roadmap § 4 | open |
| **E11** | **A fresh live walk of the PySCF / spectra decks.** The 2026-08-28 review exercised them only through the guard suites and says so | audit 08-28 § 5 | open |

## 3. Open — configuration and ops

| # | item | from | state |
|---|---|---|---|
| ~~**C2**~~ | **DONE 2026-09-03 — measured on a real built wheel: 90 of 141 static files.** | roadmap § 4 | done |
| ~~**C3**~~ | **DONE 2026-09-03 — deleted.** The package was 35 lines re-exporting `molbuilder.builders.backends`, kept *"for external callers"*; and `build.py` carried a comment saying in-tree code goes direct. | roadmap § 4 | done |
| ~~**C1**~~ | **DONE 2026-09-03 — and there were more than three.** | `config-access` § 5 | done |

## 4. Open — the front end

| # | item | from | state |
|---|---|---|---|
| **W1** | **The document tier (step C).** `html, body`, `header`, `button`, `footer`, `textarea` genuinely differ per page; the `*` reset is already deleted. Blocked on a browser pass over all pages | `css-system` § 4C | partly |
| **W2** | **One home per component (step D).** `.card`, `.status`, `header .tagline`. **One value to settle first:** `.card`'s padding is `var(--space-md) 18px 18px` and 18 is off the 4px grid the contract declares — moving it shifts every page by 2px | `css-system` § 4D | not started |
| **W3** | **Per-page token/namespace passes (step E)**, one page per commit: `spectra`, `structure-optimization`, `transport`, `results`, `documents` | `css-system` § 4E | partly |
| **W4** | **Guards 1 and 2 (step F)** — one home including elements; a page sheet contains only its own tier. Guard 3 landed; a fourth (no long block copied between sheets) landed 2026-09-01 | `css-system` § 4F | partly |
| **W5** | **The inspectors module's appearance still lives in `results/style.css`** — 70 mentions. Mount an inspector on another page and it renders unstyled. Three of its sheets have been repatriated; the rest is a module change, deliberately | `css-system` § 7.0 | partly |
| **W6** | **The editor module.** **Re-measured 2026-09-06 and the row was overstated in one half and stale in the other.** The loader half is *half done*: `lib/codemirror-load.js` is the ONE loader as of 2026-08-16 and two of the three surfaces import it — `lib/inspectors/markdown.js` still hand-rolls its own `loadCSS`/`loadScript` pair (`markdown.js:45–50`), so it is **two loaders, not three**. **Three** sheets theme `.CodeMirror` (`markdown.css`, `task-setup/style.css`, `projects-sidebar.css`) carrying **30** rules — not the 40 this row claimed; 30 is what the original plan measured, and the increase never happened. The caps (1500-line selection, 1 MB view-only) on **one** surface stand | `editor-module` | partly |
| **W7** | **I7 close-out** — the browser walk of Results export → cite → describe → prep. I1–I6 done | `structure-info` I7 | open |
| **W8** | **Caller-less endpoints — decide the ROLE, do not just delete.** `/api/docs/list` (a second answer to "what docs exist" beside the `/api/docs/toc` the tab calls); `/api/checkpoint/config` (the read half of a route whose write half was removed); `/api/selection/atoms` (pinned by five test files) | `structure-info` § 3 | open |
| **W9** | **Transport viewer default orientation** — a MolView camera-door contract question | `structure-info` § 3 | open |
| **W10** | **Results transmission inspector** — the record exists, the reader does not | `structure-info` § 3 · roadmap § 2 | open |
| **W12** | **The live browser walk-throughs** — checkpoint swap at narrow and wide widths, per-tab reload round-trips, a real Data/Image export, click-selection on frames ≥ 1. Carried since the archived molview-and-checkpoint plan | roadmap § 0.3 | open |
| **W13** | **Raw px/rem literals — re-measured 2026-09-06, and the row's number does not survive its own definition.** In *the page sheets* (the eight per-page directories) there are **160**, not 384. The bulk is somewhere this row never said: **`lib/` carries 740**, which is where the shared components live and where a token would pay for itself most. Definition first next time — 777 → 384 → 160 are three scopes, not three measurements of one thing | roadmap § 7.4c | partly |
| **W14** | **A web plan view and a per-stage status roll-up.** The web *describes* a staged calculation (Task setup) and *observes* runs (Results); neither shows the plan as a whole | roadmap § 6 | open |
| **W15** | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. Plus routing the CLI through the shared codec and exercising the last annotation-channel kind | roadmap § 3 | partly |
| ~~**W11**~~ | **DONE — verified 2026-09-06 across both halves.** | `structure-info` § 3 | done |

## 4a. Open — needs a decision from you

| # | item | from | state |
|---|---|---|---|
| ~~**N1**~~ | **CLOSED 2026-09-03 — the rule is written and there is nothing to fix.** | audit 08-28 O4 | done |
| ~~**N2**~~ | **ALL NINE RETIRED 2026-09-03** | roadmap § 4 | done |
| ~~**N3**~~ | **ALL THREE WERE ALREADY FIXED — the row was stale, verified 2026-09-03.** | audit 08-28 O5 | done |
| **N4** | **The science-validation tail** — checks deferred with recorded rationale in `science/pseudopotentials.md`. The roadmap carried them under a heading that said *"needs a home"* for five weeks | roadmap § 5 | needs a home |

## 5. Open — documentation drift

| # | item | from | state |
|---|---|---|---|
| ~~**D1**~~ | **DONE 2026-09-03 — closed, and now guarded.** | `consolidated-cleanup` § 7 | done |
| **D2** | **Tests with no target, remainder.** 12 removed; ~14 flagged. The two files with zero test functions were **checked and left** — each is a signpost recording where retired coverage moved, which is a service, not residue | `consolidated-cleanup` § 9 | partly |
| ~~**D3**~~ | **DONE — and guarded.** | `consolidated-cleanup` § 9 | done |
| **D4** | **The README screenshots are three tabs stale** — five captured, eight ship. Nothing can enforce this (no test can count tabs in a PNG); the *owner* of the count is pinned as of 2026-09-01 | `screenshots.md` | open |

## 5a. A row is evidence of when it was written — re-derive before acting

Twice now, this file's rows have been checked against the tree and most of
them had not survived. **The number is the point, not the rows.**

**2026-09-01, the roadmap's ~40 items.** Of everything it presented as open,
about a quarter survived contact with the code; the rest were closed and never
struck. It said so about itself twice before anyone acted on it — § 7.4's
*"carried as open for a day after they shipped"*, § 7.5's *"re-read against the
tree and most of it was already gone."*

**2026-09-06, nine of this file's own rows. Four held, five did not.** W11 was
done (both halves verified); W6 and W13 were overstated (two loaders not three;
160 px/rem in the page sheets, not 384 — and the bulk, `lib/`'s 740, was in
neither count). But R4 is **not** done, W2/W5/W8 are accurate exactly as
written, and two findings withdrew entirely: a deck assertion that never
executes (not reproducible under two mechanical searches) and a dropped
`max_memory_mb` unit (both halves already shipped).

**The rule.** Re-derive before acting, and **write the measurement into the
row** — every row corrected on 2026-09-06 now carries its own `file:line`, so
the next check is a check rather than a survey. A count with no stated
definition is not a measurement: 777 → 384 → 160 and 233 → 256 → 173 → 49 were
each three or four *scopes*, which is why the second one now has a tool
(`tools/classify_source_reads.py`) instead of a number.

---

## 5b. Open — found 2026-09-02, the run-decision round

Two code-review agents and six test-audit agents, every acted-on claim verified
by hand. **Priority is the first column and means: P0 misleads a person right
now; P1 is a check that silently is not checking; P2 is a gap in what shipped
today; P3 is cleanup, large and mechanical.**

### The front end — what a person is told to run

| P | # | item | evidence |
|---|---|---|---|
| ~~P0~~ | ~~**R1**~~ | **DONE 2026-09-02.** `--from` is composed in the browser and was wrong for `flat`. | **this is V2 of the 2026-08-13 program, resurfaced.** Closed on the CLI, never on the browser |
| ~~P0~~ | ~~**R2**~~ | **DONE 2026-09-02, and UNVERIFIED** — no harness drives the stage panels, so the fix is read-checked only; a test needs **B3** and is tracked with **M2**. | verified |
| ~~P2~~ | ~~**R3**~~ | **DONE and PROVEN 2026-09-02.** Page state survived a folder change, against `task-setup.md` § 2.1's *"the page holds no state of its own"*: `_extraRunRows`, `_pendingDrop`, `_queue`. | verified |
| ~~P2~~ | ~~**D8**~~ | **DONE 2026-09-02.** | verified against all six surfaces |
| **P2** | **R4** | **Three writers, one buffer, a 400 ms loss window.** `syncFromModel` writes from `_task`; `applyAsksToDoc` and `applyNotifyToDoc` patch the buffer and never touch `_task`. Type a memory value, blur, click "+ Add stage" inside the debounce and the `allocation` block is gone | verified |
| ~~P0~~ | ~~**R6**~~ | **DONE 2026-09-02.** | found by `test_task_setup_prep_e2e.py`; user: *"all environments have to be explicitly probed and stored. no environment json, error"* |
| ~~P1~~ | ~~**R7**~~ | **DONE 2026-09-02** *(user: "safety first … compatibility is not an issue. | pinned + mutation-tested |
| ~~P1~~ | ~~**T4b**~~ | **DONE 2026-09-02.** | pinned |
| **P3** | **R5** | **`"(this machine)"` means `LOCAL_TARGET` at the prep door and `None` at the bench-grid door.** Real asymmetry, but **the fix is not unification** — `None` is what lets the reader prefer the bundle's own snapshot, and forcing them together broke a live GPU test. The narrow gap: on an unprepped folder with named records, both fit blocks 400 and hide themselves. Fix the *surfacing*, not the value | tried and reverted 2026-09-02; the reasoning is in the code |

### Tests that are not testing

| P | # | item | evidence |
|---|---|---|---|
| ~~P1~~ | ~~**T1**~~ | **DONE 2026-09-02 — by making the claim TRUE, not by editing it away.** | verified |
| ~~P1~~ | ~~**T2**~~ | **DONE 2026-09-02.** Resolves from `__file__` like every other path in the file, and asserts the source list is non-empty so a future blind run says so. | verified |
| ~~P1~~ | ~~**T3**~~ | **DONE 2026-09-02.** Names `dataclasses.FrozenInstanceError`; mutation-tested by un-freezing the dataclass. | reported, spot-checked |
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
| ~~P3~~ | ~~**B1**~~ | **WRONG, and corrected 2026-09-02 — the tests stay.** | investigated in code + contract, not grepped |
| **P3** | **B2** | **PARTLY DONE 2026-09-03 — and the number was the wrong instrument.** Of the four shapes named here, only one is mechanically decidable: a `Test*` class whose body is a docstring collects nothing. Five existed. **Three were empty promises** — `TestBuildSiestaHonorsSidecarFrozenAtoms`, `TestWorkspacePayloadRegionsAndFrozen`, `TestGenerateWritesToWorkspace`, each stating in the present tense that it pins something (*"Tests pin both layers"*) while holding no test, so a reader scanning for coverage reads yes. Each is replaced by a pointer at the file that DOES cover it. **Two are deliberate retirement markers** that say so and name their successor — the same call D2 already made for two zero-test files. The other three shapes do not survive measurement: `assert len(X) == 5` where the test BUILT X is a real check, and `m = re.search(...); assert m` is a precondition with the real assertions after it. **A list of ~45 that cannot be re-derived is not a finding anyone can act on** — what is left needs the file-by-file read, not a regex | 5 measured |
| **P3** | **B3** | **CLASSIFIED 2026-09-06 — and the population is a fifth of what three earlier counts claimed.** 233, then 256, then 173 were three definitions, none written down. Measured now by `tools/classify_source_reads.py`, which states its definition and can be re-run: of **1,255** assertions over a file's text, **1,147 read GENERATED output** and are correct as text — a property of a real product, never a defect. **108 read hand-written source**, in 31 files. Of those, **59 stay** (51 lints, where text is the only instrument that can prove absence, and 8 vendored/data files) and **49 convert**. Full method, per-bucket file list and the mutation proof are **§ 5h** | 49, not 233 |
| ↳ | **B3.1** | **DONE 2026-09-03 — the self-confessing subset.** A test whose own docstring says the real check lives elsewhere is retired, and the test it names gets written (`process/testing.md` § 3a.1, and `code-audit.md` § 5 rule 5 which was still *instructing* auditors to write these). Eleven confessed; **three were contrastive, not confessional** (`test_page_ids_unique.py`, `test_pages_no_js_errors.py`, `test_pdb_workflow_integration.py` all drive the artifact and cite string-pins only as what failed before) and one more survived reading (`TestSpectraIssuesPanelSeverityCoverage` is a CSS one-home lint whose confession describes the greps it replaced). **40 test functions removed** (38 deleted, 2 rehomed) **and 12 written** — 10 new plus the 2 rehomed. Every replacement mutation-verified. Every replacement mutation-verified. Two were *not* written and the reason is on record where each belongs: the second-load widget defect is unreachable by a single break (two publishers each rescue the other), and "a row is born at its value in force" states no rule any document carries | 40 |
| ↳ | **B3.2** | **DONE 2026-09-03 — reviewing the replacements found eight defects in them**, three substantive: a Slack channel was tested carrying a signing key (there is no such control — only a listener has one), an assertion demanded that NO part of a webhook appear when `MASK_TAIL = 4` shows the last four on purpose, and an `ok` check passed on an absent key. Two were vacuous (a vocabulary check that passes on a class with no tags; a chooser asserted to exist but not to offer anything) and three fragile (both timer tests counted the page's own intervals; a checkpoint picked by position not name; one dialog accepted where `_restore` asks twice). A green run plus one mutation had hidden all of it. Asserting the page reports no JS errors also found a **live product bug** — `pattern="[A-Za-z0-9_-]{1,64}"` never compiled under the `v` flag, so the channel-name rule was stated and never enforced (`d243e852`) | 8 |
| **P3** | **B4** | **MEASURED 2026-09-03; the envelope half is done, the fixture half is proposed and NOT applied.** The `_envelope()` count was seven, and only **three** were re-implementations: `test_pseudos.py` and `test_task_setup_tab.py` hand-listed the envelope's fields (so a field the envelope grows would never reach them) and both now go through the one builder — which immediately surfaced a real defect: a test built a 2-atom envelope and overwrote `elements` to three, leaving `atom_names` describing the old atoms, and the route's own guard caught it the moment the canonical dict was used. The third was `test_structure_envelope_protocol.py`, carrying TWO docstrings back to back (the second was dead). The remaining four are a delegating alias and one-line `struct.to_dict()` calls — not the hand-rolled XYZ parsers the helper was written against. **`flask_server`: DONE 2026-09-03, without touching a single scope.** 18 of the 20 now call one context manager, `tests/support/live_server.py::serve()`; each module keeps its own `@pytest.fixture(...)` line, because a scope is a decision about how much state a file's tests share and a de-duplication does not get to change it for them. ~230 lines and 18 now-unused `import threading` go with it. The two left alone pass a non-default app config, which is a real difference. **`_node_esm`: 24 of 47 `*_js.py` files drive it** (the row said 7 of 48), and 13 more shell out to `node` themselves | 3 done · 16 proposed |
| ~~P3~~ | ~~**B5**~~ | **DONE.** The shadowed definition is gone, and `tests/test_no_test_is_shadowed.py` makes the next one fail instead of disappearing. | 46 lines |

### Gaps in what shipped today — mine

| P | # | item |
|---|---|---|
| ~~P2~~ | ~~**M1**~~ | **DONE.**
| ~~P2~~ | ~~**M2**~~ | **DONE 2026-09-02** , and NOT the way this row expected.
| ~~P2~~ | ~~**M3**~~ | **DONE — and the fix was the OTHER direction.**

### Documents that state something false

| P | # | item |
|---|---|---|
| ~~P0~~ | ~~**D4**~~ | **DONE 2026-09-02** — and the function's OWN docstring was stale the same way, still describing the both-keys rule for a second shape that no longer exists.
| ~~P1~~ | ~~**D5**~~ | **DONE 2026-09-02**
| ~~P1~~ | ~~**D6**~~ | **DONE 2026-09-02.** Six labels, `buffer` included, in § 6.6 and § 13.3; no test hard-coded the count, so only the document was wrong.
| ~~P2~~ | ~~**D7**~~ | **DONE — the TEST was wrong, and its exemption was protecting something else.**

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

## 5d. `parse/scripts/` retired — CLOSED 2026-09-05

`parse/` reads foreign formats; a reserved block in a script molbuilder
generated is its writer's. Steps 1-3 and 5 shipped (`5f742911`, `69de4f3b`,
`e05d3c65`): `parse/scripts/` is gone, with six `TextParser` classes and
`ScriptResult`.

**Step 4 is abandoned, not pending.** Routing the private `_extract_*`
importers onto a `read_script` door was never done, and the door built for it
sat with **zero callers for three weeks** while carrying a version gate the
live readers do not have — two answers for one block. It was deleted
(`1325ca18`), not adopted. The lint step 4 proposed — *no `_extract_` name
imported across a package boundary* — is therefore not owed.

**The rule this earned now lives where it belongs**, not here:
[`model/parse.md` § 1a](?doc=model/parse.md). The 144 lines of argument that
stood here were the case for a decision already taken.

## 5e. Engine additions — a person's own engine text, as an INPUT

*(Your ruling, 2026-09-05: option 2 — a distinct input to the writer, not a
catalogue extension; **engine-specific**; and **the person is told the
consequence**. Contract-first: this section is the design. **Not started.**
No code has moved.)*

### The need

A person wants to run a SPECIFIC task on a structure that is already
optimised, using engine content molbuilder does not model: a Lua setup driving
SIESTA, a `%block` for a feature with no form field, a PySCF call.

Today the only place for that is the USER-CUSTOM zone of a relaxation deck,
which is the wrong shape twice over — it is the wrong kind of task, and (below)
the zone cannot carry the content anyway.

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
| **the writer places it** | whatever writes that task's script takes additions as an input and places them, the way `render_deck` places a `Section`'s parameters today — by the engine's authority, in a position the engine chooses, never concatenated after the walk |
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

### What this does NOT disturb — which is the point

**This is ADDITIVE. It removes nothing that ships.** The USER-CUSTOM zone, the
read-back merge, `write_script`'s round trip and `check_deck`'s reason to read
the written file all stay exactly as they are, serving the relaxation decks
they serve today. A task kind that does not exist yet cannot be a reason to
disturb one that does.

*(An earlier draft of this section claimed the design "deletes the read-back
merge" and closes the transport gap. **Withdrawn.** That followed from the
withdrawn assumption that additions would flow through the relaxation deck
path. They do not, so those mechanisms are untouched and their known defects —
the stray marker that silently drops text above it, transport having no zone at
all, § 3.5's inaccurate "byte-for-byte" — remain open on their own terms,
listed where they belong rather than as credit claimed here.)*

**What it buys instead** is that the new need lands in its own layer:

- nothing in the relaxation path changes to accommodate it;
- the mechanism is defined by what it IS (an input, engine-placed, switchable,
  unvalidated) rather than by which existing function it borrows;
- when a task kind does need it, that task brings its own writer and this
  mechanism plugs into it — no structural change to make room.

### THIS IS NOT A STAGED RUN, and must not be fitted into one

*(Your correction, 2026-09-05, replacing what this section said first.)*

**Stages exist for one reason: a calculation that needs several steps to fit
the computational resources and constraints** — coarse before tight, a ladder
that accommodates a machine. That is a different problem from this one.

A customised block is for **a specific task, on a structure that is already
optimised**. It is the mechanism molbuilder EXPOSES for a future kind of task
that needs it — not an extension of the relaxation path.

So the following, which this section asserted in its first draft, is **wrong
and withdrawn**:

> ~~"An addition needs no new home: it follows the path a parameter already
> takes — collected in Structure optimization, overridden per stage in Task
> setup."~~

That reasoned from the tabs that exist to the need, which is backwards: it took
a mechanism for a *future task kind* and forced it into the ladder built for
multi-step resource accommodation. Per-stage override is a stage concept, and
this has no stages to override across.

**What survives that correction, because it does not depend on staging:**

- an addition is an **INPUT** to whatever writes the task's script, never text
  spliced into a written file (the whole of § 5e above);
- it is **engine-specific**, and where it goes is engine knowledge;
- it carries an **include switch**, which is what makes the responsibility
  workable;
- **molbuilder does not validate it.**

**What is deliberately left open**: which task kind first needs this, and what
its own description looks like. That question belongs to that task, not to this
mechanism — and answering it early is how this would get forced into stages
again.

### Who is responsible
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

### The toggle is part of the block, and it is the whole mechanism

*(Your ruling, 2026-09-05.)* An addition carries an **include** switch in the
UI: *do you want this customised block in the final task?* That is not a
convenience and not a `prep` flag — it is the instrument that makes the
responsibility workable.

- **Off** — the task is prepared and run exactly as molbuilder would have
  written it. This is the reference.
- **On** — the same task with the addition placed.

A person compares the two and decides for themselves whether a failure belongs
to their block. They can do it through whatever they are already doing —
a benchmark trial, a debug run — because the two differ in one input and
nothing else. **That is only true because the addition is an input**; stripping
a zone out of a written deck changes more than the thing under test.

The switch belongs to each addition, not to the calculation, so several can be
carried and enabled one at a time.

### What an addition IS, per engine

Open-ended by design — a script, a variable, a setting molbuilder has not
exposed. What it means is the engine's business, and the two engines differ in
a way that matters for placement:

| engine | an addition is | why placement differs |
|---|---|---|
| **PySCF** | Python that RUNS — a call, a hook, a few statements | the deck is a program, so an addition must land where the objects it uses already exist |
| **SIESTA** | fdf settings molbuilder has not exposed, or a Lua setup (`MD.TypeOfRun Lua` + a script path) | the deck is a settings file, so what matters is libfdf's first-wins and the block structure |

So **where an addition goes is engine knowledge**, which is already where the
framework puts layout: the engine owns its `Section`/`Block` layout, and an
addition is placed by the same authority rather than by a framework rule that
would have to be right for both.

### Whose responsibility, stated plainly

**molbuilder does not validate the content and does not claim to understand
it.** It places it, records it, states the consequence when it collides with a
declared item, and gives the person the on/off pair to test with. Making the
addition correct — that it parses, that the engine accepts it, that the run
completes — is the person's.

That is a fair deal only because of the switch. Without it, "not our
responsibility" would leave someone with a failing run and no way to tell which
half caused it.

### Open, and deliberately not decided here

1. **Does the zone survive at all** for genuinely free-form text (a comment
   with no variable in it), or does that become an addition with no attributed
   keyword?
2. **What the card shows** when an addition collides — refuse, warn-and-accept,
   or show the resolved line before saving. (The consequence must be stated;
   whether it can be overridden is separate.)
3. **Ordering among additions**, when two of them write to the same section.
4. ~~Whether the include switch is per-stage~~ — withdrawn. There are no
   stages here; see the correction above.

### Before any code

This section is the contract. The measurements it rests on
(`script_emit.py:1319`'s concatenation, the 12/520/980-vs-1222 positions, the
reproduced duplicate-keyword refusal, `Parameter.writes`) are re-checkable, and
should be re-checked rather than trusted if this is picked up later.


## 5f. Architecture seams — dropped by the consolidation, recovered 2026-09-06

**How this section came to be missing.** The 2026-09-01 merge fact-checked the
roadmap's §§ 3, 4, 7.4c, 7.5, 7.8, 7.10 and 7.11 and the two audits — and
**never reached § 6, *Architecture seams***. So § 6's bullets were neither
verified nor carried, and § 5a's *"of roughly forty items… the ones that
survive are the twenty-odd rows above"* was written without them in view.
Nothing pointed this out for five days because **R3's own sentence still sent
readers to the archive**: when `roadmap.md` was archived, the line in
`docs/README.md` was re-pointed at its new path rather than at the file that
replaced it, and **53 pointers across 31 documents followed it in**. Fixed
2026-09-06; the pointers now name this file.

**Read the state column literally.** *measured* means re-derived against the
tree on 2026-09-06 and the evidence is in the row. *inherited* means carried
verbatim from roadmap § 6 and **not checked** — § 5a's lesson was that ~85% of
what a roadmap carried as open had already shipped, so assume the same here
until each is re-derived. Numbering is `S`, not `W`, because
`backend-architecture.md § 5` already has a **W1–W5** of its own and this file
already has a different **W1**; the two collided in every conversation about
them.

| # | seam | owner | state |
|---|---|---|---|
| **S1** | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — four branches, `runwrap.py:420 / 446 / 645 / 728`, the same four counted 2026-08-19; 128 engine-name literals in the file |
| **S2** | **`jobset/runstatus.py`'s warm-file table → producer-supplied inventory** | `backend-architecture.md` § 5 (**W2**) | inherited — **and likely closable**: § 4.2a's data file shipped since (S-note below), so this table may already have one source to derive from |
| **S3** | **`runtime_config`'s untyped scheduler dicts + mixed concerns** | `backend-architecture.md` § 5 (**W3**) | inherited — overlaps **S6** |
| **S4** | **Transport bypasses the framework.** Gated on a branching workflow, which has no representation today and would arrive as something a person asks for at launch, never as a field a description stores | `backend-architecture.md` § 5 (**W4**) | inherited — pairs with the parked transport-wizard `.fdf` item |
| **S5** | **`script_emit` re-filing** (its former sibling `bundle_writer` retired 2026-08-29) | `backend-architecture.md` § 5 (**W5**) | inherited |
| **S6** | **The scheduler menu is handed out as plain dictionaries**, so the typed record and the code using it never meet — how `gpu_partition` came to redirect GPU work from inside an unexamined bag | roadmap § 7.6 phase 3 | **measured partly** — the *record* is typed (`Domain`, `Device`, `Topology`, `Site` in `scheduler/record.py`); the *menu* is not (`known_machines() -> List[Dict[str, object]]`, `Domain.to_row() -> Dict[str, Any]`). Phases 1, 2, 4, 5 are done — phase 2 landed as `scheduler/admit.py`, split out so the check cannot drift from the record it checks |
| **S7** | **The preparation layer against its contract** — **P1** the enforced floor map puts `runwrap` and `jobset/prep` on floor 5; **P3** nothing names the shared package (`jobset/prep._shared_for` globs); **P5** PySCF's seam entry. P2, P4, P6 closed 2026-08-18 | `execution/script-preparation.md` | inherited |
| **S8** | **Boundary-condition contract rollout per engine** — four obligations (declare consumed labels, schema pre-fill, Stage-3A divergence warn + 3B unrecognized-label notice, verbatim emission), with **spectra the only fully-wired instance**. Each engine adoption is one item with its own pins | `engines/overview.md` § 5 | inherited |
| **S9** | **`structure_to_dict` disposition — two documents disagree.** `model/structure.md` calls it the retained web composer; `backend-architecture.md` § 2 calls it a vestigial wrapper to delete. **One decision, then align both** — this is a ruling, not a code change | both docs | **needs a decision from you** |
| **S10** | **Capability and allocation reach `prep`.** **M2a** — capability is assembled twice and never reconciled: topology and the detected partition go to `environment.json`, the `molbuilder.json` `scheduler` block goes straight to the `.sbatch` emitter, and nothing compares them, so *the record can name one partition while the header submits to another*. **M3** — a declared `qos` or `account` appears in no run-directory record. M1, M4, M5, M6 hold | `execution/project-layout.md` § 2.3.1b | inherited — **and it carries an open question that is yours**: how a person states an allocation, and whether a per-project default belongs beside the `scheduler` block |
| **S11** | **The run wrapper's string assembly** — `render_run_wrapper` is ~1780 lines emitting bash through ~295 f-strings. Recorded rather than scheduled *because neither 2026-08-17 defect entered there*: one entry point, one caller, both arriving above it. Worth doing on its own terms; **not** worth folding into a boundary fix | roadmap § 6 | inherited |
| **S12** | **GPU detection is implemented twice** — Python at prep (`.sbatch` header) and awk at launch (after a person may have edited the deck). **Two implementations are required** — one runs on a login node, the other on a compute node hours later — and the truthy set is already one constant. **The fix is a test rendering both against one deck set, never a merge** | roadmap § 6 | inherited — the shape of the answer is already decided |
| **S13** | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built** — the only occurrence in the tree is `transport/wizard.py:65`, a comment naming it |
| **S14** | **Floor 6, flat layout: one stage's verdict is still read from the whole folder.** Now sharper than when it was written — the 2026-09-05 ruling made *stages separate runs*, so reading a verdict folder-wide is reading several runs as one | `execution/architecture.md` | inherited — re-scope against the 2026-09-05 ruling before acting |

**Closed on the way in, 2026-09-06.** Roadmap § 6's *warm-file rules file*
bullet is **built** and its pointer is retired: `molbuilder/warmfiles.py` is the
one reader (`rules_for` type-scoped, `inventory` type-blind), and both engines
ship `warm-files.toml`. `job-contracts.md` § 4.2a's heading said
*"implementation tracked in [the roadmap]"* until today.

---

## 5h. The source-reading assertions — the remaining work list

*(Supersedes **B3**'s estimate. The method and the lint-vs-pin rule live in
[`process/testing.md` § 6](?doc=process/testing.md), which owns the boundary;
the instrument is `tools/classify_source_reads.py`. **Run it — do not quote a
number from here.**)*

**Measured 2026-09-06: 33 to convert, 54 to keep** — after clusters 1-3. Re-run the tool; the browser bucket GREW (9 → 12) when reading a file showed the extension had routed it wrong. Of 1,253 assertions over a
file's text, 1,153 read **generated output** — a deck, a wrapper, an
`.sbatch`, a log — which is a real property of a real product and correct as
text. 100 read a file a person wrote. Three earlier counts said 233, 256 and
173 because each used a different definition and none wrote it down.

| | | |
|---|---|---|
| **KEEP — lint** | 48 | quantifies over a class; text is the only instrument that proves absence |
| **KEEP — not ours** | 8 | vendored bundles, licences, the contact-distance data file |
| **convert — browser** | 9 | needs the CSS cascade, layout or real visibility |
| **convert — node** | 27 | the code must run; nothing needs painting |
| **convert — python** | 8 | calling the function beats reading its source |

**Order, cheapest first — each cluster mutation-tested on its own**, because
**B3.2** found eight defects in the *previous* round's replacements after a
green run had hidden all of them.

1. ~~`test_run_index_covers_every_artifact.py`~~ — **DONE 2026-09-06**
   (`29b655ec`). The directory is built by the real `prep_jobset`, so the
   shipped `mb_monitor.py` is the one under test. Four mutations caught; two
   left the retired pin strings byte-identical, which is the proof they were
   blind. The round found two defects in the replacements themselves: a
   **second consumer** of `OUR_FILE_PATTERNS` the new tests did not reach
   (`runwrap.py:490` substitutes one stem and does not expand, so `--cold`
   would have stopped protecting indexed artifacts silently), and a row-count
   assertion that **flaked 50%** because the sampler is change-gated.
2. ~~**`test_trajectory_clocks_js.py`**~~ — **DONE 2026-09-06.** The badge's
   clock choice was extracted into `badgeClocks(state)` (user-approved), beside
   `cumulativeElapsed`, which exists for the same reason; six node tests run it.
   The swap that passed **233 tests** now fails five of them.
3. ~~**`test_structure_info_bridge.py`**~~ — **DONE 2026-09-06.** Three of the
   five now run `model-jobs.js` under node with a stubbed `fetch`, so the
   REQUEST the door posts is what is asserted. Mutation-tested by **putting the
   original bug back** — reading a flat `payload.info`, the key no route sends
   — which is the mutation the retired pin passed through. **Four assertions
   are deliberately NOT converted and are reclassified browser, not node**: the
   aliasing runs inside `mountInspector` and the resets sit in `transition()`,
   a reducer that exists only once a viewer is mounted, and nothing mounts one
   headless. Their two whitespace-measuring assertions (nine embedded spaces,
   counted twice) now match on structure instead — same coverage, one less way
   to be wrong for no reason.
4. ~~**`test_task_setup_tab.py`**~~ — **MOSTLY DONE 2026-09-06.** Twelve
   assertions became one e2e reading the rendered card: every enabled stage
   offers both commands naming its own stage, the bench order is shown in
   order, and the hints say what each half is for. Mutation-tested — hardwiring
   bench to `enabled[0]` and swapping the order each fail it.

   **Two are BLOCKED and stay pins, with the blocker measured.** `_targetArg()`
   returns `""` unless a NAMED machine is chosen, and the page can only be
   driven to *(this machine)* without a named record in the live server's
   config root — so an e2e check for `--target` passes whatever the code does.
   I wrote that assertion, then measured it: adding `_targetArg()` to the launch
   line left the e2e **green**. It is deleted. A vacuous assertion is the thing
   this whole cluster exists to remove, so the pin stays until a fixture can
   supply a named target — recorded in the tool's overrides, not just here.

   **Three CSS assertions remain, and are browser work** (`.ts-state[hidden]`,
   `.ts-facts[hidden]`): cascade questions jsdom cannot answer.

## 5i. The projects root — audited 2026-09-06, and where it is NOT one door

*(Asked because a fixture I wrote assumed the tree's location and silently
opened the wrong folder. The suspicion was right in shape and wrong about
where.)*

**Production is one door, verified hop by hop.** `projects.projects_root()`
is the single definition and the only place `$MOLBUILDER_PROJECTS` is read;
it feeds `Capabilities.file_picker_roots()`, which the browser gets from
`GET /api/files/roots`, which reaches `setProjectsRoot()` — one caller — and
every consumer then asks `projects.getProjectsRoot()` / `getCurrentDir()`.
**Zero hand-rolled joins**: no `/ "projects"` anywhere in `molbuilder/`, and
no second reader of the environment variable. `find_projects_root(start)` is
not a second answer — it answers a **different question** (*which tree does
this calculation live in*, walked up from the folder) and its docstring
argues why, naming the 2026-08-21 defect where anchoring on the process's
working directory made a template resolve to a folder that does not exist on
the cluster.

**The tests are where it is not one door — and this is recorded, not
discovered.** `isolated_projects_root` is **opt-in, not autouse**, and its own
docstring says why: *"A blanket autouse override broke 13 of those: they
construct a tree and then ask a route to serve it, and the route's root guard
was pointed somewhere else entirely."* So **13 sites across 7 files build
`ROOT / "projects/_t_…"` — inside the developer's REAL tree** — because that
is where the route's guard resolves. They clean up in a `finally` and no
residue is present today, but a crashed run leaves a folder in a directory
that is the user's own data.

| | |
|---|---|
| **What is fine** | production resolution, on both sides of the wire |
| **What is open** | a browser-driven test cannot build its tree anywhere but the real root, so isolation and route-serving are mutually exclusive today |
| **What would close it** | the live server taking a projects root — `serve()` already accepts a `config` dict and passes it to `create_app` — so a test can point the whole app at `tmp_path` instead of pointing one env var at it and leaving the route guard behind |

**Not scheduled here** — it is a testability change to the app's config
surface, and that is a decision rather than a defect.

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
| `roadmap.md` | R3's old home. 1770 lines, ~85% closed work never struck; its live items are **E7–E11, C2, C3, W12–W15, N1–N4** — **plus § 6's architecture seams, which the 2026-09-01 fact-check never reached and this merge dropped**; recovered 2026-09-06 as **§ 5f**, **S1–S14** |
| `archive/2026-09-01-audit-2026-08-21-fullstack-review.md` | its open list became roadmap § 7.5, and § 7.5 is now empty — verified |
| `archive/2026-09-01-audit-2026-08-28-full-review.md` | O1–O3 closed in the document; O4's general case is **N1**, O5 is **N3**, its uncovered lane is **E11** |
