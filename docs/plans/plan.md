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
| **E1** | **Benchmark iteration count, settable per calculation.** No field exists on `task.json` or `Resources` — confirmed 2026-09-07. **Bigger than the row says:** the archived design was a one-point `bench` entry overriding the pin, and that path is now explicitly closed — `_cli.py:1961` `pins = {**declared_pins, **_MEASUREMENT_PINS}`, with the comment that measurement pins *"must win over any declaration — one-point declarations and value-axis coordinates alike."* So this needs a written precedence rule reversed, not a field added | `bench-and-junction` § 2.1 | not started |
| ~~**E2**~~ | **DONE 2026-09-03 by your ruling — *use the exact one when it's there, fall back to the csv*.** `parse_utilisation` is the one door: the monitor's `[UTIL-SUMMARY]` means (averaged over EVERY tick) where it wrote them, `util.csv`'s time-weighted reconstruction (a ≥10%-change-gated subset) where it did not — which is what a KILLED trial leaves, the one a benchmark most needs to read. | `bench-and-junction` § 2.2 | done |
| ~~**E3**~~ | **DONE 2026-09-03 — named, with the tell that detects it.** | `bench-and-junction` § 3B | done |
| ~~**E4**~~ | **DONE 2026-09-03 — but this row overstated what shipped, corrected 2026-09-07.** The `[MACHINE]` line is written first, unconditionally, on every path. It carries `node`, `cores`, `mem_gb` and `gpu` (`monitor.py:555`) — **there is no CPU model field**, and `gpu` is a device model, not one. If the CPU model is wanted, it is a new item, not a done one. | `bench-and-junction` § 3D | done (narrowed) |
| ~~**E5**~~ | **DONE — the run already happened, 2026-08-28/30; the row outlived its own evidence.** 57 files under `projects/` carry `[MACHINE]` lines from live Sol jobs across 8 distinct nodes, in the exact format `machine_line()` emits today (`monitor.py:555`). And `lightwork`'s cap is answered: four probed `environment.json` files record `max_cpus_per_job: null` **with the key present**, which is the contract's *asked and unstated* — the suspected 8-core cap does not exist. **Why it survived:** the evidence is in the gitignored `projects/` tree, so no document could see it. A row whose proof cannot be checked by reading the repo needs its outcome written into a document at the time, or it stays "blocked" forever | `machine-identity` § 6 | done |
| ~~**E6**~~ | **DONE 2026-09-03 by your ruling — a reference, not a default.** | found 2026-09-01 | done |
| **E7** | **D7's cluster half** — the prep→submit→watch loop for a **run** through SLURM on Sol. **Not blocked on access, re-derived 2026-09-07:** the machine was used on 2026-08-30 and every Sol ledger entry is the *bench* lane (real `sbatch` ids, real nodes). No `kind: "run"` submission exists on any Sol tree — the only ones are workstation `direct` mode, which is the half D7 already passed. So this is a session someone has to sit down for, not a door that is shut | roadmap § 1 | open |
| ~~**E8**~~ | **ALREADY DONE — verified 2026-09-03, not changed.** | roadmap § 1 | done |
| ~~**E9**~~ | **BOTH HALVES CLOSED 2026-09-03.** | roadmap § 6 | done |
| **E10** | **NEEDS A RE-SCOPE, not implementation — re-derived 2026-09-07.** The detection is **built**: `parse/engines/_run_ending.py` is the one marker table (`abnormal_termination` → `stopped`, OOM markers, `SCF_NOT_CONV`), `scan_ending()` returns `(run_state, scf_converged, error_message)`, and `summarize.py:217` + `cli.py:2771` consume it — the zero-exit case being the whole point. What is NOT built is the monitor half, and **deliberately**: `monitor.py:138` says *"NO COMPLETION MARKERS HERE"* and `job-contracts.md:214` now rules the monitor follows the launcher's PID *"rather than guessing from output markers"* — which contradicts this row's own "belongs in `mb_monitor.py`". The real gap is narrower: **the monitor's finish report carries no convergence fact.** Decide whether it should before writing anything | roadmap § 4 | open (re-scope) |
| **T1** | **A LIVE POLL THAT MUST REBUILD DOES NOT UPDATE THE MOVIE — found 2026-09-07.** The append path is fine; the full-rebuild path is not. Reproduced: move the frame at `oldLen - 1` (the one `_frameEqualAt` reads) and grow the feed 4 → 6, so `canAppend` refuses and `applyNewData` takes the `else` branch. The status line then says *"Loaded 6 … frames"* and the frame bar still holds **4** — the feed's count and the movie's disagreeing, which `core.js` itself names as bug **#35**. The trigger is ordinary: a frame that was still being written when the last poll caught it, and has since settled. **Two things hide it, each worth its own look:** `setStatus` is a no-op on `/results` (`if (!document.getElementById("status")) return;`), so `rebuildModel`'s *"Viewer failed to load the run"* reports into nothing on the page the inspector lives on; and *"Loaded N frames"* is written by `applyNewData` from the FEED's count while `rebuildModel` runs unawaited beside it, so the tab can claim frames it is not showing. Recipe at the foot of `tests/test_inspector_registry_e2e.py`; it blocks the last three source pins in `test_structure_info_bridge.py`, which is how it surfaced | found 2026-09-07 | open |
| **E12** | **The Methods paragraph is still a placeholder — DROPPED BY THE CONSOLIDATION, recovered 2026-09-07.** Roadmap § 2 listed four TranSIESTA items; plan.md carried three (one became **W10**) and lost this one. `transport/transiesta.py:1015` still emits *"(Full Methods paragraph deferred to the follow-up release…"* and `transport/engine_base.py:135` calls the fragment *"composed by the future Methods generator"* — pointing at the archived roadmap, which is where it went to die. The work: interpolate the record's real run parameters instead of the placeholder | roadmap § 2 · `transport-design` § 6 | open |
| **E13** | **The first real junction walk — BDT–Au, workstation then Sol. DROPPED BY THE CONSOLIDATION.** Named in the roadmap and again in `transport-design` § 7's *"order of proof"* as the run that follows P6. plan.md has the machine-blocked infrastructure (E5/E7) and the browser and deck walks (W12/E11) but no row for the transport composite's first real science run | roadmap · `transport-design` § 7 | open |
| **E14** | **Two bibliography keys — Reed 2006 and Stokbro 2003 — are cited nowhere in `science/references.bib`.** Verified absent 2026-09-07. Mechanical, small, and dropped by the consolidation | roadmap § 4 | open |
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
| **W4** | **Guards 1 and 2 (step F)** — one home including elements; a page sheet contains only its own tier. Guards 3 and 4 landed. **Both remaining gaps are now provable, 2026-09-07:** guard 1 is absent *by an explicit skip* — `test_css_no_duplicate_selectors.py:150` reads `if "." not in norm: continue`, so element-only selectors are exempt by construction; guard 2 has no test at all | `css-system` § 4F | partly |
| **W5** | **The inspectors module's appearance still lives in `results/style.css`.** **Re-derived 2026-09-07 and the number was prose:** 70 was `grep -c inspector`, which counts the file's 200-line comment header and a hierarchy diagram. Comments stripped and classified by who EMITS each class: **22 module-owned rule blocks**, and **6 of those are dead** — `.inspector-section`, `-section-header`, `-section-body`, `-section-hint`, `.source-body-error`, `.structure-error` have **zero emitters anywhere** in the repo and are deletable outright, which this row never said. Three sheets are already repatriated. *"Renders unstyled elsewhere"* is **latent, not reachable**: `registry.js` is script-tagged by `results.html` only, so the css-system doc's premise (it also loads on /molbuilder and /spectra) is stale. Also: `inspectors/bench-summary.css` is missing from the boundary guard's `MODULE_SHEETS`, so that guard treats a module sheet as a page sheet | `css-system` § 7.0 | partly |
| **W6** | **The editor module.** The loader half is confirmed and accurate: `lib/codemirror-load.js` is the one loader, two of three surfaces import it, and `lib/inspectors/markdown.js` still hand-rolls its own pair — *definitions* at `markdown.js:31` and `:38` (this row cited only the call site). **The sheet number was wrong twice over, re-derived 2026-09-07: 21 rule blocks / 60 declarations, not 30 and not 40.** The original 25+4+1 was never reproducible as a block count either — `projects-sidebar.css` has held 16 CodeMirror blocks at every commit back to 2026-08-28. The caps (1500-line selection, 1 MB view-only) are on `preview.js` alone, confirmed | `editor-module` | partly |
| **W7** | **I7 close-out** — the browser walk of Results export → cite → describe → prep. I1–I6 done | `structure-info` I7 | open |
| **W8** | **Caller-less endpoints — decide the ROLE, do not just delete.** All three confirmed still present with no JS and no non-test caller, 2026-09-07. `/api/docs/list` (`docs.py:415`) — the "second answer" framing is exact: the tab calls `/api/docs/toc` and `/api/docs/read`, never this. `/api/checkpoint/config` (`checkpoint.py:239`) — the write half is genuinely gone, and it has **zero** test references; the 45 `checkpoint_config` hits in `tests/` are a pytest fixture of the same name, a collision that will mislead the next reader. `/api/selection/atoms` (`selection.py:238`) — **3 test files call it, 2 only name it in prose**, not the "five" this row claimed | `structure-info` § 3 | open |
| **W9** | **Transport viewer default orientation** — a MolView camera-door contract question | `structure-info` § 3 | open |
| **W10** | **Results transmission inspector** — the record exists, the reader does not | `structure-info` § 3 · roadmap § 2 | open |
| **W12** | **The live browser walk-throughs** — checkpoint swap at narrow and wide widths, per-tab reload round-trips, a real Data/Image export, click-selection on frames ≥ 1. Carried since the archived molview-and-checkpoint plan | roadmap § 0.3 | open |
| **W13** | **Raw px/rem literals — re-derived a THIRD time, 2026-09-07, and the definition finally holds still.** 160 / 740 reproduce exactly, but only because the regex reads raw file text *including comments*. Counting literals **in declarations**: **133** across the eight page sheets, **650** in `lib/`. Two things the row hides: `lib/tokens.css`'s 44 literals ARE the scale definitions — the token layer, not violations — and `lib/molview/molview.css` alone is **252**, 39% of the whole `lib/` figure. So "lib/ carries 740" is really "MolView carries 252, and the rest of lib carries ~400". 777 → 384 → 160/740 → 133/650 are four scopes, not four measurements | roadmap § 7.4c | partly |
| **W14** | **A web plan view and a per-stage status roll-up.** The web *describes* a staged calculation (Task setup) and *observes* runs (Results); neither shows the plan as a whole | roadmap § 6 | open |
| **W15** | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. Plus routing the CLI through the shared codec and exercising the last annotation-channel kind. Confirmed 2026-09-07: `lib/molview/` has **no `_seal.js`** where `spectrumchart/` and `vibrationview/` both do, and seven inspector scripts on `results.html` are still classic `<script defer>` against two on `type="module"` | roadmap § 3 | partly |
| **W16** | **Spectrum-tab display settings do not survive a reload — DROPPED BY THE CONSOLIDATION.** The `uiPrefs` bucket holds six real knobs (mode filter, sort column and direction, broadening, animation amplitude and speed) and the save/restore is never wired: `lib/spectra/core.js` has **zero** actual `sessionStorage` calls, only comments proposing the key `molbuilder.results.spectra.uiPrefs.v1`. The contract's stated promise — survive a *file switch* — IS met, so this is the enhancement the code marks "PR 3.1?", not a violation. Trajectory leaves the same bucket empty on purpose and is not affected | roadmap § 3 | open |
| **W17** | **Flat-shape citation ergonomics — DROPPED BY THE CONSOLIDATION.** `structure-info-plan` § 5.6's backlog had five entries; three became W8/W9/W10 and this was not carried. A flat calculation folder holds several stage decks and one shared `.XV`, so citing it refuses as ambiguous; the recorded refinement was to disambiguate by the concluded marker's stage. No such logic exists | `structure-info` § 5.6 | open |
| **W18** | **"Modify functions (Molbuilder tab)" — item ZERO of `structure-info-plan` § 5.6's own priority order, annotated *"user calls it higher priority; not yet described."*** It never reached plan.md in any form, and is still described nowhere. Blocks nothing technically; it needs a description before it can be planned | `structure-info` § 5.6 | needs describing |
| ~~**W11**~~ | **DONE — verified 2026-09-06 across both halves.** | `structure-info` § 3 | done |

## 4a. Open — needs a decision from you

| # | item | from | state |
|---|---|---|---|
| ~~**N1**~~ | **CLOSED 2026-09-03 — the rule is written and there is nothing to fix.** | audit 08-28 O4 | done |
| ~~**N2**~~ | **ALL NINE RETIRED 2026-09-03** | roadmap § 4 | done |
| ~~**N3**~~ | **ALL THREE WERE ALREADY FIXED — the row was stale, verified 2026-09-03.** | audit 08-28 O5 | done |
| ~~**N5**~~ | **DECIDED AND BUILT 2026-09-07 — two flags, because they invalidate different things.** One `structure_modified` was set by geometry ops, cell ops AND label writes alike, which made it unusable by the only reader that wanted it: acting on it meant telling a person their mesh cutoff might not apply because they renamed an electrode, which on a junction is routine. Now `structure_modified` (geometry/cell — invalidates the inherited mesh cutoff and transverse k-mesh, both functions of the CELL) and `labels_modified` (a label write — the settings stand, but the electrode/device partition the categorical sort reads IS labels). Both warn, neither refuses. Surfaced at the citation line a person reads when choosing and again on the prep path through `validation.report`. Rule first, in `molview.md` § 8.4a; the source-text pin that counted the calls is now a lint over the edit-landed sites, and which flag each door raises is driven under node | `structure-info` § 5.5a | done |
| **N4** | **The science-validation tail** — checks deferred with recorded rationale in `science/pseudopotentials.md` § 3. **Now four, not five, and each has a different answer**, re-derived 2026-09-07: *valence-charge correctness* (is Au run with the right 11- vs 19-electron valence — the count is parsed for sanity, never judged), *ghost states* (needs generation-level analysis, not header inspection), *transferability* (deliberately delegated to PseudoDojo's δ-factor — arguably not ours to check at all), and *basis ↔ pseudo consistency* (marked "Deferred." with no rationale at all — the only one with nothing behind it). The fifth, **mesh-cutoff adequacy, was already implemented** and the doc contradicted its own § 2a.1 saying otherwise; corrected 2026-09-07 | roadmap § 5 | needs a home |

## 5. Open — documentation drift

| # | item | from | state |
|---|---|---|---|
| ~~**D1**~~ | **DONE 2026-09-03 — closed, and now guarded.** | `consolidated-cleanup` § 7 | done |
| **D2** | **Tests with no target, remainder.** The two files with zero test functions were **checked and left** — each is a signpost recording where retired coverage moved, which is a service, not residue. **Re-derived 2026-09-07 with the definition stated:** 417 test files, 6,529 test functions; 2 files with no test function, **0** empty test bodies, and **10** `Test*` classes that collect nothing — not the 5 recorded. Eight of the ten are in `test_results_state_contract_js.py` and its spectra sibling, stating pins in the present tense while holding nothing. Three named remainders still cannot fail: `test_doc_claims.py:92` (loop filters on a string that appears 0 times in its target), `test_monitor.py:342` (`assert callable(fn)` on a `def`), `test_vibration_form_honesty.py:34` (`STILL_OPEN = {}`, iterated empty) | `consolidated-cleanup` § 9 | partly |
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
This section is the plan, and the caller list below is the completeness check
— the requirement is that nobody is left behind.)*

**TWO DIFFERENT THINGS SHARE THIS NAME, so read the state carefully.** The
`JobDirParser` that *existed* — an eleven-field `JobResult`, ten of whose
fields had no reader anywhere and whose eleventh was reached by parsing every
result file to build plots and then discarding them — **was DELETED
2026-09-04**, replaced by `job.run_status`. `parse/dirs/__init__.py` says so:
*"No DirParser ships today."* That work is done and is not what follows.

**What follows is a NEW consolidation, and it is not started** — verified
2026-09-07, not taken from this row: `_resolve_run_directory` and `_engine_of`
are still their own readers in `web/blueprints/watch.py`, and `run_status` is
still its own in `jobset/runstatus.py`.

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
| `run_status(dir)` | defined `parse/dirs/job.py:157`; called from `jobset/runstatus.py:200` ×1 | `.status` |
| `_resolve_run_directory(dir)` — the 4-rung chain | `web/blueprints/watch.py` ×1 | `.openable` + `.attempts` |
| `_engine_of(...)` | `web/blueprints/watch.py` ×3 *(`:797`, `:886`, `:930` — the table said ×4; re-derived 2026-09-07)* | `.engine` |
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
2. Move `runstatus` (1 site), then `watch` (**6**, not 9 — the table above
   sums to 6 and always did), then `summarize` (5).

   **`run_status` is already through the door**, which the prose above got
   wrong until 2026-09-07: it is defined in `parse/dirs/job.py` and merely
   CALLED from `jobset/runstatus.py`. Step 2 moves a caller, not a reader.
3. Delete the absorbed functions and sweep their documents.
4. Re-run the caller map above and require it to come back empty.

---


## 5d / 5i / 5j — CLOSED, and archived 2026-09-07

`parse/scripts/` retired, the projects root made one door in tests as well as
production, and `parse/dirs/_assembler_helpers` deleted. All three shipped;
each earned a rule, and each rule now lives in the document that owns it —
`model/parse.md` § 1a, `process/testing.md` § 2a, and
`execution/running-a-job.md` § 4.2. The sections themselves are in
`archive/2026-09-07-plan-closed-sections.md`.

**§ 2a of the testing doc was written on the day of the archive move**, not
before: until then the projects-root rule existed only in the closed plan
section, which is exactly what the substance-first rule forbids.

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

**ALL FOURTEEN RE-DERIVED 2026-09-06** *(user: "verify the 14 seams in
5f")*, and the ratio is the finding again.

> **⚠ THIS PARAGRAPH CONTRADICTED ITS OWN TABLE UNTIL 2026-09-07**, on four
> rows of fourteen. It counted **S8** and **S11** among "confirmed open" —
> the table marks both **WITHDRAWN**, and I wrote those cells. It counted
> **S14** among "already closed"; the table marks it **OPEN** and says the
> 2026-09-05 ruling made it sharper. And it listed **S12** as open where the
> cell reasons it closed. A summary that disagrees with the evidence
> directly beneath it is worse than no summary: this is the shape of the row
> that survived a whole day in § 4 because nobody re-derived it. Corrected
> below from the cells, not from memory.

**Three are CLOSED** (S2, S5, S12), **two are WITHDRAWN** as not-work-items
(S8, S11), **one is mostly wrong** (S4), **six are confirmed OPEN** (S1, S3,
S6, S13, S14, and S10's own subject), **S9 is now DONE** — the two documents
stopped disagreeing on 2026-09-06 and `backend-architecture.md` § 2
retracts the claim in writing — and **two state cells describe the wrong
seam**: S10's answers S9's question, and S14's describes S12's GPU guard and
asserts it "does not exist" when S12's own cell shows it does. **S10's and
S14's real states are unrecorded.** The line counts S11 quoted — `render_run_wrapper` grew to 2,134 lines and 499 f-strings
while carried as a stable ~1,780 / ~295. Three were measured when the section
was written (S1, S6, S13) and stand. Two halves are still unchecked and say so
(S7's P1/P5, S10's M2a).

**Read the state column literally.** *measured* / *verified* means re-derived
against the tree and the evidence is in the row — § 5a's lesson was that ~85% of
what a roadmap carried as open had already shipped, so assume the same here
until each is re-derived. Numbering is `S`, not `W`, because
`backend-architecture.md § 5` already has a **W1–W5** of its own and this file
already has a different **W1**; the two collided in every conversation about
them.

| # | seam | owner | state |
|---|---|---|---|
| **S1** | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — four branches, `runwrap.py:420 / 446 / 645 / 728`, the same four counted 2026-08-19; 128 engine-name literals in the file |
| **S2** | **`jobset/runstatus.py`'s warm-file table → producer-supplied inventory** | `backend-architecture.md` § 5 (**W2**) | **CLOSED — verified 2026-09-06.** `runstatus._warm_files()` calls `_carry_inventory(engine)`, the § 4.2a data file's own reader; its docstring records retiring the module-level dict that stood there |
| **S3** | **`runtime_config`'s untyped scheduler dicts + mixed concerns** | `backend-architecture.md` § 5 (**W3**) | **OPEN — verified 2026-09-06.** `runtime_config._validate_scheduler` still returns `Dict[str, Any]`; overlaps **S6** |
| **S4** | **Transport bypasses the framework.** Gated on a branching workflow, which has no representation today and would arrive as something a person asks for at launch, never as a field a description stores | `backend-architecture.md` § 5 (**W4**) | **MOSTLY WRONG — verified 2026-09-06.** Transport preps through the jobset door (`prep.py::_prep_transport`, reached from `prep_calculation`), and `transport/wizard.py` RENDERS (`render_electrode_fdf`) rather than writing — no `write_text` in it at all. Re-scope or retire; the parked wizard task needs re-deriving before it is acted on |
| **S5** | **`script_emit` re-filing** (its former sibling `bundle_writer` retired 2026-08-29) | `backend-architecture.md` § 5 (**W5**) | **CLOSED — verified 2026-09-06.** `script_emit` sits in `backend-architecture.md`'s **Data management** column now; W5's row is the stale half |
| **S6** | **The scheduler menu is handed out as plain dictionaries**, so the typed record and the code using it never meet — how `gpu_partition` came to redirect GPU work from inside an unexamined bag | roadmap § 7.6 phase 3 | **measured partly** — the *record* is typed (`Domain`, `Device`, `Topology`, `Site` in `scheduler/record.py`); the *menu* is not (`known_machines() -> List[Dict[str, object]]`, `Domain.to_row() -> Dict[str, Any]`). Phases 1, 2, 4, 5 are done — phase 2 landed as `scheduler/admit.py`, split out so the check cannot drift from the record it checks |
| **S7** | **The preparation layer against its contract** — **P1** the enforced floor map puts `runwrap` and `jobset/prep` on floor 5; **P3** nothing names the shared package (`jobset/prep._shared_for` globs); **P5** PySCF's seam entry. P2, P4, P6 closed 2026-08-18 | `execution/script-preparation.md` | **P3 CLOSED — verified 2026-09-06**: `_shared_for` calls `seam.shared_package(base)`, and the code names the glob it retired as *'an accident of which suffix the glob happened to name'*. P1 and P5 not re-derived |
| **S8** | **WITHDRAWN 2026-09-06 — this is not a work item.** `engines/overview.md` § 5 is titled **"Adding a new engine"**: it is the checklist a FUTURE engine must satisfy to join the sidecar contracts. It names no existing engine that fails it, and there is nothing to roll out until one is added. | `engines/overview.md` § 5 | **Withdrawn.** The roadmap bullet framed a how-to-add checklist as a rollout "with spectra the only fully-wired instance", which reads as pending work; spectra is simply the engine that has sidecar labels. I then compressed it further, into "four frozen-atom rules only one engine follows" — off one grep line that happened to mention `_seed_frozen_indices_from_sidecar`, which § 5 cites once as an example to mirror. Two layers of drift, the worse one mine. |
| **S9** | **`structure_to_dict` disposition — two documents disagree.** `model/structure.md` calls it the retained web composer; `backend-architecture.md` § 2 calls it a vestigial wrapper to delete. **One decision, then align both** — this is a ruling, not a code change | both docs | **needs a decision from you** |
| **S10** | **Capability and allocation reach `prep`.** **M2a** — capability is assembled twice and never reconciled: topology and the detected partition go to `environment.json`, the `molbuilder.json` `scheduler` block goes straight to the `.sbatch` emitter, and nothing compares them, so *the record can name one partition while the header submits to another*. **M3** — a declared `qos` or `account` appears in no run-directory record. M1, M4, M5, M6 hold | `execution/project-layout.md` § 2.3.1b | **CLOSED 2026-09-06 — there was no decision to make.** `web/web-api.md` owns the HTTP API and documents the legacy aliases as part of the response shape, with no deprecation; `model/structure.md` agreed. `backend-architecture.md` was the single restatement out of step, and is corrected. **I asked about this three times instead of opening the contract that owns the surface** — the rule is to find the owner first, and a disagreement between two documents is a question about which one owns the concept, not a question for the user. |
| **S11** | **WITHDRAWN 2026-09-06 — the measurement was meaningless** *(user: "you should use framework and understand what the addition of those code is… instead of counting beans")*. A script generator SHOULD grow as engines gain parameters; line and f-string counts say nothing about whether it is sound. Asked properly instead: `render_run_wrapper` has **18 named section emitters** (`_cold_restart_block`, `_gpu_loadbalance_block`, `_phys_cores_probe_block`, …), uses the existing doors (`script_emit`, `warmfiles`, `identity`, `resolve`) and hand-rolls **zero** path joins. It grew by gaining sections. The framework holds. | roadmap § 6 | **Withdrawn.** What the right question *did* find is recorded rather than lost: the run's basename is read back by four hand-rolled parsers (Python and awk, per engine). Compared on twenty-odd inputs — they differ only where `_validate_basename` already refuses, so nothing is broken. The design behind that is now written where it is read (`job-contracts.md` § 2.5b, `_validate_basename`, both extractors, the wrapper's branch) because it had been re-derived from the regexes more than once. The two Python extractors have **no production caller** and are a deletion candidate. |
| **S12** | **GPU detection is implemented twice** — Python at prep (`.sbatch` header) and awk at launch (after a person may have edited the deck). **Two implementations are required** — one runs on a login node, the other on a compute node hours later — and the truthy set is already one constant. **The fix is a test rendering both against one deck set, never a merge** | roadmap § 6 | **CLOSED 2026-09-06 — the guard is written.** Twenty decks (both spellings, every truthy and falsy value, last-wins including across spellings, case, leading whitespace, a longer token, a commented-out line, a bare keyword, a trailing comment) driven through **both** detectors: the Python one directly, and the wrapper's own awk + `case` block **extracted from a rendered wrapper**, so the test exercises what ships. They agree on all twenty — there was no live divergence, and now one cannot appear silently. Mutation-verified twice: dropping the older `Diag.ELPA.UseGPU` spelling from the awk alone, and making the awk take the FIRST occurrence, each fail with the deck they diverged on named. The row's own resolution held — a test, not a merge. |
| **S13** | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built** — the only occurrence in the tree is `transport/wizard.py:65`, a comment naming it |
| **S14** | **Floor 6, flat layout: one stage's verdict is still read from the whole folder.** Now sharper than when it was written — the 2026-09-05 ruling made *stages separate runs*, so reading a verdict folder-wide is reading several runs as one | `execution/architecture.md` | **OPEN — verified 2026-09-06.** Python at `runwrap.py:1594` (`_fdf_requests_gpu`, called at 1786), awk at `runwrap.py:2320`, and the emitted comment states they share one rule: *'BOTH keyword spellings, SIESTA fdf_get's truthy set, LAST occurrence wins'*. **The guard this row proposes does not exist** — `test_siesta_use_gpu.py` drives the Python half alone |

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

**Measured 2026-09-08 after the sweep: 0 to convert, 57 to keep** (was 64 to
keep, 0 to convert, on 2026-09-07 — the "3 to convert" that stood here was a
mis-read of a run whose overrides had come unanchored; see below).

**The sweep, 2026-09-08.** 59 source pins retired across the task-setup lane
and the rest of the suite, 7 restored after `testing.md` § 3a was re-read —
it blesses artifact-property checks by name ("a stylesheet declares no raw hex
colours, or no duplicate selector"), and the first pass had cut on the surface
feature *reads a shipped file* rather than on what the assertion ASKS of it.
Of the 10 assertions that left this population, 7 were in the KEEP bucket by
the SYNTACTIC pass with no override recorded — which this tool's own preamble
says is a first pass and not a verdict ("Every site was then READ, and the ones
the rules got wrong are named in `_OVERRIDES`"). What replaces them is
[`process/code-audit.md` § 1c](?doc=process/code-audit.md), four classes of
silent failure rather than 59 spellings.

**AND THE OVERRIDES CAME UNANCHORED, which is the finding worth keeping.**
`_OVERRIDES` is keyed by `(file, line number)`, and each entry is a reason
someone wrote after READING that site. Deleting tests moved the lines, so two
override entries silently stopped matching — one displaced by two lines, one
orphaned when its test went — and the sites fell back to the syntactic pass.
The tool then reported "3 to convert" for assertions nobody had reclassified.
Re-anchored 2026-09-08. **A verdict anchored to a line number is a verdict
that edits can move without saying so**, and this population is edited by
definition. Re-keying it — by test function name plus the asserted text, or
any anchor that survives an edit — is open work, and until it is done, editing
a file in this population means re-running the tool and checking its overrides
still point where their reasons say. Of 1,220 assertions over
a file's text, 1,153 read **generated output** — a deck, a wrapper, an
`.sbatch`, a log — which is a real property of a real product and correct as
text. 67 read a file a person wrote, and 64 of those are lints. Three earlier
counts said 233, 256 and 173 because each used a different definition and none
wrote it down.

**The verdicts moved as much as the code did, and both directions matter.**
The browser bucket GREW twice when reading a site showed the extension had
routed it wrong; it then SHRANK by six when reading showed the opposite —
*which sheet defines a vocabulary* is not a question a browser can answer,
because the cascade yields a computed value and never the file it came from.
Every reclassification carries its reason in the tool's override table, so a
verdict can be argued with instead of taken on trust.

| | | |
|---|---|---|
| **KEEP — lint** | 56 | quantifies over a class; text is the only instrument that proves absence |
| **KEEP — not ours** | 8 | vendored bundles, licences, the contact-distance data file |
| **convert — browser** | 3 | all three in `test_structure_info_bridge.py` — see the entry below |

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
6. ~~**`test_trajectory_csv_redaction_js.py`**~~ (2, node) — **DONE
   2026-09-06, and it turned up a circular one.** The ten redaction cases
   EXTRACTED the function's source by anchored slicing (`marker_end_token`
   was `"        return p;\n    }"` — eight spaces and a newline) and ran that
   text; a separate assertion pinned that `core.js` exports
   `_redactSourcePath` *"so this test can drive the function"*. **No test drove
   it that way** — removing the export failed only the string pin. Now the
   module is loaded through `tests/_node_esm.run_node` with `static_root`, so
   its browser-absolute `/static/…` import resolves, and the function is
   called on the namespace production uses. The export pin is deleted as
   genuinely redundant: **verified** — removing the export now fails the real
   cases. Gutting the redaction fails six of ten.

   *`core.js` loads under the shared harness*, which the file had assumed
   impossible (*"the module's IIFE captures `window`/`this` which Node doesn't
   have"*). That is what `globals_js` is for, and it opens the same route for
   the remaining `core.js` pins.

5. ~~**`test_launch_ask_mode.py`**~~ (4, python) — **DONE 2026-09-06.** The
   query cap now ASKS about 30 trials against a cap of 24 and reads the answer;
   the `--test-only` check compares what `ask` reports against what `submit`
   plans, using `JobResult.command` — no spying, and that is the surface a
   person sees. Mutation-verified: dropping the `"not asked"` result instead of
   naming it, and appending the flag instead of inserting it, each fail.
   **The first draft SKIPPED** — the deck lacked the restart group the
   submission door verifies — which is the read-green-never-ran shape; fixed
   with a real deck rather than left as a skip.

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

7. **2026-09-06/07 — the rest, and one that did not land.** Twelve more
   converted, each mutation-tested. Two were more than test work:

   * **A DEFECT THE PIN WAS HIDING.** `setMachine()` never re-ran
     `renderNext()`, so choosing a remote machine left the copied command
     without `--target` — it preps for THIS machine while the card names
     another, invariant **C1** in `preparing-for-another-machine.md`. The pin
     asserted `"_targetArg()" in src`, true throughout. Fixed.
   * **A CHECK GREEN BY COINCIDENCE.** `#slab-orthogonal` has no wrapping
     `<label>`; `html.rfind("<label", 0, i)` landed on an unrelated field's
     label fifteen lines up, and the failure message described a DOM that does
     not exist.

   **The "BLOCKED" verdict on `--target` in entry 4 above was wrong**, and the
   reason it was wrong is worth more than the fix: `support.live_server` runs
   the app on a THREAD in the test process, so `$MOLBUILDER_CONFIG_DIR` is
   shared and a record written by a test is a record the route serves. The
   fixture is six lines. **A blocker is a measurement, not a memory** — this
   one was recorded confidently and held for a day.

   **`test_structure_info_bridge.py`'s three stay pins**, and this is the
   second time that entry has been re-examined. A Playwright test for APPLY's
   keep-on-`undefined` was written, passed, and was **vacuous**: `rebuildModel`
   runs at LOAD as well as after a poll, so a mutation that empties the store
   fails on the *before* assertion while proving nothing about any poll. On
   that reading it sat green through the guard being deleted three ways.

   **It stalled on something that is probably not a test problem — see T1
   below.** The facts the next attempt needs are recorded at the foot of
   `tests/test_inspector_registry_e2e.py`: how to build a run that states a
   contract without SIESTA (one `.fdf` beside a generic `*_geom_optim.xyz`),
   why an appended frame tests nothing (a strict tail extension takes
   `addFrames` and never rebuilds), and why "wait until a poll has happened"
   is already true before the file is touched (polling starts at mount).

   **Also converted:** the CSV export's redaction (downloaded and read, not
   grepped for an indentation), both inspector cores' listener teardown
   (measured at the lifecycle scope, because a global add/remove spy counts
   Plotly, 3Dmol and the projects sidebar), the recommended-value panel on
   both engines, the bench grid's out-of-order reply, the notify one-door
   (issued on the tab, then on the CLI, against one file), prep's trial path
   (checked on disk), and all three spectrumchart box traps.



## 5k — CLOSED 2026-09-08, and in git rather than a second file

The paths framework's first program, M1–M8: every name molbuilder composes got a
door that FINDS it, and every counter-keyed name got composed by its owner.
Twenty-four handcrafted searches → 0, eight hand-built counter names → 0, both
guarded (`tests/test_path_framework.py`). Shipped as `6822f639` and `981534b9`.

The rule it earned lives in the document that owns it —
[`execution/project-layout.md` § 4.5](?doc=execution/project-layout.md).

**Its METHOD did not survive its own last day.** Closing each asymmetry with a
door per question left ~40 public functions answering four questions four ways,
five of them added on 2026-09-08 to fit individual call sites. **§ 5l is the
live plan.** The 367 lines this section held are in git (the commits above and
the plan as it stood at `e9586c09`) — read them there to recover *why*, never to
decide what is open.

## 5l. The paths STANDARD — one address, three verbs *(user ruling, 2026-09-08)*

**§ 5k's RULE was right and is now enforced. § 5k's METHOD has to stop.** The
rule — *for every name it composes, the framework owns the search* — closed 24
handcrafted searches and is guarded (§ 5k.4c). But it was applied by **adding a
door for whatever question a call site happened to ask**, and the bill came due
the same day. Measured after M1–M8:

| the question | how many APIs answer it |
|---|---|
| which files are this rung's? | **4** — `runfiles.find(stage=)`, `find_by_role`, `Shape.stage_glob` (returns a glob *string*), `parse.dirs.job._enumerate_files(match)` (another glob) |
| what is this file called? | **3** — `compose`, `stem` + caller concatenates, `tail` + caller prepends |
| where does this thing live? | **16, across 3 modules** — `paths` 3, `jobset.materialize` 11, `sidecars.molstruct` 2 |
| what stage is this file's? | **2** — `runfiles.parse` and `identity.parse_stage_token`, **measured to disagree** (§ 5l.4) |

~40 public functions, and **five were added on 2026-09-08 alone, each to fit one
call site**: `role_matches`, `find_by_role`'s underscore refusal, `sidecars_in`,
`is_sidecar`, and the `compose`-vs-`tail` split. That is the API being bent to
arbitrary use — by the framework whose reason to exist is to stop that.

> **The ruling (user, 2026-09-08).** *"The API should be rigid standard, but
> accommodating for a certain flexibility — but it cannot allow for any random
> cases. All real needs, if they need to differentiate and build in a
> hierarchical system, have to follow the hierarchy of that system to start
> with. They may have their own labels or design systems, but that's it."* And:
> *"don't twist the API design such that you can fit arbitrary kind of need,
> but rather a standardized API that has reasonable extensibility."*
>
> So the direction of fit reverses. **A use that does not fit the standard is a
> use that changes** — § 5l.4's third column is the deliverable, not an
> apology.

### 5l.1 The address — § 2.6's hierarchy, not a new one

**§ 2.6 is already the authority and already numbers the tree.** This framework
serves it and never redefines it. Five levels:

> ① project → ② topic → ③ calculation → ④ stage → ⑤ attempt

and § 2.6 is explicit that the benchmark rows get **no circled number** —
*"they are nested containers inside a stage (④), not levels of the tree."*

**Where the standard starts, and where it stops.** `projects.py` owns ① and ②;
this standard owns ③–⑤. So an address is **relative to a calculation
directory**, and the caller supplies that root. Two owners for ①② is the
duplication this section exists to prevent, and putting them in the address
would create it.

```
Ref(label, stage, bench, attempt, role, counters, fields)      # + a root
```

- **`label` is the FILENAME's label, and it is NOT the folder name.** § 2.6
  row ③: the calculation folder is *"whatever the user types"* and *"the folder
  is not derived"* — what makes it a calculation is the `task.json` inside it
  (`run-identity.md` § 3.0). The two are free to differ, and any door that
  assumes otherwise is wrong on a folder someone renamed.
- **`bench` is a qualifier on ④, not a sixth level.** A trial is a stage's
  sub-container; inside it an attempt is ⑤ exactly as anywhere else. This is the
  answer § 2.6 gives, and it is why `materialize`'s eleven layout functions can
  collapse rather than fight the model.
- **`counters`** are the declared numeric qualifiers (`runfiles.QUALIFIERS`).
- **`fields`** are the row's own keys — *the bounded flexibility*. This is where
  a need brings its own label without inventing a level.

**`counters` and `fields` stay separate, and that is § 6.3's rule not a
preference.** A hyphen announces a COUNTER and an underscore a NAME, which is
what lets a name be read back at all; collapsing the two into one typed
dictionary loses the separator rule and the declared ordering that gives one
name one spelling. Whoever is tempted to unify them should read § 6.3 first.

**An absent coordinate is a STATEMENT, never a default.** `stage=None` means
*this file crosses rungs*; `attempt=None` means *not attempt-scoped*. That is
already `runfiles`' rule for `stage` (it refuses `""` rather than reading it as
None) and it becomes the rule on every axis — which is `process/code-audit.md`
D1 applied to the address instead of re-learned per parameter.

### 5l.2 Three verbs, and nothing else public

| verb | signature | answers |
|---|---|---|
| **compose** | `compose(ref) -> Path` | full coordinates → the one path, directory and filename together |
| **find** | `find(root, **partial) -> [(Path, Ref)]` | partial coordinates → every match, ordered |
| **parse** | `parse(path, …) -> Ref \| None` | a path → its coordinates, or None when it is not ours |

**Extensibility is a catalogue row and its fields — never a new function.** If a
question seems to need a fourth verb, the question is malformed; that is the
test § 5l.4 applies.

`compose` returning a whole path (not a filename) is deliberate: a caller that
wants a file wants a path, and the three-call detour (`stage_dir` +
`attempt_dir` + `compose`) is where a caller starts joining strings.

**The browser is not a fourth verb, and it does not hold an address.** 48k lines
of JS cannot import this module, so *"one address"* would be false the moment a
page needed a path. The rule is the one M5 already applied: **a surface ASKS,
never composes** — the server answers with the path or the list, and the page
renders it. Four browser re-implementations were deleted on that basis
(`stage_token`, `/^run-\d+$/`, the flat `_<token>-run` form, and a shape
inferred from disk); a fifth appearing is a § 5l violation, not a JS problem to
solve in JS.

### 5l.3 What the standard DELETES that M7/M8 added

| added 2026-09-08 | why the standard removes it |
|---|---|
| `runfiles.role_matches`, and patterned roles | `.runwrap-*.log` is **not** a role with a wildcard in it. It is role `.runwrap.log` carrying a **`stamp` field**. A glob inside a role is a coordinate that escaped into the vocabulary, and `role_matches` is the machinery built to chase it. |
| `find_by_role`'s underscore refusal | `find` takes partial coordinates, and the catalogue disambiguates an underscore role — which is already how `parse` does it. The refusal was an internal limitation published as API. |
| `sidecars_in`, `is_sidecar`, `sidecar_path_for` | `.molstruct.json` is a **role** whose address has no stage and no attempt. Three functions in a second module become three verbs on one address. |
| the `compose`-vs-`tail` rule | invented to accommodate `attempt_concluded` being handed a foreign stem (`my.relaxation`). **A foreign file is not in the address space.** The caller should be told that, not served — and `tail` should not be the door that makes serving it possible. |
| `_stage_state(label, stage, out_glob)` | two ways to say one rung, in one signature. `out_glob` goes when `run_status` takes coordinates. |

**This is not a retraction of M7/M8.** Both fixed real, measured defects (§ 5k.4c,
§ 5k.4d) and both are green. What is being retracted is the *method* of adding a
door per question, and the five above are its residue.

### 5l.4 The inventory — every case, with a verdict

**Three axes, three instruments** — all three shipped as of N1 (2026-09-08),
all three in `tools/classify_path_finders.py`, all three in `--check`.

| axis | instrument | state 2026-09-08 |
|---|---|---|
| searches for a name we compose | `tools/classify_path_finders.py` | 51 sites, **0 handcrafted**, guarded by `tests/test_path_framework.py` |
| counter-keyed names built by hand | the same tool, `compositions()` | **0**, guarded |
| hierarchy segments spelled inline | the same tool, `segments()` — **N1, shipped** | **2 undeclared segments over 10 sites**: `pseudos` ×5 (`prep.py` 486, 1225, 1226, 1739, 1740) · `launch` ×5 (`submit.py` 1201, 1341, 1514, 1603, 1652). Guarded as a SET, not a count |

**N1 corrected this row's own numbers, which is why it exists.** The hand-run
version said *"`pseudos` ×4"* and missed `prep.py:1739` — the `f"pseudos/{p.name}"`
form, a path built as a string rather than by `Path` division. The real figures
are **5 and 5**. § 5a, demonstrated on the section that states § 5a.

**What is guarded is the SET of undeclared segments, not the site count.** The
ten sites go to zero in N5; what must not happen before then is a *third*
invented segment joining them, because that is a level of the tree created at a
call site and only § 2.6 may add one. `GUARDED_UNDECLARED` is asserted by
equality, so closing one is a deliberate edit and not a quiet pass.

**Why the vocabulary is a calculation's containers and nothing more.** A broad
net — every `X / "plain-name"` — finds **67 distinct segments across 128 sites at
11% precision**: units (`Ha/`, `eV/`, `n/a`), a MIME type (`application/json`),
conda's own trees (`bin`, `conda-meta`, `opt`, `envs`), git's (`refs/`), the docs
tree. That is the nag list this section warns about. **Levels ① and ② are
`projects.py`'s and are already closed**: `CANONICAL_TOPICS` is declared *and*
`validate_topic` refuses anything outside it, so a topic cannot be invented — and
including topics measured 2 false positives out of 2 (`f"transport/v{sv}"` is a
schema string; `client.get("user/orgs")` is a GitHub endpoint, both flagged
because `transport` and `user` are also topic names).

**What the throwaway version got wrong, now fixed in the shipped pass:** it
matched a segment name anywhere in a string and so flagged the Flask routes
`/api/structure/analyze`, `/api/bench/summary` and `/api/task-setup/attempts`.
The rule is the FIRST path component — a route starts with `/`, so its first
component is empty. A test holds that case.

**Two of the four containers are already declared**, which is what makes the
other two findings: `bench` has `materialize.bench_container` (and `_cli.py`'s
hardcoded `base/"bench"` was closed 2026-08-13), and the history directory is
`checkpoint.ARCHIVE_DIR`. A test asserts a declared segment names a door that
really exists, so a rename cannot leave the claim standing.

Then the uses, each judged against § 5l.1–5l.2 rather than accommodated:

| case | verdict | why |
|---|---|---|
| `.runwrap-<stamp>.log` | **regulate — a field** | role `.runwrap.log` + `stamp`; deletes `role_matches` |
| `.molstruct.json` | **regulate — a catalogue row** | a role whose address has no stage and no attempt |
| `pseudos/` | **regulate — a coordinate** | § 2.6's shared package. A door exists (`prep._pseudo_dir`), is private, and is bypassed by one of its own module's sites |
| `Shape.stage_glob`, `_enumerate_files(match)` | **the use changes** | a glob-shaped API becomes a partial-address `find` |
| `identity.parse_stage_token` | **delete** | verb 3 on a coordinate the address already carries — and it disagrees with `parse` on `<label>_<stage>_geom_optim.xyz` (says the stage is `01_coarse_geom_optim`) and on `<label>_<stage>.runwrap-*.log` (says no stage). `parse` is right in both |
| `attempt_concluded(foreign stem)` | **the use changes** | reject as not-ours; do not loosen the grammar to serve it |
| `submit.py`'s `launch/<name>.{run.sh,sbatch}` | **the use changes** (P-4) | `launch/` is an ad-hoc segment and `<name>` is a *group*, not a calculation. Either a group becomes a real qualifier on ④, or the group launcher moves to where a launcher belongs |
| `web/blueprints/files.py`'s sidecar copy | **the use changes** (P-5) | the ONLY constant collision in the whole tree, sitting behind a decision layering does not require |
| `workspace_storage`'s `.wc.json` | **out of scope, and stated** | not a run file. Its own grammar, self-consistent, module-local — § 4.5 satisfied already |
| SIESTA `.XV` and the warm suffixes | **out of scope, and stated** | § 4.2 — molbuilder does not compose them, so no door may claim them |
| `_geom_optim.xyz` inside emitted script TEXT | **deferred, recorded** | `makov_payne.py` ships beside a job; needs `runfiles` in `runwrap.MONITOR_COMPANIONS` first |

### 5l.4a The three LIVE defects, measured — N5's actual content

§ 5l.4's verdict column says *the use changes*; these three are the ones a
person is already feeling. Each was measured by running it, not by reading it,
and each measurement is the test N5 must ship.

**① Every staged run loses its frozen atoms.** `parse/engines/_sidecar.read_frozen_atoms`
rebuilds the sidecar's name by stripping the rung off the artifact's stem, which
needs the label. All three callers omit it — `engines/molwatch.py:590`,
`engines/pyscf.py:626`, `engines/siesta.py:1877` — so *"Hide frozen atoms"* and
`runtime_info["frozen_atoms"]` are empty for every laddered calculation:

| artifact | no label | with label |
|---|---|---|
| `bdt.out` | `[5, 6, 7]` | `[5, 6, 7]` |
| `bdt_01_coarse.out` | `[]` | `[5, 6, 7]` |
| `bdt_01_coarse.molwatch.log` | `[]` | `[5, 6, 7]` |
| `bdt_01_coarse.pyscf.log` | `[]` | `[5, 6, 7]` |

**And the obvious fix is the wrong one.** `model/parse.md` § 5.3 says a
`FileParser` finding its own companion **stays path-only** — *"folding those in
would make every engine parser depend on the directory composer, inverting § 5's
own rule."* So threading a label down is against the spec. The fix that satisfies
both documents: **ask by ROLE, not by reconstructing a stem.** A folder is ONE
calculation (`project-layout.md` § 1.4), so the sidecar is found by its role, no
label required, still path-only. Under § 5l that is `find(root, role=…)` with no
`label` coordinate — the standard answering it directly.

**② The stage token has two readers, and they disagree.** `runfiles.py`'s header
states the contract — *"every reader that looks one up. Nothing composes or
splits a run-file name inline"* — and `identity.parse_stage_token` splits one:

| filename | `identity.parse_stage_token` | `runfiles.parse` |
|---|---|---|
| `bdt_01_coarse_geom_optim.xyz` | stage `01_coarse_geom_optim` | `01_coarse` + role `_geom_optim.xyz` |
| `bdt_01_coarse.runwrap-…log` | no stage | `01_coarse` |

`runfiles` is right both times: `_geom_optim.xyz` is a *declared* underscore role
and `parse` documents this exact case. **Latent, not live** — the three callers
(`parse/dirs/job.py:80`, `materialize.py:384`, `:423`) only feed it deck and
`.out` names, where the two agree. The duplication is the defect. N5 deletes it.

**③ A phantom rung for an unstaged calculation.** `parse_stage_token("run_01_setup.out")`
returns `(1, 'setup')` with no label and `None` with it. `parse/dirs/job.py:80`
(`_detect_stage`) omits it, and it feeds the sort at `job.py:237` that picks
`active = sorted_outs[-1]` — *which file speaks for the directory* in
`run_status`. § 5.3's own test says that question **does** need the whole
directory, so the label is legitimately in hand and threading it breaks no
boundary. Fixed by ② rather than separately.

**The two blocked on a decision, kept here so they are not re-derived:**

- **The group launcher.** `submit.py` builds `<container>/launch/<name>.run.sh`
  and `.sbatch` at ~8 sites (1201, 1273, 1341, 1392, 1514, 1603, 1609, 1652,
  1657); `launch_dir = container / "launch"` appears twice. Both roles are in
  `WRITTEN`, but `<name>` is a *group*, not a calculation, so `compose` is not
  the door. **Decision: does a group become a real qualifier on ④, or does the
  group launcher move?** Until that is answered the sites stay.
- **`web/blueprints/files.py`.** The ONLY duplicated constant in the whole tree
  (`_SIDECAR_SUFFIX` at :233 against `sidecars.molstruct.SUFFIX`), plus a
  re-implementation of the pairing rule at :236. It sits behind a written reason
  — *"the blueprint doesn't import from sidecars/molstruct directly to keep its
  dependency graph narrow"* — which layering does not require (`sidecars` is L2,
  `web` L3). **Decision: keep the narrow-graph choice and add the parity test
  `process/code-audit.md` D4 then demands, or import the door and delete the
  copy.**

### 5l.5 The dependency order — the workflow the API must have

1. **catalogue** — the roles and their fields. Data; no dependencies.
2. **address** — the coordinate record; validates a role against the catalogue.
3. **name grammar** — the filename half of compose/parse.
4. **layout** — the directory half: a function of (shape, address).
5. **the three verbs** — compose / find / parse over whole paths.
6. **everyone else** — calls only the verbs.

Each layer may import only the ones above it. **L1 and stdlib-only throughout**,
so the monitor still ships beside a job (`runwrap.MONITOR_COMPANIONS`) and runs
under the job's python.

### 5l.6 Migration — each step separately green

| step | what | done when |
|---|---|---|
| ~~**N1**~~ | **DONE 2026-09-08.** `segments()` in the same tool, wired into `--check` and the summary, with the override discipline the other two passes use. Corrected the row above: 10 sites, not 9. Five mutations, five killed — including the `GUARDED_UNDECLARED = ()` kill switch and the first-component rule (dropping it lets Flask routes back in) | shipped |
| **N2** | **the catalogue grows `fields`, and the NAME GRAMMAR honours them** — layers 1 and 3 of § 5l.5, which is why this can precede the address. `.runwrap-*.log` → `.runwrap.log` + `stamp`; `role_matches` and patterned roles deleted | the wrapper log is composed and parsed through a field, and the patterned-role machinery is gone |
| **N3** | `Ref` + the three verbs, **beside** the existing API. No caller moves | the verbs answer every question § 5l.4 lists, with tests over the address, not over call sites |
| **N4** | the 16 layout functions and the 13 name functions delegate to the verbs. Nothing deleted | one implementation behind every existing name; the suite unchanged |
| **N5** | **the uses that change**: `stage_glob` / `_enumerate_files` → partial-address find; `parse_stage_token` deleted; `attempt_concluded` takes an address; `pseudos/` becomes a coordinate | each is its own commit with the behaviour asserted before and after |
| **N6** | P-4 (the group launcher) and P-5 (`files.py`), both **blocked on a decision**, not on code | the decision is recorded here first |
| **N7** | delete the superseded surface, **one module per commit** — `sidecars.molstruct`, then `identity`, then `materialize`, then `paths`/`runfiles`. Forty functions cannot be retired in one separately-green step | the guard asserts the framework's public surface IS the three verbs plus the catalogue |

> **N2 was written after N3 in the first draft of this section, and that was an
> ordering bug** — it deleted `role_matches` and introduced `fields` before
> anything existed to carry a field. Caught on review the same day. A field
> shows up *in the filename*, so it belongs to the name grammar (layer 3) and
> genuinely can precede the address (layer 2's consumer); the corrected N2 says
> so explicitly rather than leaving the next reader to rediscover it.

**Do N1 and the § 5l.4 re-derivation before N2.** § 5a's rule applies to this
section as much as any other: every count above was measured on 2026-09-08 and
will be wrong by the time anyone acts on it.

### 5l.7 What must not change

`runfiles`/`paths` stay stdlib-only; **the shape is DECLARED, never inferred
from disk**; `identity.stage_token` stays the one speller of `<NN>_<name>`; and
**§ 2.6 remains the authority on the tree** — this framework serves that
hierarchy and never invents a level of its own. A migration step that needs one
of these to move has found a design problem, not a step to push through.

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

---

## 7. The document survey and the four indexes — PLANNED, not started

*(User, 2026-09-07: "the documents are very fragmented and a lot of key
information cannot be easily found. we also should have an index that
summarizes key design aspects, such as api, module, data structure etc. we
also should have a reference list that gives the foundation of all the
constants, scientific validation foundation and paper citations summarized in
one place.")*

### 7.1 What is actually wrong — measured 2026-09-07, before proposing

**Not coverage.** The first guess was that documentation had fallen behind the
code, and it has not. Every one of the **77** `/api/*` routes the blueprints
serve is named in `web-api.md` — re-derived by expanding the doc's own braced
forms (`/api/checkpoint/{state,list,config}`), which a naive matcher scores as
26 missing. Do not plan a coverage sweep; there is nothing to catch up on.

**Findability.** 75 live documents, **50,609 lines**, nine directories. The
information is there and a person cannot reach it.

**The existing index has the problem it is meant to solve.** `README.md`
§ Index is one row per DOCUMENT, and several rows run past 400 words. To find
out which document owns a fact you read essays about documents. It is a good
map of the tree and a poor answer to "where is X".

**Two documents contradicted themselves in one day**, both found by accident
while doing something else — `science/pseudopotentials.md` § 3 listed a check
as not-done that its own § 2a.1 documents as done, and `process/testing.md`
§ 6 quoted a number three sentences after telling the reader never to quote a
number from a document. Neither is exotic. Nothing systematic would have
found them, which is the survey's real justification.

### 7.2 The four indexes — organized by THING, not by document

Each is a lookup table, each row pointing at the document that owns the
detail. **None of them may become a second home**: a row says where a fact
lives, never what the fact is. (`README.md` R-W1, and the reason `docs/`
survived its 2026-06-02 over-compression.)

| index | one row per | the row answers | generated or written |
|---|---|---|---|
| **API** | route | what it answers · who calls it · which doc owns it | **generated** — the route list is derivable from the blueprints, and a generated index cannot drift. The 2026-09-07 W8 finding is the argument: `web-api.md` claimed the Documents tab read `/api/docs/list`, which it has not since the commit after the one that added it |
| **Module** | module | its role · its layer (L1/L2/L3) · its doc | **generated** from the layering the suite already classifies (`test_layering.py` walks every `molbuilder/*.py` and asserts every name is classified — the data exists) |
| **Data structure** | persisted file / schema | its shape · its version · its one reader and one writer | **written** — the doors are a design fact, not derivable |
| **Reference** | constant · citation | its value · its source · **why this value** | **written** — see 7.3 |

### 7.3 The reference index, which is the one with real content in it

Constants have one home already (`molbuilder/constants.py`, with a lint and a
documented allowlist). What has no home is the **reasoning**, and it is
genuinely scientific. The worked example, found while measuring this:
`trajectory_log/emitter.py` retypes `HARTREE_BOHR_TO_EV_ANG = 51.42208619`
rather than deriving it from CODATA-2018 (which gives 51.422067476, ~4 ppm
apart) — deliberately, so emitted forces line up with what a person reads in
ASE, VASP and QE logs. That is a real convention choice with a real
justification, and it lives in a code comment nobody will find.

Citations are split the same way: `science/references.bib` holds **21**
entries, **5** live documents reference it, and at least **12** cite
literature in prose instead. Two keys the roadmap called for — Reed 2006 and
Stokbro 2003 — are in neither (row **E14**).

So the reference index carries three things per row: the value, where it is
defined, and the sentence saying why that value and not the neighbouring one.

### 7.4 Order, and the one risk

1. **Survey first, and record findings without fixing them.** Every live doc
   read against the code, one pass, findings in a list. Fixing while reading
   is how a survey becomes a month.
2. **The two generated indexes**, which are cheap and cannot go stale.
3. **The two written indexes**, from the survey's findings.
4. **Only then** decide what merges. Fragmentation may be the symptom; the
   fix might be four indexes and no merges at all.

**THE RISK IS THIS PLAN'S OWN HISTORY.** The 2026-09-01 consolidation merged
nine documents and dropped § 5f entirely; a re-check on 2026-09-07 found
**seven more** items lost (E12–E14, N5, W16–W18), from source documents
credited with "nothing open" that were never checked row by row. A survey
that reorganizes without a per-item carry-over check will do it again. The
rule for step 4: **nothing moves until the thing it says is written where it
is going**, which is the archive's substance-first rule, and it is the reason
§ 5i could not be archived until `process/testing.md` § 2a existed.

---

## 8. One nature, fourteen instances — the reading doors have no guard

*(Consolidated 2026-09-07, after the selection/MolView review. Written because
every defect that review found is the SAME defect, and fixing them one at a
time is what produced fourteen of them.)*

### 8.1 The pattern

**A door exists. Someone needs it to behave slightly differently for a local
reason. They write a second implementation instead of widening the first. The
two then drift, and the copy is the one that is wrong.**

Every instance below was found by accident, chasing something else. None was
found by a guard, because no guard covers this class.

| # | the door | the second implementation | how it drifted | state |
|---|---|---|---|---|
| 1 | reading the `.xyz`+`.molstruct.json` pair (`StructureCodec`) | `molbuilder.load()` | dropped the whole sidecar; `jobset init` therefore wrote descriptions with no regions, frozen atoms or cell | **FIXED 2026-09-07** — deleted |
| 2 | ” | `selection.py::_load_structure` | applies `regions` only — drops identity columns, cell, annotations, `info`; three rule kinds then return nothing | dead code, deletion pending |
| 3 | ” | `transport/_cli.py::_load_device` | applies the whole sidecar — correct, but a third copy of the walk | open |
| 4 | ” | `compose.py::labeled_citation_structure` | globs `*.molstruct.json` instead of asking the pairing rule | open |
| 5 | the pairing rule (`sidecar_path_for`) | `parse/engines/_sidecar.py` | own suffix-strip table; **disagrees** on `x_optim.xyz` and `z.molwatch.log` — deliberate, but a third derivation | open |
| 6 | ” | `files.py::_paired_sidecar_path` | agrees today; a fourth copy is a coin-flip on the day one changes | open |
| 7 | per-atom rows (`_shared.atoms_list`) | `/api/selection/atoms` | skips the cell resolver, so it answers with a box that was never resolved | dead, deletion pending |
| 8 | "is isolate in effect" (`isolate && selection.length > 0`) | `mount.js:228` reads the raw switch | **live UI defect** — with nothing selected, the 3-D window stops accepting clicks, silently | open, fix agreed |
| 9 | ” | `stores.js:145` auto-off | a second mechanism for the same rule; it is also what makes the button un-press itself | open, fix agreed |
| 10 | the `forceScale` switch | the trajectory template's slider | DOM value overwrites the restored one on every load | open |
| 11 | the filter row kinds | hard-coded in `stores.js:243`, `stores.js:379`, `ui.js:2161` | three literal lists, no shared constant | open |
| 12 | what an edit invalidates | one `structure_modified` flag for geometry, cell AND labels | unusable by its only reader | **FIXED 2026-09-07** — split in two |
| 13 | `molview.md` § 1.1 vs §§ 6.6 / 9.5 / 11.6 | the same facts stated twice | five stale claims; one is a defect fixed four days earlier **in the same file**, in one copy only | open |
| 14 | `structure-annotations.md` § 6 | names four JS modules that were never built | describes a layer that does not exist | open |

### 8.2 Why nothing caught them, and what the codebase already does about it

**This project already knows the answer.** Nine guards exist, each written
after one of these was found the hard way:

`test_layering.py` · `test_one_home_for_a_constant.py` (the Bohr radius, once
written out eight times) · `test_one_naming_authority.py` · `test_config_dir_has_one_home.py`
· `test_css_no_duplicate_selectors.py` · `test_css_module_boundary.py` ·
`test_vibrationview_module_boundary.py` · `test_no_duplicated_ui_components.py`
· `test_no_test_is_shadowed.py`

They cover imports, constants, a naming rule, a config path, CSS, two module
seals, UI components and test names. **Not one covers a reading or writing
door** — which is where rows 1-7 live.

So this is not a new mechanism to invent. It is **the missing member of an
existing family.**

### 8.3 The framework fix

**A guard that enumerates who may read the pair, and fails when a second
implementation appears.**

Shape, following `test_one_home_for_a_constant.py` exactly — an owner, an
allowlist with a REASON per entry, and a check that each allowance still
describes the code:

* **owner** — `StructureCodec`.
* **the lint** — AST-walk `molbuilder/**`; any function that both reads a
  geometry path AND looks for a sidecar beside it, and is not the codec, is a
  finding.
* **allowlist** — entries carry the reason they are exempt, and the guard
  fails if an allowed site stops doing the thing it was allowed for (the
  existing constant guard does exactly this, so an allowance cannot outlive
  its reason).
* **the same for the pairing rule** — `sidecar_path_for` is the owner; rows
  5 and 6 are its findings.

**Why a hand-written list is not enough, and this is the load() lesson:** a
list of doors written by a person would have carried `molbuilder.load()` on it
looking entirely reasonable — right name, in `__all__`, docstring accurate for
what it did. Only *deriving* the readers from the code and comparing against
one declared owner surfaces a rival. **The index must be generated** (§ 7.2),
and the guard must read the code, not a list of names.

### 8.4 What is NOT an instance, and must be fixed on its own

Two findings are ordinary defects, not door duplication, and a guard will
never catch them:

* **Row 8, the isolate click gate** — a live UI defect. Nothing selected,
  press "Show selected only", and the 3-D window silently stops accepting
  clicks while the drawing does not change. Fix agreed with the user: one
  accessor in the store, read by all three, and the auto-off deleted (a
  preference stays where you put it; "nothing selected" means nothing to
  hide).
* **`lastApply` never cleared by a selection change** (`stores.js:136-147`) —
  the panel can read *"No atoms matched this filter"* beside *"N of N
  selected"*.

### 8.5 Order

1. The guard for the pair readers **first**, with rows 2-6 as its initial
   findings — so the deletions that follow are checked by something rather
   than by me.
2. Delete row 2 and row 7 (dead), rehome the ~10 rule tests onto the live
   door, and re-home the atom-row wire-shape claim, which today is pinned
   only through the dead route.
3. Rows 3-6 by the guard's allowlist: each either goes through the door or
   earns a written reason.
4. Rows 8 and 9 as their own fix.
5. Rows 13-14 fold into § 7's survey.

