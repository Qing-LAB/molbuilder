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
`archive/2026-09-01-*`. They were 2044 lines between them.

**R3a — a plan section is kept current while its work runs** *(user,
2026-09-16)*:

> **A programme with an implementation section updates that section when each
> step lands, and records a problem found mid-step before working around it.**

Not afterwards, and not in a commit message alone. A step that changed shape, a
step that turned out to depend on something unlisted, a defect found while doing
it: each goes in, with what was measured. **A plan written once describes an
intention; a plan kept current describes the work.** The first section under
this rule is § 5p (transport).

**And R3 now means what it says: there is ONE table** *(2026-09-10)*. Until
that day the open items sat in six tables across §§ 2, 3, 4, 4a, 5, 5b, 5f, 5l
and 5m, which is how three different rows came to be numbered `P3`, how `W8`
could cite line numbers for three endpoints that are not registered, and how
`W9` could carry one sentence naming no behaviour for weeks. **§ 2 is the
list.** A design section keeps the *why* and sends its rows there; a row that
is done, withdrawn or measured untrue leaves for
[`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md)
the same day, with what it turned out to be.

---

## 1. The 2026-09-01 fact-check — ARCHIVED

*Nine plan documents read against the code; three headers were flatly false. The detail is history now and lives in [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md). The one live fact it found — bench § 2.2's `parse_util_bound` reads the VERDICT and not the numbers, so the item is open — is a row in § 2.*

---

## 2. OPEN — the one list

**Every open item, in one table** *(consolidated 2026-09-10 at the user's instruction: "consolidate plan and to do, archive finished things and untrue things, clean it up so we have one list")*.  §§ 3, 4, 4a and 5 held four separate tables of the same kind of thing and are now pointers — the numbers stay so links from other documents still land.

**Rows that were done, withdrawn, or measured untrue are NOT here.** They are in [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md), with what each one turned out to be. Read § 5a before acting on any row below: a row is evidence of when it was written.

| # | area | item | from | state |
|---|---|---|---|---|
| **E1** | engine / science | **Benchmark iteration count, settable per calculation.** No field exists on `task.json` or `Resources` — confirmed 2026-09-07. **Bigger than the row says:** the archived design was a one-point `bench` entry overriding the pin, and that path is now explicitly closed — `_cli.py:1961` `pins = {**declared_pins, **_MEASUREMENT_PINS}`, with the comment that measurement pins *"must win over any declaration — one-point declarations and value-axis coordinates alike."* So this needs a written precedence rule reversed, not a field added | `bench-and-junction` § 2.1 | not started |
| **E7** | engine / science | **D7's cluster half** — the prep→submit→watch loop for a **run** through SLURM on Sol. **2026-09-10, the user: "we have tested it on Sol" — and the row must not contradict that.** What is actually on disk here: Sol slurm records exist (`optimization/sol/AuBDTAu-slabcorrected/01_coarse/bench/launch/slurm.62380919.out`, real ids) and every `kind: "run"` ledger entry in `projects/` is `workstation`/`direct`. That is the absence of a LOCAL record, which the earlier wording ("No `kind: run` submission exists on any Sol tree") presented as the absence of the WORK. A tree on Sol is not in this repo. **This row needs the user to say whether it is closed, not another file scan** | roadmap § 1 | needs the user |
| **E10** | engine / science | **NEEDS A RE-SCOPE, not implementation — re-derived 2026-09-07.** The detection is **built**: `parse/engines/_run_ending.py` is the one marker table (`abnormal_termination` → `stopped`, OOM markers, `SCF_NOT_CONV`), `scan_ending()` returns `(run_state, scf_converged, error_message)`, and `summarize.py:217` + `cli.py:2771` consume it — the zero-exit case being the whole point. What is NOT built is the monitor half, and **deliberately**: `monitor.py:138` says *"NO COMPLETION MARKERS HERE"* and `job-contracts.md:214` now rules the monitor follows the launcher's PID *"rather than guessing from output markers"* — which contradicts this row's own "belongs in `mb_monitor.py`". The real gap is narrower: **the monitor's finish report carries no convergence fact.** Decide whether it should before writing anything | roadmap § 4 | open (re-scope) |
| **T1** | engine / science | **A LIVE POLL THAT MUST REBUILD DOES NOT UPDATE THE MOVIE — found 2026-09-07.** The append path is fine; the full-rebuild path is not. Reproduced: move the frame at `oldLen - 1` (the one `_frameEqualAt` reads) and grow the feed 4 → 6, so `canAppend` refuses and `applyNewData` takes the `else` branch. The status line then says *"Loaded 6 … frames"* and the frame bar still holds **4** — the feed's count and the movie's disagreeing, which `core.js` itself names as bug **#35**. The trigger is ordinary: a frame that was still being written when the last poll caught it, and has since settled. **Two things hide it, each worth its own look:** `setStatus` is a no-op on `/results` (`if (!document.getElementById("status")) return;`), so `rebuildModel`'s *"Viewer failed to load the run"* reports into nothing on the page the inspector lives on; and *"Loaded N frames"* is written by `applyNewData` from the FEED's count while `rebuildModel` runs unawaited beside it, so the tab can claim frames it is not showing. Recipe at the foot of `tests/test_inspector_registry_e2e.py`; it blocks the last three source pins in `test_structure_info_bridge.py`, which is how it surfaced | found 2026-09-07 | open |
| **E12** | engine / science | ~~**The Methods paragraph is still a placeholder.**~~ **CLOSED 2026-09-18 — the subject was deleted and the item outlived it.** Every citation this row rested on is gone: `transport/transiesta.py:1015` (that file is **759 lines**), `transport/engine_base.py:135` (**the file does not exist**), and `methods_fragment` itself, deleted 2026-09-17 with `TransiestaEngine` — only a tombstone at `transiesta.py:24` records it. **There is no placeholder paragraph left to replace**, so the work as written has no subject. *(What is NOT closed: transport has no Methods paragraph at all now. PySCF/spectra keep a live one — `pyscf/vibration_emitters.py::pyscf_methods_fragment` — so if transport should emit one, that is a NEW item written against `spec_for`, not this one.)* Found by a full-text audit of the transport module on 2026-09-18, not by grep: an OPEN row whose every pointer was deleted code is exactly what sends someone to do work that no longer exists. | roadmap § 2 · `transport-design` § 6 | **closed** |
| **E13** | engine / science | **The first real junction walk — BDT–Au, workstation then Sol. DROPPED BY THE CONSOLIDATION.** Named in the roadmap and again in `transport-design` § 7's *"order of proof"* as the run that follows P6. plan.md has the machine-blocked infrastructure (E5/E7) and the browser and deck walks (W12/E11) but no row for the transport composite's first real science run | roadmap · `transport-design` § 7 | open |
| **E14** | engine / science | **Two bibliography keys — Reed 2006 and Stokbro 2003 — are cited nowhere in `docs/science/references.bib`.** Verified absent 2026-09-07. Mechanical, small, and dropped by the consolidation | roadmap § 4 | open |
| **E11** | engine / science | **A fresh live walk of the PySCF / spectra decks.** The 2026-08-28 review exercised them only through the guard suites and says so | audit 08-28 § 5 | open |
| **W1** | front end | **The document tier (step C).** `html, body`, `header`, `button`, `footer`, `textarea` genuinely differ per page; the `*` reset is already deleted. Blocked on a browser pass over all pages | `css-system` § 4C | partly |
| **W2** | front end | **One home per component (step D).** `.card`, `.status`, `header .tagline`. **One value to settle first:** `.card`'s padding is `var(--space-md) 18px 18px` and 18 is off the 4px grid the contract declares — moving it shifts every page by 2px | `css-system` § 4D | not started |
| **W3** | front end | **Per-page token/namespace passes (step E)**, one page per commit: `spectra`, `structure-optimization`, `transport`, `results`, `documents` | `css-system` § 4E | partly |
| **W4** | front end | **Guards 1 and 2 (step F)** — one home including elements; a page sheet contains only its own tier. Guards 3 and 4 landed. **Both remaining gaps are now provable, 2026-09-07:** guard 1 is absent *by an explicit skip* — `test_css_no_duplicate_selectors.py:150` reads `if "." not in norm: continue`, so element-only selectors are exempt by construction; guard 2 has no test at all | `css-system` § 4F | partly |
| **W5** | front end | **The inspectors module's appearance still lives in `results/style.css`.** **Re-derived 2026-09-07 and the number was prose:** 70 was `grep -c inspector`, which counts the file's 200-line comment header and a hierarchy diagram. Comments stripped and classified by who EMITS each class: **22 module-owned rule blocks**, and **6 of those are dead** — `.inspector-section`, `-section-header`, `-section-body`, `-section-hint`, `.source-body-error`, `.structure-error` have **zero emitters anywhere** in the repo and are deletable outright, which this row never said. Three sheets are already repatriated. *"Renders unstyled elsewhere"* is **latent, not reachable**: `registry.js` is script-tagged by `results.html` only, so the css-system doc's premise (it also loads on /molbuilder and /spectra) is stale. Also: `inspectors/bench-summary.css` is missing from the boundary guard's `MODULE_SHEETS`, so that guard treats a module sheet as a page sheet | `css-system` § 7.0 | partly |
| **W6** | front end | **The editor module.** The loader half is confirmed and accurate: `lib/codemirror-load.js` is the one loader, two of three surfaces import it, and `lib/inspectors/markdown.js` still hand-rolls its own pair — *definitions* at `markdown.js:31` and `:38` (this row cited only the call site). **The sheet number was wrong twice over, re-derived 2026-09-07: 21 rule blocks / 60 declarations, not 30 and not 40.** The original 25+4+1 was never reproducible as a block count either — `projects-sidebar.css` has held 16 CodeMirror blocks at every commit back to 2026-08-28. The caps (1500-line selection, 1 MB view-only) are on `preview.js` alone, confirmed | `editor-module` | partly |
| **W10** | front end | **Results transmission inspector** — the record exists, the reader does not | `structure-info` § 3 · roadmap § 2 | open |
| **W13** | front end | **Raw px/rem literals — re-derived a THIRD time, 2026-09-07, and the definition finally holds still.** 160 / 740 reproduce exactly, but only because the regex reads raw file text *including comments*. Counting literals **in declarations**: **133** across the eight page sheets, **650** in `lib/`. Two things the row hides: `lib/tokens.css`'s 44 literals ARE the scale definitions — the token layer, not violations — and `lib/molview/molview.css` alone is **252**, 39% of the whole `lib/` figure. So "lib/ carries 740" is really "MolView carries 252, and the rest of lib carries ~400". 777 → 384 → 160/740 → 133/650 are four scopes, not four measurements | roadmap § 7.4c | partly |
| **W15** | front end | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. Plus routing the CLI through the shared codec and exercising the last annotation-channel kind. Confirmed 2026-09-07: `lib/molview/` has **no `_seal.js`** where `spectrumchart/` and `vibrationview/` both do, and seven inspector scripts on `results.html` are still classic `<script defer>` against two on `type="module"` | roadmap § 3 | partly |
| **W18** | front end | **"Modify functions (Molbuilder tab)" — item ZERO of `structure-info-plan` § 5.6's own priority order, annotated *"user calls it higher priority; not yet described."*** It never reached plan.md in any form, and is still described nowhere. Blocks nothing technically; it needs a description before it can be planned | `structure-info` § 5.6 | needs describing |
| **W19** | front end | **The Modify slab panel shows several findings as one sentence at one tone.** `/api/modify/lattice-from-run` answers with `notes` — the same `{severity, message}` rows every other door sends — and `modify/slab-panel.js:389` joins their messages with `·` into a single toast. Three findings become one line, and the panel has no list to put rows in. **The severity half is fixed** (2026-09-11: it folded any-warn→warn and drew an *error* in the info tone; it now takes the worst of the three). What is left is presentation, and it is a UI change on the Modify tab rather than a refactor: give the panel a findings list and render through `lib/validation-findings.js` like every other surface (`science/validation.md` § 4.1 R2a) | found 2026-09-11 | **needs your call** |
| **E15** | engine / science | **`--pipeline-log` is not wired for the transport arm.** `jobset/prep.py:1295` prints *"--log is not wired for the transport arm yet; prep proceeds without it"* — it says so rather than eating the flag, which is right, but a junction prep cannot produce the one file that answers *"how did this value get into this deck?"*, and the junction is where a ladder's rungs disagree | found 2026-09-11 | open |
| **W20** | front end | **A prep from the browser can never produce a pipeline log.** `pipeline_log` is off by default and only `jobset prep --pipeline-log` sets it (`_cli.py:2188`); the web door builds its kwargs at `build.py:1610/1634/1653` and never passes it. Measured: 2 `*.pipeline.log` in the tree, both from e2e fixtures, **none** under any real project. Not a defect — the flag is deliberately opt-in (`pipeline_log.py`: *"the log observes the pipeline, it is not a step in it"*) — but if a person prepping from the UI should be able to ask for one, the door needs a way to say so | found 2026-09-11 | **needs your call** |
| **W21** | front end / science | **ONE spectrum view, every mode on it — IR, Raman and the silent ones.** *(designed with the user 2026-09-11, walking a real CO2 Raman+IR run.)*  **What is measured today.**  The run computes both channels into `<job>.spectra.json` — per mode, `raman_activity_a4_amu` AND `ir_intensity_km_mol` beside each other — and the viewer shows only Raman.  It touches `ir_intensity_km_mol` in exactly ONE place, `lib/spectra/core.js:1485`, inside the change-detection fingerprint: it reads IR to notice the data changed and never to draw it.  `results.phase_ir` is in the same fingerprint at `:1488` while the phase-indicator list at `:1354-1357` is Relaxation / Frequencies / Raman / Per-mode ES.  The modes table is `# · ω · Raman (Å⁴/amu) · imag? · ES? · HOMO · LUMO · Gap · ΔGap max` — **no IR column**; the chart's y-title is hardcoded `"Raman activity (Å⁴/amu)"` at `lib/spectrumchart/index.js:119` with the unit repeated at `:186`, though the module is otherwise quantity-agnostic (it takes a generic `m.activity`).  So ticking **Compute IR intensities** produces correct physics a person can only read by opening the JSON: the CO2 run gave 653.45 cm⁻¹ ×2 at **32.85 km/mol**, 1388.81 at **14.74 Å⁴/amu**, 2460.11 at **613.04 km/mol** — mutual exclusion exactly right, and the 613 asymmetric stretch is THE band of the CO2 IR spectrum.  **Three things are wrong and they are one thing:** the split into "a Raman viewer" is false — one Hessian, two property derivatives on one set of eigenvectors, so activity is an ATTRIBUTE OF A MODE, not a separate spectrum.  **The shape agreed:**  ① **a mirror plot** — Raman up on a left axis (Å⁴/amu), IR down on a right axis (km/mol), each axis coloured to its curve.  Mirroring is why: both channels peak at the same frequencies, so one half-plane makes them collide, and IR drawn downward reads the way absorption does.  ② **a rug of ticks along y=0** carrying EVERY mode, coloured by class (Raman-only · IR-only · both · silent).  Position only, never a height — a silent mode given a stick height is a lie in either direction, and the rug is the only place a mode active in NEITHER channel can honestly appear.  ③ **a relative threshold slider** in Results — *"above X% of the strongest peak in its own channel"*, one control for two incommensurate units — with its default declared in the Spectrum calculation tab.  **Two prerequisites, both server-side.**  (a) `spectra.json` carries **no activity classification** (measured fields: `index_1based, frequency_cm1, raman_activity_a4_amu, ir_intensity_km_mol, eigenvector_canonical, eigenvector_display, has_imag, electronic_structure`), and the zeros are NUMERICAL not exact — the CO2 run's silent entries are `4.8e-09`, `1.0e-08`, `7.5e-08`, `3.6e-09` — so *"is this IR-active"* is a decision needing ONE home, stored, never a magic epsilon invented in the viewer.  (b) the per-mode electron-structure probe **selects its modes by Raman brightness**: `pyscf/vibration_emitters.py:1416` ranks `key=lambda m: (-m['raman_activity_a4_amu'], …)` and `:1421` cuts on `> ES_THRESHOLD`, whose config label at `config/pyscf.py:1055` is literally *'Raman-activity threshold'* (Å⁴/amu).  Gap modulation is ∂ε/∂Q; Raman is ∂α/∂Q — different selection rules, and in a centrosymmetric molecule the filter is wrong in a *systematic* direction, keeping the gerade modes and dropping every IR-active one.  **Note the distinct thresholds**: `es_threshold` decides what is COMPUTED (2 SCFs per mode) and stays absolute; the new one decides what is SHOWN and is free.  **And a third curve is legitimate later, not a decoration**: ∂ε/∂Q against frequency is the **spectral density** J(ω) of electron-transfer / transport theory — for a junction project the one that governs IETS and vibrational broadening of transmission — which also settles the selector: rank by \|ΔGap\| once computed, or simply take `all` where 2N SCFs is cheap (8 for CO2) | found 2026-09-11 by a live UI walk | **designed, not started** |
| **W22** | front end / ops | **A Jupyter notebook tab with a LIVE kernel, and a lifecycle that leaves nothing behind.** *(designed with the user 2026-09-11.)*  **The constraint that shapes it:** molbuilder serves through Werkzeug's `ThreadedWSGIServer` and there is no `flask-sock` / socketio / gevent in the tree, so **WSGI has no WebSocket** and Flask cannot proxy a kernel connection — the tab is an iframe to a separately-run Jupyter and the socket goes browser→Jupyter directly.  Two consequences are not optional: TLS on the Jupyter port (the app page is HTTPS, so an `http://` frame is blocked as mixed content) and `frame-ancestors` naming molbuilder's origin (Jupyter refuses framing by default).  **The lifecycle is the hard half, and the kernels are GRANDCHILDREN** — killing the Jupyter server alone leaves `ipykernel` processes holding memory and GPUs — so: `PR_SET_PDEATHSIG` (the only mechanism that survives `kill -9` of the parent, which runs no handler), a private session plus `killpg` so the whole tree goes, and pidfile reconciliation at startup for the machine-crash survivor, each verified the way `serve_daemon` already verifies — alive, ours, actually a Jupyter we started.  **PARENTED TO THE SUPERVISOR** (the user's choice against two alternatives): the supervisor respawns the server child on `RELOAD_EXIT_CODE`, so a Jupyter parented to that child would be killed by every unrelated code reload, destroying notebook state; parented one level up it survives a reload and dies with the daemon.  Nothing runs until asked, and Jupyter's own `cull_idle_timeout` / `shutdown_no_activity_timeout` shrink the idle window rather than molbuilder hand-rolling one.  **And it exposes arbitrary code execution on a network port**, which belongs in `access-control.md` as a decision rather than as an implication of having added a tab | `web/jupyter.md` | **built 2026-09-14** — the tab is **JupyterNB** at `/jupyternb`; the env is self-contained (server + kernel + analysis stack, no other env carries notebook tooling); the exposure rule is `jupyter.md` § 6.  **Two things in the design cell did not survive contact and are left there as the record of what was planned:** TLS on the notebook port is NOT mandatory — the notebook takes the same scheme the app page is served with, and runs plain http when `serve` has none; and the process group does NOT reach the kernels — each gets its own session from `jupyter_client`, so what `killpg` guarantees is that the SERVER dies, and Jupyter collects its own kernels (measured, `jupyter.md` § 3.2) |
| **D2** | doc drift | **Tests with no target, remainder.** The two files with zero test functions were **checked and left** — each is a signpost recording where retired coverage moved, which is a service, not residue. **Re-derived 2026-09-07 with the definition stated:** 417 test files, 6,529 test functions; 2 files with no test function, **0** empty test bodies, and **10** `Test*` classes that collect nothing — not the 5 recorded. Eight of the ten are in `test_results_state_contract_js.py` and its spectra sibling, stating pins in the present tense while holding nothing. Three named remainders still cannot fail: `test_doc_claims.py:92` (loop filters on a string that appears 0 times in its target), `test_monitor.py:342` (`assert callable(fn)` on a `def`), `test_vibration_form_honesty.py:34` (`STILL_OPEN = {}`, iterated empty) | `consolidated-cleanup` § 9 | partly |
| **D4** | doc drift | **The README screenshots are three tabs stale** — five captured, eight ship. Nothing can enforce this (no test can count tabs in a PNG); the *owner* of the count is pinned as of 2026-09-01 | `screenshots.md` | open |
| **R5** | run-decision round | *(priority P3)*  | **`"(this machine)"` means `LOCAL_TARGET` at the prep door and `None` at the bench-grid door.** Real asymmetry, but **the fix is not unification** — `None` is what lets the reader prefer the bundle's own snapshot, and forcing them together broke a live GPU test. The narrow gap: on an unprepped folder with named records, both fit blocks 400 and hide themselves. Fix the *surfacing*, not the value | tried and reverted 2026-09-02; the reasoning is in the code |
| **T4** | run-decision round | *(priority P1)*  | **THREE OF FOUR DONE 2026-09-02.** ✅ `Config = SiestaConfig` **deleted** — alias, both `__all__` entries, the two docstring examples that taught it, and the test, together (its only callers were those). ✅ the gcc pin: the test asserted the substring `gcc_linux-64=14`, which **`14.4` satisfies as well as `14.3`** — and 14.4's gfortran miscompiles SIESTA's `kpoint_t.F90` into wrong k-points, so the one thing the pin exists to prevent was indistinguishable from success; now a property check (three packages, one version, minor present), mutation-tested through `MOLBUILDER_GCC`. ✅ the envelope test: rewritten to the property that is still true (a stray top-level key changes nothing, **ignored not refused**, because a request body is not a config file) — `struct_from_body`'s stale docstring head, which still led with the retired flat shape as *canonical*, fixed with it. ⛔ **`_FLAT_ALIASES` is NOT a code shim and I did not remove it** — `cert`/`key` is a **config-file format users have on disk**, and the loader refuses unknown keys, so deleting it stops their server booting. The no-shims rule is about renames in code; this is a migration and needs your call. Was: **Four tests actively block a correct change**: `test_review_fixes.py:237` (`assert Config is SiestaConfig`) and the three `runtime_config._FLAT_ALIASES` tests pin **backward-compat shims** against the project's no-shims rule; `test_envs_siesta_gpu_recipe.py:89` pins `gcc=14` where `installation.md:202` reverses it; `test_structure_envelope_protocol.py:87` pins a deleted legacy branch — **and that one needs the doc fixed first**, since `web-api.md` still claims `/api/modify/*` accepts the old flattened shape | verified |
| **B2** | run-decision round | *(priority P3)*  | **PARTLY DONE 2026-09-03 — and the number was the wrong instrument.** Of the four shapes named here, only one is mechanically decidable: a `Test*` class whose body is a docstring collects nothing. Five existed. **Three were empty promises** — `TestBuildSiestaHonorsSidecarFrozenAtoms`, `TestWorkspacePayloadRegionsAndFrozen`, `TestGenerateWritesToWorkspace`, each stating in the present tense that it pins something (*"Tests pin both layers"*) while holding no test, so a reader scanning for coverage reads yes. Each is replaced by a pointer at the file that DOES cover it. **Two are deliberate retirement markers** that say so and name their successor — the same call D2 already made for two zero-test files. The other three shapes do not survive measurement: `assert len(X) == 5` where the test BUILT X is a real check, and `m = re.search(...); assert m` is a precondition with the real assertions after it. **A list of ~45 that cannot be re-derived is not a finding anyone can act on** — what is left needs the file-by-file read, not a regex | 5 measured |
| **B3** | run-decision round | *(priority P3)*  | **CLASSIFIED 2026-09-06 — and the population is a fifth of what three earlier counts claimed.** 233, then 256, then 173 were three definitions, none written down. Measured now by `tools/classify_source_reads.py`, which states its definition and can be re-run: of **1,255** assertions over a file's text, **1,147 read GENERATED output** and are correct as text — a property of a real product, never a defect. **108 read hand-written source**, in 31 files. Of those, **59 stay** (51 lints, where text is the only instrument that can prove absence, and 8 vendored/data files) and **49 convert**. Full method, per-bucket file list and the mutation proof are **§ 5h** | 49, not 233 |
| **B4** | run-decision round | *(priority P3)*  | **MEASURED 2026-09-03; the envelope half is done, the fixture half is proposed and NOT applied.** The `_envelope()` count was seven, and only **three** were re-implementations: `test_pseudos.py` and `test_task_setup_tab.py` hand-listed the envelope's fields (so a field the envelope grows would never reach them) and both now go through the one builder — which immediately surfaced a real defect: a test built a 2-atom envelope and overwrote `elements` to three, leaving `atom_names` describing the old atoms, and the route's own guard caught it the moment the canonical dict was used. The third was `test_structure_envelope_protocol.py`, carrying TWO docstrings back to back (the second was dead). The remaining four are a delegating alias and one-line `struct.to_dict()` calls — not the hand-rolled XYZ parsers the helper was written against. **`flask_server`: DONE 2026-09-03, without touching a single scope.** 18 of the 20 now call one context manager, `tests/support/live_server.py::serve()`; each module keeps its own `@pytest.fixture(...)` line, because a scope is a decision about how much state a file's tests share and a de-duplication does not get to change it for them. ~230 lines and 18 now-unused `import threading` go with it. The two left alone pass a non-default app config, which is a real difference. **`_node_esm`: 24 of 47 `*_js.py` files drive it** (the row said 7 of 48), and 13 more shell out to `node` themselves | 3 done · 16 proposed |
| **S1** | architecture seams | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — four branches, `runwrap.py:420 / 446 / 645 / 728`, the same four counted 2026-08-19; 128 engine-name literals in the file |
| **S3** | architecture seams | **`runtime_config`'s untyped scheduler dicts + mixed concerns** | `backend-architecture.md` § 5 (**W3**) | **OPEN — verified 2026-09-06.** `runtime_config._validate_scheduler` still returns `Dict[str, Any]`; overlaps **S6** |
| **S4** | architecture seams | **Transport bypasses the framework.** Gated on a branching workflow, which has no representation today and would arrive as something a person asks for at launch, never as a field a description stores | `backend-architecture.md` § 5 (**W4**) | **RETIRED 2026-09-17.** It was already *mostly wrong* when verified 2026-09-06 — transport preps through the jobset door (`prep.py::_prep_transport`, reached from `prep_calculation`) — and the remainder is now moot: `render_electrode_fdf` is deleted, and `wizard.py` neither renders nor writes. It holds `ElectrodeModel` + `extract_electrode_model`, which DERIVE a lead (`as_structure()` hands `prep` a `Structure`, rendered by the one writer like every other rung). The one path that did write outside the framework, `transport/_cli.py`'s `write_text`, went with the verb. **The "parked wizard task" is closed by deletion, not by re-deriving.** |
| **S6** | architecture seams | **The scheduler menu is handed out as plain dictionaries**, so the typed record and the code using it never meet — how `gpu_partition` came to redirect GPU work from inside an unexamined bag | roadmap § 7.6 phase 3 | **measured partly** — the *record* is typed (`Domain`, `Device`, `Topology`, `Site` in `scheduler/record.py`); the *menu* is not (`known_machines() -> List[Dict[str, object]]`, `Domain.to_row() -> Dict[str, Any]`). Phases 1, 2, 4, 5 are done — phase 2 landed as `scheduler/admit.py`, split out so the check cannot drift from the record it checks |
| **S7** | architecture seams | **The preparation layer against its contract** — **P1** the enforced floor map puts `runwrap` and `jobset/prep` on floor 5; **P3** nothing names the shared package (`jobset/prep._shared_for` globs); **P5** PySCF's seam entry. P2, P4, P6 closed 2026-08-18 | `execution/script-preparation.md` | **P3 CLOSED — verified 2026-09-06**: `_shared_for` calls `seam.shared_package(base)`, and the code names the glob it retired as *'an accident of which suffix the glob happened to name'*. P1 and P5 not re-derived |
| **S16** | architecture seams | **`molbuilder envs advise` does not do a correct job, and does it in the wrong layer.** It prints MPI-rank / OMP-thread / MPS presets for `molbuilder-siesta-gpu` from inside the INSTALLER's command group, while the jobset already owns resource assignment and `runwrap._gpu_runtime_defaults_block` already emits that policy as bash onto the compute node. The user's position, 2026-09-12: *"i don't see any point of keeping this… it has no business in installation."*  **It is wrong by its own stated contract, measured:** its docstring calls itself the user-facing counterpart to the wrapper's policy and its comments claim to mirror it *"exactly"* and be *"kept in sync"*. It is not. On 8-core/1-socket/MPS the wrapper picks **2 ranks x 3 threads** and advise recommends **4 x 1**; on 64-core/2-socket/no-MPS the wrapper picks **2 x 16** and advise recommends **1 x 32**. Its output is exports a person pastes, so following the advice OVERRIDES the placement the system chose, with nothing saying which is right.  **And it enforces a rule that was overruled:** `advise.py:313` clamps ranks by atom count calling it *"the same rule the wrapper uses"*; `runwrap.py:1495` deleted that clamp on the user's 2026-09-03 ruling because the theory behind it was factually wrong (the `propor IMAX=0` abort was a psml problem, not system size) and because how many ranks to spend is the user's decision. `advise --n-atoms 3` still caps to 3 ranks.  **16 tests in `test_envs_advise.py` pin these numbers and no document states them**, so the wrong answers are held in place by coverage with no admission ticket.  Only two of its rules are actually correct (the core-budget and threads-per-rank arithmetic), and both are restatements of the wrapper's.  **Expected resolution: delete `envs advise`, `molbuilder/envs/advise.py` and `test_envs_advise.py`.** Confirm before removal, since the host-probe helpers may have another reader | found 2026-09-12 by the install review, deferred as out of scope | **DONE 2026-09-12** -- deleted (advise.py, test_envs_advise.py, the CLI command, the shim row, the wrapper's tune hint, and numactl's stated reason, which now names its real consumer) |
| **S18** | ops / envs / config | **The 2026-09-12 env-installer and config-and-secrets session has its own hand-over file: [`2026-09-12-env-config-handover.md`](?doc=plans/2026-09-12-env-config-handover.md).**  80 items, each marked by how it was checked (RAN / READ).  It exists because that session's own commit messages are not reliable: three independent audits were told to FALSIFY them, 14 of 16 behavioural claims held, and the failures were overstatements of scope -- four documentation statements written that day are false, two of them in text `envs init-config` ships into a user's config directory.  Its § 1 is the verified DONE list and exists to stop the next session re-deriving settled work; § 2 is the work, grouped as defects introduced (A), false docs (B), an instruction not implemented (C), sweeps that stopped at the first instance (D), pre-existing finds (E) and decisions for the user (F).  **§ 0 states the TARGET first** -- the installer's two state machines and one runner, and config's one resolver / one name / one writer, as 13 checkable invariants T1-T13 -- so every item reads as a named deviation rather than a patch.  **§ H is the residue of the pre-state-machine design**, swept against those invariants rather than against any diff: one question answered in two or more places four times over, a string where the design says state three times, dead parameters, and a door that never sanitises the environment it dispatches into.  **§ I is the config/secret residue**, and its first lesson is that **A11's own text in `architecture.md` still names a pre-consolidation owner**, so the rule as written licenses the three `.parent` climbs it forbids -- fix the rule before the sites.  § G records what was checked and found clean.  **§ 3 is the MIGRATION PLAN** -- eight phases scoped to `install-env.sh`, the `envs` verbs, deployment and how config is placed and validated, each stating what it closes and which end-state row (Z1-Z9) it realises; § 3.0 states that end state so it can be checked, and § 5 records what is deliberately out of scope.  **Read § 1 before touching § 2, and A0 before anything** -- `envs install molbuilder --clean` currently deletes the env the process is running from, and `envs doctor` prints that command as its remedy for a failed host verify.  No clean full-suite result exists for the work yet (F3) | audited 2026-09-12 | open -- nothing in § 2 started |
| **S17** | architecture seams | **`run_tool` dispatched into an env without the mamba-1.x workaround every other path applied** -- so on a host whose manager emits the ``exec -- "$@"`` stub, `run_tool("tleap", ...)` died on a shell error about the manager's generated file, naming nothing to do with AmberTools.  **CLOSED 2026-09-12.** The trade that kept it open -- the workaround needs a prefix, and resolving one costs up to four manager subprocesses, which is wrong on a path walked once per structure build -- is resolved by MEASURING rather than predicting: `builds.dispatch_into_env` is now the one door for all three dispatches, the manager's own `run` is the route, and the prefix is resolved only after that stub has actually been seen.  A working manager still pays exactly one subprocess.  Still not reproducible on this machine (conda only), so it is verified by a fake manager emitting that exact signature | found 2026-09-12 by the install review | **done** |
| **S13** | architecture seams | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built** — the only occurrence in the tree is `transport/wizard.py:65`, a comment naming it |
| **N5** | parse / run files | **The three defects § 5l measured, which outlived it** (§ 5l.a). **① LIVE:** every staged run loses its frozen atoms — `_sidecar.read_frozen_atoms` needs a label to strip the rung and three of four callers omit it, so *"Hide frozen atoms"* and `runtime_info["frozen_atoms"]` are empty for every laddered calculation. **② duplication, NOT a latent defect — re-measured 2026-09-18:** `identity.parse_stage_token` is a second reader of the stage token alongside `runfiles.parse`. They differ on `_geom_optim.xyz` (a declared underscore ROLE, which the second swallows into the stage name) and on `.runwrap-*.log`. **Neither shape is ever passed to it.** Its three callers feed decks (`materialize` ×2, `job.script`) and `.out` / concluded `.molwatch.log` (`parse/dirs/job.py::_detect_stage`); measured on all five real shapes, the two readers AGREE every time. *This row said "latent" and was reported to the user as a bug waiting to happen, with an invented `.xyz` example — user: "why would you fucking pass a .xyz to a parser and ask which step this run belongs to?" Nothing does.* What is real is one grammar with two readers, worth collapsing on the one-home rule and on nothing more urgent. **③** a phantom rung for an unstaged calculation, fixed by ②. *The migration framing is gone with § 5l — these are ordinary defects in `parse/` and `identity`* | § 5l's inventory, re-measured 2026-09-17 | **① FIXED 2026-09-17. ② and ③ open.** *The fix was already in the same module.* `_siesta_fdf_path_for` — the function `model/parse.md` § 5.3 names as the shape a companion lookup may legitimately take — solves the identical problem identically: try the exact stem, then *"fall back to a single `*.fdf` in the same directory"*. `read_frozen_atoms` never got that fallback. It has one now, asked through `sidecars.molstruct.sidecars_in` (the framework's own search, § 4.5) rather than a hand-rolled glob, and **guarded twice**: the lone sidecar's label must be a prefix of the artifact's on a `_` boundary (so an unrelated sidecar that merely happens to be alone is refused), and two candidates decline rather than pick. Licensed by `project-layout.md` § 1.4 — a run directory holds one invocation's output. **Strictly additive**: it runs only where the answer was already nothing. ② and ③ remain, and are one deletion: `identity.parse_stage_token` goes, its three callers (`parse/dirs/job.py:81`, `materialize.py:394`, `:433`) move to `runfiles.parse`, which is right on both shapes the two disagree about |
| ~~**N6**~~ | execution / web | ~~the group launcher, and `files.py`'s duplicated sidecar constant~~ — **CLOSED 2026-09-18, and one of the two was never a question.** *`files.py`*: the copy is deleted; it asks `sidecars.molstruct.sidecar_path_for`, the module that owns the pairing. It needed no decision and should not have been filed as one. *The group launcher*: **withdrawn — there is no defect.** `launch/` holds the GROUP's own machinery (the sequencer, its `.sbatch`, its log, SLURM's stdout) beside the trial directories so they are not mixed among them — a deliberate decision recorded at `submit.py:1199` (*roadmap 7.10, user 2026-08-24*). I read a folder holding files as a claim about the TREE'S LEVELS and manufactured a design question out of a ruling already made. | § 5l, re-read 2026-09-18 | closed |
| ~~**N7**~~ | paths standard | ~~delete the superseded surface in favour of the three verbs~~ — **WITHDRAWN 2026-09-17** with § 5l. There is no replacement surface to migrate onto: `ref.py` is deleted, and `runfiles` + `paths` ARE the framework. The ~40 functions § 5l counted stay as they are; whether that number is itself a defect is a question § 5k's rule never raised and nothing has measured since | § 5l | withdrawn |
| **W23** | front end / ops | **The JupyterNB feature is hand-built where it should be declared — fifteen items, one plan: § 5n.** *(user, 2026-09-15: "why is jupyter.py not following a data-driven design but rather handcrafted jibberish of code?" … "use .json or .jsonl or .toml to help clean this up. this is a systematic design, not some hacking" … "make sure that you don't have other hackish code in the design".)*  The settings a framed Jupyter starts with are expressed three ways inside one function, 40 lines of real Python live inside a string literal no tool can read, the control routes are gated twice on two different facts, the tab's waiting is five ad-hoc timers, and **the whole feature has no test** — 1,610 lines in its own five files, plus the notebook half of `serve_daemon` and six CLI verbs.  The contract, the admission rule that keeps the data file from becoming a dumping ground, the sweep of the other residue, and the order of work are in **§ 5n** | § 5n · found 2026-09-15 | **all fifteen shipped 2026-09-15**, then reviewed with fresh eyes the same day — nine further defects, one of them destructive and one re-creating J13's own bug. § 5n.8 has them and they are fixed; two are recorded as **J16** and **J17** below |
| **W24** | front end / engines | **The transport tab is one panel per ENGINE, not one badge per field.** *(user, 2026-09-15: "i am confused to see mainly pyscf settings on that page while the main design should be focused on transiesta … let's separate transiesta and pySCF engine completely … why don't we use tab of different engine to separate them rather than marking each parameters".)*  Measured: of the 12 fields the tab renders, **5 name PySCF** and the only two with an engine name in the LABEL are `pyscf_functional` / `pyscf_basis` — in the NEGF section, for an engine `registered_engines()` does not list and `engine`'s own `choices` excludes. They are neither sealed nor contract-locked, so they travelled into `task.json`'s device-stage bag and merged into a config where `engine` is hardcoded `"transiesta"` and nothing reads them — the trap the schema endpoint's own docstring refuses. And card 3 claims the advanced fields "stay collapsed"; `tier: advanced` sets `opacity: 0.85` and a bullet, and collapses nothing | **contract settled in `engines/transport.md` § 3.2** — the `index.html` pattern (one card, a sub-tab strip, one panel and one schema endpoint per engine, one config dataclass per engine, which is what actually separates them: `SiestaConfig` and `PySCFConfig` share no field name). A known engine with no backend is a DISABLED tab saying what would make it live (the user's choice against hiding it and against live fields). `TransportConfig` keeps its name — 14 modules and 16 test files reference it — and loses both `pyscf_*` fields; the override gate's vocabulary becomes the selected engine's, so a PySCF name is refused rather than ignored | not started |
| **W25** | engines / science | **THE TRANSPORT TAB'S PARAMETER SURFACE IS INERT — measured against the installed binary, not a manual.** `molbuilder-siesta` ships **SIESTA 5.4.2**, whose fdf labels are compiled into `siesta`/`tbtrans` as literal strings, so this is countable. **Of the 12 fields the tab renders, 10 cannot affect the run:** the four transmission scalars write `TS.TBT.Emin` / `Emax` / `NumE` / `Erange.RelToEF` and `tbtrans` contains **zero** occurrences of `Emin`, `Emax`, `NumE`, `Erange` or `RelToEF` in any spelling; the three contour fields name `TS.ComplexContour.NumCircle` / `NumLine` / `Emin`, all **zero** in `siesta` (only the unused legacy `ComplexContour.NPoles` survives); `log_level` claims `WriteVerbosity`, **zero** in `siesta`. Four of those have no consumer in the tree at all, and **`contour_n_circle` reaches only the Methods paragraph**, which reports a contour the deck never carried — the one finding here with a publication consequence. fdf ignores a label nobody queries, so all of this is SILENT: the run completes and T(E) comes out on tbtrans's default grid. **What is sound:** the five-stage ladder is the standard recipe, the electrode→`.TSHS`→device→`TBT.HS` plumbing is correct and was measured live, every `%block TS.Elec.<name>` key is the right 5.x spelling, and the shared-electronic-contract invariant is the right physics. **Missing controls, each verified present in the binary:** `TBT.Contours` + `%block TBT.Contour.<name>`, `TBT.k` / `TBT.kgrid.MonkhorstPack` (T(E) needs a denser transverse grid than the SCF — the standard convergence study, inexpressible today), `TBT.Elecs.Eta`, `TBT.Contours.Eta`, `TBT.ElectronicTemperature`, `TS.Contours.nEq.Eta` / `Eq.Pole` / `nEq.Fermi.Cutoff`, the `TBT.DOS.*`/`TBT.T.*` outputs **W10** would read, `TS.Elecs.Bulk`, `bloch` (hardcoded `1 1 1`), `TBT.Spin` | § 5o · found 2026-09-15 | **contour fix landing; the rest sequenced in § 5o.5** |
| **W26** | front end / engines | **The transport tab's ORDER, and the Task setup seam — browser walk 2026-09-15 on a real cited junction.** *(user: "i want the full framework of how to do transport calculation scientifically sound/complete, and UI design to be logical, in the right sequence … and works with the 'task setup' framework".)*  The assessment is `engines/transport.md` § 3.4, which also records what is SOUND: the electrode is derived from the citation's labels rather than built by hand, and the three rules that make a lead self-energy trustworthy are enforced — device `kz = 1` as an **error**, electrode `kz` dense as a warning, transverse k matched — plus the sealed electronic contract. **The Task setup seam works**: `prep-plan` answers for a transport description with the five stages, the hierarchical shape, each deck named by the producer, and the same machine/queue card every kind uses.  **What is wrong:** ① the **bias sits in the Describe card**, beside the save button, when it is the experiment — and it governs the whole non-equilibrium half of the density contour, which is inert at zero bias; it belongs at the top of the physics card. ② card 1's atom list **traps the page scroll** (a wheel scrolled rows 38→53 of 444 and left the page still; hit three ways including `Page_Up`) and puts **444 checkboxes ahead of every transport control** in the interactive order. **Diagnosed, and it is MolView's not transport's:** `.molviewer-selection-list-wrap` is `overflow-y: auto` and the list is UNVIRTUALISED, so 444 rows is ~10 screens the wheel is correctly consumed by. Three options in § 3.4.3 — the recommended one is transport-scoped (card 1 exists to CHECK labels, which are assigned on the Molbuilder tab, so a 444-row editor for something uneditable here is the wrong control); virtualising is the source fix and touches SIX templates. ③ Task setup's empty state names only the Structure-optimization tab as a source, when the Transport tab writes `task.json` by its own door. ④ ~~Task setup's "What gets written" promises `<label>.template.toml`~~ — **WITHDRAWN, my error**: `task-setup/viewer.js` already hides the template and structure rows when `calculation === "transport"`, with § 4.1's reason in a comment. I read the EMPTY state, where there is no description to read, and blamed the kind. A claim about a kind made without selecting a folder of that kind | § 3.4 · found 2026-09-15 | ① ③ **done 2026-09-15** · ② open · ④ withdrawn |
| **W27** | engines / execution | **Transport is not on the seven-floor stack — put it there.** *(user: "fix your fucking plan by having first a correct top-down architecture"; "why the fuck is the emit not based on template based approach".)*  The architecture and the derivation are `engines/transport.md` §§ 3.2–3.7.  **THE ROOT, in the project's own vocabulary** (`execution/architecture.md` § 2): `molbuilder/transport/` appears in **none of the seven floors**, while that document's one mention of transport claims its stages "run INSIDE the job system (each an ordinary prep/launch rung)".  Four rules broken: floor 3 renders the text of every file from a `ParameterSet` through `prepare_deck` — transport's `render_script` concatenates literal f-strings; floor 2 holds what the person asked for — transport had no template, so `TransportConfig` became the definition; `prep` is the conductor and may never decide — `_prep_transport` is a second conductor that does; floor 2 must never name a machine — `max_memory_mb`/`num_threads` sit on it.  **WHY, from the history:** `transiesta.py::render_script` 2026-06-10; the pipeline landed 2026-08-19 (`refactor(prep): the seam carries the engine's FORM`) and migrated siesta + pyscf in one commit, leaving transport behind; the composite was then built outward from the unmigrated emitter.  `template.md` § 9.2 has recorded the missing arm all along, filed as one lost feature (USER-CUSTOM) rather than as *transport cannot render from a template*.  **MEASURED:** the seed deck `prep` renders carries **13 keywords / 4 blocks** against a template offering **45 deck-reaching items**.  **EVERY KNOWN DEFECT IS DOWNSTREAM:** the seed dying at 1000 SCF iterations (`MaxSCFIterations` cannot travel from the citation — no transport emitter writes it and no transport field held it); the device deck aborting on "the continued fraction method requires at least 20 poles" (the pole *energy* is written, never `TS.Contours.Eq.Pole.N`); `TBT.k` as a bare scalar the parser rejects; `tbt_k_grid`'s unguarded transport axis; the electronic contract as two frozensets and a twice-spelled predicate.  **ORDER OF WORK IS FLOOR ORDER**, § 3.6: (1) render through `spec_for`/`DeckSpec`/`prepare_deck`, (2) no value syntax by hand, (3) validation report + read-back check, (4) `_prep_transport` stops deciding, then the floor-2 items.  My first draft of this plan put the pipeline at step 6 of 7 because it was written from the parameters down; read from the floors down it is step 1.  **DONE:** 4a the `citation` marker (`template.md` § 6.4's sibling answerer, per-kind); 4b/4c transport's parameters as 17 catalogue rows + 7 shared rows tagged `citation = ["transport"]` + 9 relaxation rows tagged `optimization`, with matching `SiestaConfig` fields — which also made `electrode_kz` (invariant I9) reachable from a description for the first time. | `engines/transport.md` §§ 3.2–3.7 · 2026-09-15 | 4a/4b/4c **done** · floor-3 migration **open** |
| **TR1–TR3** | engines / transport | **Transport's description gains a template, and the catalogue gains what the design needs.** T1: `jobset init --calculation transport` writes `<label>.template.toml` with its Class A values defaulted from the cited relaxation — today a transport folder has **no template at all**, so the shared baseline has no home and the stage table has nothing to read. T2: the three keywords with no catalogue row — `TS.HS.Save`, the equilibrium pole **count**, `TS.Voltage`. T3: `kgrid` split, because its three components fall in three classes. **Reuses `build_description` / `template_with_values`, which already narrows the catalogue by `calculation`** | § 5p · `transport.md` § 2a | **ALL DONE 2026-09-16** — TR1 § 5p.3c, TR2/TR3 § 5p.3d |
| **TR4–TR6** | engines / transport | **Transport renders through the one path.** T4: `_prep_transport` keeps the compose and hands to `resolve`, so a transport run has a `ParameterSet` with provenance and `--pipeline-log` stops being a no-op. T5: the remaining four rungs onto `spec_for` — ⚠️ **blocked on a seam question**, what a composite kind hands its renderer, since `spec_for(struct, cfg, stage_token=)` does not carry the `ComposedJunction`. T6: `TransportConfig` retires | § 5p · § 2a | **TR4 done 2026-09-16** (§ 5p.3e); TR6 half-done — `siesta_config_for` already deleted; **TR4 and TR5 DONE**; TR6 all but done — one projection survives, feeding the lifted NEGF block, and goes with it |
| **TR7–TR8** | front end / transport | **The tab reads the catalogue, and an override reaches the rung that owns it.** T7: the bespoke dataclass form is deleted for the kind-aware catalogue route plus the existing **stage table** — rows are stages, columns are `varies`, an empty cell inherits the template. T8 fixes the **live defect**: every override currently lands on the `device` bag, so a parameter the transmission owns never reaches the transmission deck. **T8 needs the `stages = [...]` declaration** (§ 5p.3), not yet approved | § 5p · § 2a.7 | **TR8 done 2026-09-16** (§ 5p.3k) — the live defect is closed. **TR7 half done** (§ 5p.3l): the seal's reason is corrected and the form-B hole closed; the interface half — § 2a.6's Panel 0 — awaits a ruling on where Class A is edited |
| **TR9–TR10** | execution / front end | **Grouping and the deliverable.** T9: the preparatory block (seed + both leads) as one submission — ⚠️ **must first reconcile with `task-setup.md` § 1's "no run-all-stages button"**; § 5p.5 has the argument, and if it does not hold T9 is withdrawn rather than the rule bent. T10: Results reads the transmission with its **treatment label** and provenance chain, so a linear-response I–V is never mistaken for a finite-bias one | § 5p · § 2a.12 | not started |
| **W29** | engines / front end | **The first validation of a junction does not exist: nothing checks that the atoms labeled `L-electrode`/`R-electrode` are the FROZEN atoms.** *(user, 2026-09-16, describing the design: "it will look for the labeled left electrodes and make sure that they are the fixed atoms … this is the first validation or check".)*  What exists is a different, LATER gate — `compose.py` checks the electrode atoms did not MOVE, by comparing the cited deck's coordinates against the `.XV`, **after the relaxation has run**. Label the electrodes and forget to freeze them and nothing objects: the relaxation runs, the leads relax, and the junction is unusable at compose time — a wasted relaxation on a metal junction. The closest thing today is a hint inside an unrelated warning (`validation/sidecar.py`: *"If you meant those atoms to be held fixed, assign them to frozen_atoms in /modify"*), which suggests but does not check. Belongs in the settings gate on the OPTIMIZATION of a junction — the run being set up when the mistake is made. Tier 2: detectable before anything runs | § 5p.3g · found 2026-09-16 | **DONE 2026-09-16** — warns rather than refuses; the severity is open to a ruling |
| **W28** | tests / front end | **`test_build_e2e.py::test_commit_mounts_molview_card` is FLAKY, and it is a race in the test.** It waits for the MolView card and its canvas to mount, then *immediately* reads `.molviewer-selection-count` — which renders slightly later, so on a loaded machine the text has no `" of "` and the split raises `IndexError`. **Measured 2026-09-16, interleaved against a pristine-HEAD worktree under identical load: mine 3/4 pass, HEAD 2/4.** Both flaky, so it is not a regression — and the first, unfair comparison (one HEAD run against three of mine) read it as one. The fix is to wait for the count line as well as the canvas. Recorded rather than fixed: it is a false signal for everyone who runs the suite | found 2026-09-16 while doing TR1 | not started |

---

## 3. Configuration and ops — merged

*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---

## 4. The front end — merged

*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---

## 4a. Needs a decision from you — merged

*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---

## 5. Documentation drift — merged

*Merged into § 2 on 2026-09-10 — one list, not four. The number stays because other documents link to it.*

---

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
neither count). W2/W5/W8 are accurate exactly as
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
| ~~P0~~ | ~~**R6**~~ | **DONE 2026-09-02.** | found by `test_task_setup_prep_e2e.py`; user: *"all environments have to be explicitly probed and stored. no environment json, error"* |
| ~~P1~~ | ~~**R7**~~ | **DONE 2026-09-02** *(user: "safety first … compatibility is not an issue. | pinned + mutation-tested |
| ~~P1~~ | ~~**T4b**~~ | **DONE 2026-09-02.** | pinned |

### Tests that are not testing

| P | # | item | evidence |
|---|---|---|---|
| ~~P1~~ | ~~**T1**~~ | **DONE 2026-09-02 — by making the claim TRUE, not by editing it away.** | verified |
| ~~P1~~ | ~~**T2**~~ | **DONE 2026-09-02.** Resolves from `__file__` like every other path in the file, and asserts the source list is non-empty so a future blind run says so. | verified |
| ~~P1~~ | ~~**T3**~~ | **DONE 2026-09-02.** Names `dataclasses.FrozenInstanceError`; mutation-tested by un-freezing the dataclass. | reported, spot-checked |

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
| ↳ | **B3.1** | **DONE 2026-09-03 — the self-confessing subset.** A test whose own docstring says the real check lives elsewhere is retired, and the test it names gets written (`process/testing.md` § 3a.1, and `code-audit.md` § 5 rule 5 which was still *instructing* auditors to write these). Eleven confessed; **three were contrastive, not confessional** (`test_page_ids_unique.py`, `test_pages_no_js_errors.py`, `test_pdb_workflow_integration.py` all drive the artifact and cite string-pins only as what failed before) and one more survived reading (`TestSpectraIssuesPanelSeverityCoverage` is a CSS one-home lint whose confession describes the greps it replaced). **40 test functions removed** (38 deleted, 2 rehomed) **and 12 written** — 10 new plus the 2 rehomed. Every replacement mutation-verified. Every replacement mutation-verified. Two were *not* written and the reason is on record where each belongs: the second-load widget defect is unreachable by a single break (two publishers each rescue the other), and "a row is born at its value in force" states no rule any document carries | 40 |
| ↳ | **B3.2** | **DONE 2026-09-03 — reviewing the replacements found eight defects in them**, three substantive: a Slack channel was tested carrying a signing key (there is no such control — only a listener has one), an assertion demanded that NO part of a webhook appear when `MASK_TAIL = 4` shows the last four on purpose, and an `ok` check passed on an absent key. Two were vacuous (a vocabulary check that passes on a class with no tags; a chooser asserted to exist but not to offer anything) and three fragile (both timer tests counted the page's own intervals; a checkpoint picked by position not name; one dialog accepted where `_restore` asks twice). A green run plus one mutation had hidden all of it. Asserting the page reports no JS errors also found a **live product bug** — `pattern="[A-Za-z0-9_-]{1,64}"` never compiled under the `v` flag, so the channel-name rule was stated and never enforced (`d243e852`) | 8 |
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
2026-09-04**, replaced by `job.run_status`. That work is done and is not what
follows.

**What follows is a NEW consolidation. Steps 1 AND 2 are DONE (2026-09-18).**
`parse/types.py` carries `RunDirResult`, `parse/dirs/rundir.py` carries
`JobDirParser` + `openable_in`, the parser is registered, and
`parse_dir(<a run directory>)` answers. `web/blueprints/watch.py` calls
`openable_in` and its own copy of the chain is deleted.

> **Step 2 moved ONE of the six sites, and withdrew the other five with
> measurements.** The caller map below was written from the names — six
> functions that all take a run directory — and five of them turned out not to
> be asking a run-directory question. The withdrawals are in the table itself,
> each with the measurement that settles it. **This is the third time this
> section's table has lost rows to re-reading the code** (two went on
> 2026-09-04), and the pattern is the same each time: a function whose
> ARGUMENT is a directory is not thereby answering *a question about the
> directory*.

**The shape.** One DirParser answers everything asked *about a run
directory*; the four fields each have a named reader before a line is
written (§ 5.0's table). `active` is picked **stage, then mtime** (user
ruling) — `summarize`'s highest-`-runN` rule loses, because a run index says
nothing about which stage a file belongs to.

### The complete caller map, measured

Only **three** modules consume any of this, which is what makes the migration
checkable rather than hopeful:

> **⚠ THE MAP WAS BUILT FROM NAMES, AND THAT IS ITS ONE WEAKNESS** *(found
> 2026-09-18, by a full-text audit of the four caller files rather than by
> grep)*. Every row below was found by searching for `run_status`,
> `_resolve_run_directory`, `engine_of`, `atom_metadata_json_for_run_dir` —
> so a module that answers the same QUESTION under its own names was never a
> candidate to appear, let alone to withdraw. `transport/record.py` is
> exactly that, and it is code written for this very milestone two days
> before the audit. **A completeness check over function names cannot see a
> re-implementation**; the row below was added by reading the files.


| what it does today | where | outcome, measured 2026-09-18 |
|---|---|---|
| `_resolve_run_directory(dir)` — the 4-rung chain | `web/blueprints/watch.py` ×1 | ✅ **MOVED** to `.openable` + `.attempts`. The only true duplicate in the list: 132 lines of second copy, plus five helpers serving only it — **166 lines deleted from the web layer**. |
| `run_status(dir)` | defined `parse/dirs/job.py`; called from `jobset/runstatus.py` ×1 | ❌ **WITHDRAWN.** It passes `out_glob = sh.stage_glob(token, label)` — `"*"` in the hierarchy but **`<label>_<token>*` in flat**, where one directory holds every rung. `parse_dir` has no narrowing, so flat would go back to *every stage row showing the newest stage's state* — the exact defect fixed 2026-09-08. And there is nothing to de-duplicate: `run_status` already has one home in the parse layer, which this section itself noted on 2026-09-07. **Asking a directory door a per-RUNG question is § 5c.1's problem wearing a different hat.** |
| `_engine_of(...)` | `web/blueprints/watch.py` ×3 | ❌ **WITHDRAWN.** Not a reader — a three-source fallback (`engine_of` → `payload["source_format"]` → `parser_cls.name`) whose first source *is* the one implementation. **Two of the three sites pass `search_dir=None`**: an upload has no run directory at all. Routing it through the door means parsing a whole directory — status over every `.out`, the four-rung chain — to obtain one string. That is precisely what got the predecessor deleted. |
| `engine_of(dir)` | `web/blueprints/watch.py` ×1 | ❌ **WITHDRAWN** — it IS the one implementation (`running-a-job.md` § 4.2 owns the rule). `.engine` calls it; a caller that wants only the engine should keep calling it too. |
| `atom_metadata_json_for_run_dir(dir)` | `web/blueprints/watch.py` ×1 | ❌ **WITHDRAWN** — already in `parse/dirs/`, already one home, its own verb. Same boundary as `contract_of` below. |
| `summarize`'s per-trial reads | `jobset/summarize.py` ×4 | ❌ **WITHDRAWN**, and the prose two sub-sections below already said why without the table catching up: `_latest_run_file` goes **through `runfiles.find`**, the stage is already chosen by the basename, and the only remaining choice is the RUN INDEX. Not `active`'s question. |
| `contract_of(dir)` | `parse/dirs/run_info.py` ×1 | unchanged — its own verb, over `.files["fdf"]` if the door is handy |
| **`transport/record.py`** — `_stage_facts` ×1 + `collect_record` ×1 | **WAS IN NO ROW** *(added 2026-09-18)* | ⬜ **the map's own blind spot.** Both pick the newest `.out` by **mtime alone** — a third rule beside the door's (stage, mtime) and `summarize`'s run index. **Not a defect**: `task.py` refuses a transport calculation whose shape is not hierarchical, so each rung has its own directory and there is only one stage to order — the rules coincide. But it is a third place the question is answered, and § 5.1's ruling names only two. **Nor is it a drop-in move**: `RunStatus` carries `state / detail / last_change_at / active_source` and `_stage_facts` needs `scf_converged`, the final energy and the lead `ef` — so the door would pick the file and the caller would still parse it. |
| ~~**the Results file picker** — `lib/results/file-picker.js`~~ | **the BROWSER** | ❌ **STRUCK 2026-09-18 — NOT A CALLER OF THIS DOOR, and it never was.** Its question is *"these five directories are one run"*, which is the **ladder's**, not a directory's; `openable` answers about one directory by construction. `stages.md` § 6.7 puts the layout in `task.json` and forbids inferring it from data, so `JobDirParser` — handed a bare path — **cannot** answer per-rung and must not be extended to try. The ladder door already exists and already reads the declared shape: `jobset/runstatus.py::jobset_status`. What the picker needs is an **HTTP surface over that**, which is a Results-tab feature tracked in § 5p.3p — not a step of this migration. *(Listed here as "the seventh consumer" from 2026-09-18 until later the same day, which made a finished migration read as unfinished every time its status was asked.)* |

> **What step 2 therefore proves about step 1.** The door earned its keep on
> exactly one site, and that site is the one that could not be fixed any other
> way: a browser cannot import Python, so the chain had to leave the web layer
> before the Results tab could ever ask it. Every other row was a name
> collision. Had the map been executed as written, five call sites would have
> been made slower and one of them wrong.

> **The seventh consumer is a BROWSER, and every other row here is
> server-side** *(added 2026-09-18, owed by § 5p.3p step 7 since 2026-09-17 and
> deferred twice while other work went in)*.
>
> The picker enumerates a folder and decides what to list **from filenames, in
> JavaScript** — it asks no server door, because none answers *what is in this
> directory*. `/api/results/contract` answers `info.calculation` and five other
> fields, none of which is *what should a viewer open here*.
>
> **That guess is why a finished transport run is invisible.**
> `<label>.transport.json` matches no presenter, so the picker drops it; the
> five rungs' `.out` files are each claimed by the trajectory viewer, so the
> ladder lists as five unrelated optimizations (§ 5p.3p.3, items 1–2).
> Transport is not the cause — it is the first result that is neither a
> trajectory nor a spectrum, so the guess stops producing a plausible answer.
>
> **What this row costs § 5c:** the door needs an HTTP surface, not just a
> Python one. Every other consumer imports it; this one must fetch it.

#### 5c.1 `openable` is one answer per DIRECTORY, and a ladder is five

*Owed by § 5p.3p step 7, same date, same deferral.*

`RunDirResult.openable` answers *what should a viewer load here*, once, for one
directory. **A transport run is five directories that are one result** — seed,
both leads, device, transmission — each a real SIESTA run with its own `.out`.

Neither mechanism in play can say so. `absorbs` (`web/presenters.md` § 2)
collapses siblings **within one directory**; it is asked *"does this master
subsume that sibling?"* and has no way to express *"these five folders are one
run."* `RunDirResult` as § 5.0 specifies it is per-directory by construction.

**This is not transport's question.** It reaches every multi-rung calculation —
a staged optimization is the same shape and only looks fine today because each
rung's log is independently interesting. It needs a home **before** code moves,
because the answer decides whether the door returns one result per directory
(and something above composes them) or learns about runs that span several.

**It is genuinely open.** No recommendation is recorded here on purpose: the two
shapes differ in what the door IS, not in how it is written.

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
again.** It had no DirParser between 2026-09-04 and 2026-09-18 and could only
refuse. *(Measured before shipping it: no call site in the tree passes a
directory to `detect()` — the three in `parse/dirs/job.py`,
`transport/record.py` and `engines/pyscf.py` all pass a concrete file — so the
registration is additive, not a change of answer anywhere.)*

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

1. ~~Write `RunDirResult` + `JobDirParser`, absorbing the chain verbatim.
   Prove equivalence on the real tree BEFORE any caller moves — the
   `run_status` split did this (113/113 identical) and it is the reason that
   deletion was safe.~~ **DONE 2026-09-18.**

   The proof ran over every run directory in `projects/`: `JobDirParser.parse`
   against `_resolve_run_directory`, `run_status` and `engine_of` —
   **identical 141/141**. It found two real absorption bugs first, which is
   what a verbatim gate is for: rung 4's attempt messages had been
   paraphrased, and rung 4's `*_geom_optim.xyz` had been rewritten to
   `find_by_role`, which *raises* on an underscore role with no label — the
   exact case rung 4 exists for. Both fixed, then 141/141.

   Registered in `parse/dirs/__init__.py` (the registry's own rule: *"per-package
   `__init__.py` files own the registration order"*). Three tests in
   `tests/parse/dirs/test_rundir.py` hold what step 1 ADDS — that the registry
   answers, that a prepped-but-unrun stage is still a run directory, and § 5.1's
   `active` ≠ `openable`; the chain's own rungs stay held by the three tests in
   `test_path_framework_doors.py`, which move onto `openable_in` at step 3.
2. ~~Move `runstatus` (1 site), then `watch` (**6**, not 9), then
   `summarize` (5).~~ **DONE 2026-09-18 — one site moved, five rows
   withdrawn with measurements** (the table above carries each one).

   `watch`'s `_resolve_run_directory` call became `openable_in`. The other
   five asked questions this door does not answer, and the measurements are
   in the table rather than here so the map and its verdict cannot drift
   apart.
3. ~~Delete the absorbed functions and sweep their documents.~~ **DONE with
   step 2**, since only one function was absorbed: `_resolve_run_directory`
   and its five private helpers are gone, and `job-contracts.md` § 2.4,
   `model/parse.md` § 5.2, `process/testing.md`'s door map and the three
   tests in `test_path_framework_doors.py` all name `openable_in` now.
4. ~~**Re-run the caller map and require it empty.**~~ **DONE 2026-09-18 —
   it comes back empty.** One row moved, five withdrawn with measurements,
   and the seventh (the Results picker) struck as never having been a caller
   of this door. **§ 5c IS CLOSED.**

   The picker's work is real and is `§ 5p.3p`'s: an HTTP surface over
   `jobset_status`, the ladder door that already exists.

   *(Two method failures are recorded above rather than quietly fixed,
   because both would repeat: the map was built from FUNCTION NAMES, so it
   could not see `transport/record.py` answering the same question under its
   own names; and § 5c.1 was posed as "should `RunDirResult` learn about
   runs spanning several directories" when the answer was already on the
   books — something above it composes them, and that something is
   `jobset_status`.)*

---


## 5d / 5i / 5j — CLOSED, archived 2026-09-07

*In [`archive/2026-09-07-plan-closed-sections.md`](?doc=archive/2026-09-07-plan-closed-sections.md); the 2026-09-10 consolidation moved this pointer's body to [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md).*

---

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
(S8, S11), **one is mostly wrong** (S4), **four are confirmed OPEN** (S1, S3,
S6, S13), **S9 is now DONE** — the two documents
stopped disagreeing on 2026-09-06 and `backend-architecture.md` § 2
retracts the claim in writing — and **two state cells describe the wrong
seam**: S10's answers S9's question, and S14's describes S12's GPU guard and
asserts it "does not exist" when S12's own cell shows it does.

> **Both unrecorded states are now measured — 2026-09-10**, and both are
> closed; the rows are in
> [`archive/2026-09-10-plan-consolidation.md` § 6](?doc=archive/2026-09-10-plan-consolidation.md).
> **S14** (a flat stage's verdict read folder-wide) was fixed 2026-09-08:
> `parse/dirs/job.py::run_status(run_dir, match)` narrows the bucket and
> `jobset/runstatus.py:267` passes the same glob the existence check used,
> with the before/after in the docstring. **S10-M2a** (the record naming one
> partition while the header submits to another) cannot happen through the
> placement door: `scheduler/emit.py::Directives.of` takes the partition and
> QoS **from a `Placement`**, and `place(..., named=…)` refuses a domain the
> machine record does not offer — checked against real Sol output, where
> `bench-group-gpu-G2K24C1.sbatch` carries `-p htc -q public -t 0-04:00:00`
> and the record's `htc` row reads `partition: htc, qos: public,
> max_time: 4:00:00`. M3's `account` half has **no door to declare one**:
> `runtime_config` has no `scheduler.account` key, nothing emits `#SBATCH
> -A`, and `configuration.md` § M-1 already records `Site.account` as a
> field *"nothing has ever written"* — so the missing record is the missing
> feature, and there is one fact, not two disagreeing.
>
> **The lesson is the same one § 5a states.** The two rows nobody could close
> were exactly the two whose state cell had been misfiled onto a neighbour —
> a row with no evidence of its own is a row that stays open forever. The line counts S11 quoted — `render_run_wrapper` grew to 2,134 lines and 499 f-strings
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

*The rows that were here are in [§ 2](#2-open--the-one-list) — one list since 2026-09-10. The design above them is the substance and stays here.*

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



## 5k — CLOSED 2026-09-08

*The paths framework's first program, M1–M8. **Its rule is the part that survived**: [`execution/project-layout.md` § 4.5](?doc=execution/project-layout.md), enforced by `tools/classify_path_finders.py --check` and `tests/test_path_framework.py`. The follow-on STANDARD that was to replace the API (§ 5l) is **retired 2026-09-17**; this section's rule and guards are untouched by that. Body archived 2026-09-10.*

---

## 5l — RETIRED 2026-09-17 *(user ruling)*

**The paths STANDARD — one address, three verbs — is retired, and
`molbuilder/ref.py` and `tests/test_ref_address.py` are deleted with it.**

**What it was.** After § 5k closed 24 handcrafted searches, the API had been
grown *"by adding a door for whatever question a call site happened to ask"* —
~40 public functions, four APIs answering *which files are this rung's*, three
answering *what is this file called*, sixteen answering *where does this thing
live*. The user's 2026-09-08 ruling was that the direction of fit must reverse:
*"don't twist the API design such that you can fit arbitrary kind of need, but
rather a standardized API that has reasonable extensibility."* § 5l's answer was
a single `Ref(label, stage, bench, attempt, role, counters, fields)` with
`compose` / `find` / `parse` over it.

**What actually happened.** `ref.py` was written — 444 lines, the address and
all three verbs, layers 2 and 5 of the six-layer stack § 5l.5 specifies. **The
migration never ran.** N5, N6 and N7 were never started, so in every revision it
existed the module's only importer was its own test. It was carried as L1 in
`architecture.md`'s index and in `test_layering.py`'s enforced set, so it read
as part of the framework while answering nothing.

**Why retiring is the right call and not a loss.** The standard's own premise was
that ~40 functions answering four questions is too many doors. Shipping a
seventh module implementing a *replacement* vocabulary, and then not migrating,
made it ~40 functions **plus** a parallel address nobody called — the exact
failure § 5l was written to end, arrived from the other side.
`process/testing.md`'s rule names this shape: *"a unification that ADDS has
usually not unified anything — it has added a layer and kept the old surface."*

**What stays, and is untouched by this.** § 5k's RULE — *for every name it
composes, the framework owns the search* — is closed, enforced, and green:
`tools/classify_path_finders.py` (three axes, in `--check`),
`tests/test_path_framework.py` (22 tests), and the `GUARDED_UNDECLARED` set that
holds the two undeclared segments to exactly the ten known sites. None of them
referenced `ref`. `runfiles` (catalogue + name grammar) and `paths` (layout)
remain the L1 path modules, and § 5l.7's constraints on them remain true
independently: **stdlib-only, the shape is DECLARED never inferred, `identity.stage_token`
is the one speller of `<NN>_<name>`, and `project-layout.md` § 2.6 is the
authority on the tree.**

### 5l.a What § 5l FOUND that outlives it — re-homed, not dropped

Retiring the standard retires its *answer*. It does not retire the defects the
inventory measured on the way, and one of them is live and user-facing. **These
are now `N5` in § 2's open list**, restated here because their original framing
— *"under § 5l that is `find(root, role=…)`"* — died with the section and a
reader must not go looking for it.

**① Every staged run loses its frozen atoms — ~~LIVE~~ FIXED 2026-09-17.**
`parse/engines/_sidecar.read_frozen_atoms(traj_path, label="")` needs the label
to strip the rung off an artifact stem, and **three of its four callers omit it**
(`parse/engines/molwatch.py:590`, `parse/engines/pyscf.py:626`,
`parse/engines/siesta.py:1877`; only `pyscf.py:402` passes one). So
*"Hide frozen atoms"* and `runtime_info["frozen_atoms"]` are empty for every
laddered calculation:

| artifact | no label | with label |
|---|---|---|
| `bdt.out` | `[5, 6, 7]` | `[5, 6, 7]` |
| `bdt_01_coarse.out` | `[]` | `[5, 6, 7]` |
| `bdt_01_coarse.molwatch.log` | `[]` | `[5, 6, 7]` |

The function's own comment states the mechanism and why the obvious fix is
wrong: *"THE LABEL IS REQUIRED TO STRIP THE RUNG, AND GUESSING IT IS A BUG"* —
`bdt_01_coarse.out` (rung `01_coarse` of `bdt`) and `sample_02_test.out` (an
unstaged calculation whose label reads that way) are the same shape, and guessing
handed the second one a **different calculation's sidecar**.

> **And retiring § 5l cost nothing here, which is worth recording because I
> predicted it would.** The fix § 5l proposed was *"ask by ROLE"* —
> `find(root, role=…)` — and I wrote that losing it was *"the one real cost of
> retiring"*. It was not. **The answer was already in the same file**:
> `_siesta_fdf_path_for` pairs a `.out` with its `.fdf` by trying the exact
> stem and then *"falling back to a single `*.fdf` in the same directory"*, and
> `model/parse.md` § 5.3 names that very function as the shape a companion
> lookup may legitimately take. `read_frozen_atoms` sat beside it for three
> months without the fallback.
>
> *The lesson is `feedback_read_the_code_path_before_probing`'s, in a new
> place: I reached for a retired abstraction to solve a problem its own module
> had already solved twice over. Read the neighbours before designing the
> door.* **Fixed 2026-09-17**, guarded by a prefix check and an
> ambiguity-declines rule, both mutation-tested.

**② The stage token has two readers and they disagree — latent.**
`identity.parse_stage_token` splits a filename inline, which `runfiles.py`'s own
header forbids (*"nothing composes or splits a run-file name inline"*), and the
two answer differently:

| filename | `identity.parse_stage_token` | `runfiles.parse` |
|---|---|---|
| `bdt_01_coarse_geom_optim.xyz` | stage `01_coarse_geom_optim` | `01_coarse` + role `_geom_optim.xyz` |
| `bdt_01_coarse.runwrap-…log` | no stage | `01_coarse` |

`runfiles` is right both times. Latent because the three callers
(`parse/dirs/job.py:81`, `jobset/materialize.py:394`, `:433` — re-measured
2026-09-17) feed it only deck and `.out` names, where the two agree. **The
duplication is the defect**, and deleting the second reader does not need the
standard.

**③ A phantom rung for an unstaged calculation.**
`parse_stage_token("run_01_setup.out")` returns `(1, "setup")` with no label and
`None` with it. `parse/dirs/job.py`'s `_detect_stage` omits the label, and it
feeds the sort that picks `active = sorted_outs[-1]` — *which file speaks for the
directory* in `run_status`. Fixed by ② rather than separately.

**Two decisions § 5l was holding, now standing on their own feet as `N6`:**

- **The group launcher.** `jobset/submit.py` builds `<container>/launch/<name>.run.sh`
  and `.sbatch`; `launch_dir` is spelled at `:1204` (`container / "launch"`) and
  `:1517` (`stage_dir / "launch"`). `<name>` is a **group**, not a calculation.
  `launch` is one of the two undeclared segments `GUARDED_UNDECLARED` holds, so
  **§ 5k's live guard already owns this**, and the question is § 2.6's, not
  § 5l's: does a group become a real qualifier on ④, or does the group launcher
  move to where a launcher belongs?
- **`web/blueprints/files.py`.** `_SIDECAR_SUFFIX = ".molstruct.json"` at `:260`
  against `sidecars.molstruct.SUFFIX` — re-measured 2026-09-17, still the only
  duplicated constant of its kind — plus a re-implementation of the pairing rule
  at `:271`. Its written reason is *"to keep its dependency graph narrow"*, which
  layering does not require (`sidecars` is L2, `web` L3). Either import the door
  and delete the copy, or keep the choice and add the parity test
  `process/code-audit.md` D4 then demands. **This is a code-audit item and never
  needed the standard.**

### 5l.c "What label does this deck carry?" — one fact, three readers, and where it belongs *(investigated 2026-09-17)*

**Why this section exists.** N5 ②/③ need the calculation's label to call
`runfiles.parse`, and one caller (`parse/dirs/job.py`) has none. I first
concluded that no door answered *"what calculation is this directory for?"* and
that I would have to add one. **That was wrong** — the question is answered in
three places already. The investigation is here because the ANSWER is a layer
question, not a missing function.

| # | reader | where it runs | how it reads | live? |
|---|---|---|---|---|
| 1 | `runwrap.py`'s awk, inside the generated wrapper | the **compute node**, at LAUNCH | lower-cases, strips quotes, takes `$2`, then sanitises to `[A-Za-z0-9._-]` or falls back to the basename | ✅ |
| 2 | `parse/dirs/rundir.py::openable_in` (via `parse.fdf.system_label` / `pyscf.input.job_name`) | the server, at read time | case-insensitive, allows the dotted spelling, restricts the value charset | ✅ — called at `rundir.py:109` and `:127`. *(Was `watch.py::_basename_from_fdf` / `_basename_from_py` at `:269` / `:285`; those two wrappers went with the chain on 2026-09-18, § 5c step 2.)* |
| 3 | `parse/dirs/_assembler_helpers.py::extract_system_label` | — | case-SENSITIVE, requires whitespace, accepts any non-space value | ❌ **deleted 2026-09-06**, "six helpers, zero callers" |

**Two of them disagreed**, and the one that survived is in a web blueprint.

#### The awk one is NOT irreplaceable — it is G7's surviving violation

**`execution/gpu.md` G7 already ruled on this exact class, 2026-08-23:**

> **"G7 — The value travels; the deck is not re-read for it."** *`use_gpu`
> declares `read_by = ["wrapper"]` precisely so the wrapper can be **handed**
> the value… The deck scan remains only for a caller that states nothing,
> which is not re-deriving: that path has no allocation to ask.*

**The label is the same case and was never converted.** Measured 2026-09-17:
`[item.system_label]` is already a catalogue item carrying
`anchor = "SystemLabel"` / `engine_key = "SystemLabel"`, and **exactly one row
in the whole catalogue declares `read_by = ["wrapper"]`** — `use_gpu`, G7's own
landing. `system_label` does not, which is the only reason the wrapper digs for
a value `prep` has in hand. G7's escape clause does not cover it: there is no
caller here that states nothing.

**So the earlier reasoning in this section is superseded, all of it.** I argued
in turn that the awk (a) could not be replaced because molbuilder is not
importable there — false, the env is activated 210 lines earlier; (b) had to be
lax because decks get hand-edited — false, the deck fences a `user-custom` zone
and warns against the rest; (c) was justified because the *value* can change
after prep — true but worthless, because the wrapper's own output naming is a
baked literal, so such a run is already inconsistent with itself. **The real
answer was a ruling already on the books.**

#### What the awk's comment gets right, and what it gets wrong

*Its stated reason:* it re-reads at launch rather than having the label baked in,
because the wrapper runs *"on the deck as it is at LAUNCH, after a person may
have edited it, and a cold-restart sweep keyed to the wrong label moves aside
files the engine will then not find."* **That premise still holds** — `prep`
writes, `launch` runs later, and the USER-CUSTOM zone plus the web edit-save
path exist precisely so a person may edit a deck in between. A sweep keyed to
the prep-time label would move the wrong files, and SIESTA looks up
`SystemLabel`, not the wrapper's filename. Its laxness follows from the same
premise: fdf is case-insensitive and mawk/BSD awk have no `IGNORECASE`, so
lower-casing is required, not sloppy; and the sanitiser downstream re-imposes
the charset the generator would have guaranteed.

**I claimed a structural reason here and it was FALSE — corrected 2026-09-17
after the user challenged it.** I wrote that *"molbuilder is NOT IMPORTABLE
where that code runs, so the awk cannot call a Python reader."* **Measured in a
real emitted wrapper** (`projects/claude-audit/optimization/benzene-flat/…run.sh`):
`conda activate` is at line **210**, the awk is at line **420**. The environment
is live when the label is read; Python is available. And molbuilder not being
importable in the ENGINE's env is precisely why `mb_monitor.py` and
`config_dir.py` ship beside a job — a mechanism that proves a standalone reader
*could* travel, not that it couldn't.

**So the second implementation is a CHOICE, and these are its real grounds:**

* **Blast radius, measured.** Adding a companion file has cost a production
  outage in this repo: two hand-kept lists of what travels with the monitor
  drifted, `config_dir.py` went into one and not the other, and *"every
  production run's monitor died at import, stderr to /dev/null: no [MACHINE],
  no status, no util.csv, no reports"* — found only by a bench run that happened
  to have all four files (`runwrap.py:4156-4164`). A third companion to recover
  one string is a poor trade against one line of awk.
* **It feeds shell control flow.** `$_warm_label` is consumed by a `case`
  pattern and a glob loop in the same script; a Python reader means spawning an
  interpreter to capture one word.
* ~~**They answer DIFFERENT questions, so they are allowed to differ.**~~
  **WITHDRAWN 2026-09-17, same day, user: *"i don't buy the reason for why it
  has to be forgiving and it's just a fucking pattern matching of names. what
  kind of shit argument is this?"*** — and that is correct. I argued the awk
  must be lax because it reads a possibly hand-edited deck. **molbuilder writes
  the keyword**, the deck fences a `user-custom` zone for edits and says *"Do
  not hand-edit it"* about the rest, and nobody renames `SystemLabel` to
  `System.Label` by accident. The laxness is not earned by that story; it is
  two regexes written on different days.

  **What survives is narrower and real: the VALUE can legitimately change.** A
  person may rename the job in the deck between `prep` and `launch`, and a
  sweep keyed to the baked-in old value moves the wrong files. *That* is why it
  re-reads. It says nothing about how the keyword is matched.

**Recorded so it is not "unified" later:** a future reader who sees only the
similarity will merge these and break the wrapper. The rule is the third bullet
— *the wrapper reads the deck AS LAUNCHED; the framework reads what it wrote* —
and it belongs in the awk's own comment (N5d).

**And its comment carries a claim that was false when written.** It says
*"This is the only reader of the label now"* — written 2026-09-06 at reader 3's
deletion, while reader 2 was live in `watch.py` and had been for months. Same
class as everything else the 2026-09-17 sweep found: a claim made AT a deletion,
about the thing being deleted, never checked against the rest of the tree.

#### Where the PYTHON reader belongs — `model/parse.md` § 1a settles it

> *"A reserved block in a molbuilder-generated script is not foreign.
> molbuilder writes it, into a file molbuilder generated... **A block belongs to
> its writer** — `script_emit` owns the emitting and the reading of what it
> emitted."*

We write `SystemLabel` and `JOB = "…"`. So reading them back is the **writer's**,
and `script_emit` already demonstrates the shape with five of them —
`_extract_header_text`, `_extract_provenance_dict`, `_extract_user_custom_inner`,
`_extract_atom_metadata_dict`, `_extract_bench_marks_dict`.

**This is also the retrospective explanation for reader 3.** It was deleted for
having zero callers; the deeper fact is that it was in `parse/` — the layer for
FOREIGN formats — reading something molbuilder itself had written. Re-creating
it there, which is what I was about to propose, would have rebuilt the same
mistake with a caller attached to excuse it.

#### The solution — an EXTENSION of two existing surfaces, no new module

| step | change | why it is not a patch |
|---|---|---|
| **N5a** | ~~give each emitter a label reader~~ — **WITHDRAWN: that would be a FIFTH reader.** Instead: **move the one correct fdf reader out of `transport/preflight.py`**, where it is a leftover of the deleted cross-deck comparison, to where reading an engine's format belongs | `_parse_fdf` + `_norm` is the ONLY reader in the tree that implements fdf's real keyword rule (lower-case AND strip `.`/`-`/`_`). `parse/contract.py` already reaches across a package boundary for it with an apologetic function-level import; `watch.py` and `runwrap._parse_fdf_n_atoms` each hand-rolled a narrower one |
| **N5b** | `watch.py` deletes `_FDF_SYSTEM_LABEL_RE` / `_PY_JOB_NAME_RE` and asks the moved reader; the label is then `scalars["systemlabel"]` — **no new regex anywhere** | four fdf readers become one, and the borrow in `parse/contract.py` stops crossing a package boundary |
| **N5c** | `parse/dirs/job.py` asks N5a for the label, then `runfiles.parse`; **`identity.parse_stage_token` is DELETED**, its `materialize.py` callers moving to `runfiles.parse` + a token splitter (the inverse of `identity.stage_token`, taking a TOKEN not a filename) | kills the second FILENAME reader, which is N5 ② — and ③ falls out, because the phantom rung came from parsing with no label |
| **N5d** | **apply G7 to the label — BUT IT IS NOT TWO DECLARATIONS.** ⚠ *I described this as trivial before running the review; the review says otherwise.* `read_by = ["wrapper"]` **documents** a dependency, it does not carry a value: G7 was reached *"by carrying the answer rather than by importing the catalogue into the wrapper"* — `resolve` puts `use_gpu` on `Resources`, which already travels. Measured: `render_wrappers` / `write_run_wrapper` receive `(script_path, resources, env, emit_sbatch, project_dir, machine_record)` and **not** the unsuffixed label; A8 forbids adding a loose kwarg (it exists because eleven were removed). So the label reaches the wrapper only by **a new `Resources` field** — and `Resources` is identity-free today, carrying allocation, retry, notify and `program`. **That is a design decision, not a mechanical application**, and it waits for a yes | `gpu.md` G7 + `architecture.md` A8. The alternative — prep reads the deck once with the consolidated parser and bakes a literal — needs no new field and no plumbing, and is what N5f makes possible |
| **N5f** | the remaining deck readers consolidate onto the one correct parser — **`watch.py`'s pair and `_parse_fdf_n_atoms` DONE 2026-09-17**; `_fdf_requests_gpu` **STOPPED, see below** | **eight readers of deck content measured** — four awk in the wrapper (label ×2, GPU flag, a `%block` line counter), four Python — and only `_parse_fdf` + `_norm` implements fdf's real keyword rule |

> **N5f stopped one reader short, on a disagreement worth settling before it
> is buried** *(2026-09-17)*. `_fdf_requests_gpu` was next, and routing it
> through `_parse_fdf` would have **changed its answer**: the two readers
> disagree on what a REPEATED keyword means.
>
> | reader | repeated keyword | authority it cites |
> |---|---|---|
> | `parse/fdf.py::_parse_fdf` | **first** occurrence wins (`scalars.setdefault`) | none stated |
> | `runwrap.py::_fdf_requests_gpu` | **last** occurrence wins | *"matches SIESTA's `read_options.F90` semantics"* |
>
> Only one can be right, and the consequence is not cosmetic: a deck that
> states `Diag.ELPA.GPU` twice routes to a different environment depending on
> which reader answers. **Settling it needs SIESTA's own fdf source**, which is
> not available in this checkout — `labeleq` is called in the copy on hand but
> defined elsewhere. Recorded rather than guessed; `_fdf_requests_gpu` keeps
> its own scan until then, which is the honest state and not an oversight.
>
> *(It is also a second instance of § 6a's shape: two readers of one format,
> each stating a rule, neither checked against the other until something made
> them meet.)*
| **N5e** | `RunDirResult` gains `label`, reader named per § 5.0 | **§ 5c's, recorded not built** — the directory-level question is the door's, and N5c's call is the same one the door will make |

**What this deliberately does NOT do:** build a `label_of(directory)` door now.
That is a question *about a directory*, which § 5 says goes through
`JobDirParser`, and building it anywhere else is how a seventh consumer with its
own answer appears — the exact shape § 5l was retired for.

---

### 5l.d Reading back a deck we just wrote *(2026-09-17)*

**The question that closed this thread, and it was not the one I kept
answering.** I spent several rounds on *which reader wins when a keyword
appears twice*. The user's actual question was **why anything re-reads the deck
at all** — every value in it was known when it was written, and the layers had
already resolved each one to a single value with provenance. Reading it back to
recover what we were holding is the defect; who wins a tie is noise.

**Two different things, and I had been treating them as one:**

| | reads | verdict |
|---|---|---|
| a **cited** deck — someone's finished run from last week | `parse/contract.py`, `transport/compose.py`, `transport/citation_defaults.py` | **legitimate.** That file is the only record of what they ran; there is no description to ask |
| a deck **we just wrote, in this operation** | `runwrap._deck_label`, `_parse_fdf_n_atoms`, `_fdf_requests_gpu` | **re-derivation.** `gpu.md` G7: *"the value travels; the deck is not re-read for it"* |

**`_deck_label` is DONE, and it was mine, added the same day.** Told to bake the
label in as a literal, I implemented it by opening the deck at prep and reading
`SystemLabel` back out — the same defect as the awk it replaced, one step
earlier. The value was `task.label` the whole time. It now travels:
`prep` → `write_run_wrapper(label=…)` → `render_wrappers` → `render_run_wrapper`
→ `_cold_restart_block`, and `runwrap` reads no label from any deck.

> **A8 never forbade that parameter, which is why the first attempt went the
> wrong way.** I read A8 as *no new arguments* and designed around it. A8 says a
> door taking one of § 3's objects *"may not also name that object's FIELDS"* —
> it forbids destructuring `Resources`, not adding a parameter. `label` is not
> one of its fields. **The rule I invented to avoid was not the rule.**

**Still open, same class, each needing its own value threaded:**

* `_parse_fdf_n_atoms` reads `NumberOfAtoms` — known as `len(struct.elements)`;
* `_fdf_requests_gpu` reads `Diag.ELPA.GPU` — known as `resources.use_gpu`, and
  G7 already made that the preferred path; this is the documented fallback for
  *"a deck someone points at"*, which has no allocation. **Whether that fallback
  should exist at all is the open question** — not, as I claimed three times,
  which occurrence it picks.

**And the keyword rule is written twice more:** `siesta/layout.py::check_rules`
spells `lower().replace(".","")…` inline, and it is the same rule as
`parse/fdf._norm`. One of them should call the other.

---

### 5l.b The pattern this is the second instance of *(2026-09-17)*

§ 5l and `model/parse.md` § 5's `JobDirParser` are mirror images, and both were
surfaced by the same sweep on the same day:

| | `ref.py` | `JobDirParser` |
|---|---|---|
| specified | § 5l | `model/parse.md` § 5 |
| built | **yes — 444 lines** | no |
| adopted | **no — zero production callers** | no |
| named by a contract | **no** — only `architecture.md`'s L1 index, as a bare `ref` in a list | yes |
| outcome | **retired and deleted 2026-09-17** | still owed — § 5c |

**A parked deliverable is not free, and the two failure modes are different.**
`JobDirParser` unbuilt costs a gap a reader can see: `parse_dir()` raises, and
the Results picker guesses from filenames because there is no door to ask
(§ 5p.3p.3). `ref.py` built-and-unadopted cost something worse — it read as
*done* from the L1 index while answering nothing, so the four APIs it was meant
to replace kept growing beside it. **The lesson is the sequencing rule § 5l.6
already stated and did not follow: a step is "each step separately GREEN", and a
module with no caller is not a green step — it is an unmerged branch living in
`main`.**

---

## 5m. The test screen — what the audit found, sequenced *(2026-09-09)*

**§ 5k is the paths framework (§ 5l's standard on top of it was retired
2026-09-17); this is the suite.** Two audits ran under
`process/testing.md` § 3b — a four-partition gate over ~3,000 test functions
(2026-09-08) and a header pass over the ten most-undocumented files
(2026-09-09). Their EVIDENCE lives in two records and stays there:

- [`process/test-audit-findings.md`](?doc=process/test-audit-findings.md) — the
  numbered defect ledger (§ 0a), the reproductions, and the unapplied verdicts.
- [`science/test-design-findings.md`](?doc=science/test-design-findings.md) —
  the protected class: where a science test would pass for a physically wrong
  reason.

**This section is the WORK.** It exists because a record with no plan row is a
record nobody executes — which is how `science/test-design-findings.md` shipped
on 2026-09-08 linked from nothing, failing two of the repo's own doc lints, and
was one compaction from being lost.

### 5m.1 The rule this section is under

> **The default is REMOVAL, and the burden of proof is on keeping** (user,
> 2026-09-08). The one exception is scientific validation: protected, and only
> its DESIGN is open.

And the number that governs everything below: **subsumption reasoning was
measured wrong on 9 of 19 candidates -- nearly half** (`test-audit-findings.md` § 4;
the 1-in-5 and 1-in-3 figures came from a harness blind to the commonest
validator defect). A verdict is a
lead. `tools/verify_subsumption.py` is what turns one into a decision.

### 5m.2 The rows

**They are `TS`, not `T`, and that is not cosmetic.** The plan already had a
`T1` (a live-poll defect, § 2) and a `T4` (under `P1`), and this table was
written as `T1`–`T9` on 2026-09-09 — so an edit keyed on `| **T1** |` hit the
wrong row and destroyed it. Caught and restored the same hour. The plan has had
this collision before (`N3`/`N4`/`N5` mean one thing in § 2 and another in
§ 5l.6 before it was retired); a section that numbers its own rows must
prefix them.


| step | what | done when |
|---|---|---|
| **TS6** | **`#66`'s four redesigns — THREE DONE 2026-09-09, one BLOCKED.** The SIESTA-version test now reads the pin from `recipes.py` instead of retyping it (the recipe moved to 5.5.0 in a mutant and it failed — the exact drift the old one could not see). `task.py` stays engine-agnostic is an IMPORT check, not four literal spellings (all four spellings caught; a comment correctly does not fire — **and my first version had the same bug it was fixing**, caught by mutation). The SCF-history branch is now actually reached. **Blocked: the `setInterval` pin.** Its correct replacement drives the module through `tests/_node_esm.py`, and that harness skips — **#79**, 717 tests never execute. Replacing it would swap a weak test that RUNS for a correct test that SKIPS, which is an environment decision, not a test one |
| **TS15** | **`#80` — the X3DNA probe answers *installed* for a pack that cannot run.** **The instance is FIXED 2026-09-09 by removing the dependency**: `x3dna_utils` is 3DNA's only interpreted tool and the one sub-command we used from it (`cp_std BDNA`) is a pure file copy, so `_copy_standard_bases` does it directly and no environment needs ruby. It was USER-FACING — the Modify tab's comma-separated two-strand input is the only path through `rebuild`, and it died with `exit 127` while `ds,` kept working via `fiber`. **Still open: the probe itself**, which answers from file existence. Same question as `TS10`, opposite symptom — node's absence is honest, this one was not | the probe separates *present* from *usable* |
| **TS10** | **`#79` — 717 tests never execute.** 43 files gate on `node`, which is not installed, not in the env and not declared. Measured 2026-09-09: 67 passed, 717 skipped. **Needs a decision, not code**: node in the env, node in CI, or a documented developer prerequisite — and until one is taken, every § 3a source pin those files could replace stays put | the harness runs, and `TS6`'s last redesign is unblocked |
| **TS8** | **The science redesigns — FIVE closed 2026-09-09 under one question: WHO OWNS THE NUMBER, and what does this layer DECIDE?** Two slab tolerances re-aimed (they asserted ASE's crystallography; `add_slab` only cuts a window, so both take their expectation from ASE now and the mechanism is covered structurally over 144 cases). The `homo_idx` rule moved the other way — the record asked the READ side, which carries the index but not the occupations, so it went to the EMITTER where it is ours, and became callable because it has a branch. Then two from § 3: a rotation is now asserted RIGID and PROPER over 3 axes × 5 angles (`R(0)=I` held for a wrong axis, a flipped sign and a radians error alike — 4 mutants killed), and the junction fixture's frozen set is DERIVED FROM z rather than counted, so the audit's own scenario (freeze the interior, same count) now fails. **Remaining: the rest of §§ 3–6** | each fails for the reason it names, and asserts nothing we did not write |
| **TS9** | **The remaining 1,627 undocumented tests** (of 6,411 — the pass took the suite from 67.3% to 74.6%). Writing the header IS the audit; it is what made T1–T6 findable | every test states its goal and its contract, and the obsolete ones have surfaced |

### 5m.3 What is deliberately NOT a row

**Re-deriving the ~90 lost verdicts.** Most auditors terminated on a session
rate limit and their output was cleared with the session directory. Re-deriving
a subsumption verdict that then cannot be applied without a mutant costs more
than running the mutant on a verdict already in hand — so TS7 spends the budget
on the 40 that survive, and the rest are declared lost rather than quietly
carried as if they were pending.

**And the cuts already applied are not revisited.** They are source pins and
cannot-fails, where the evidence is a line of shipped code rather than an
argument about reachability, and their reasoning is in the commit messages
`test-audit-findings.md` § 0 lists.

---

## 5n. JupyterNB — settings as DATA, and the hand-built residue *(2026-09-15)*

**Origin.** The user, reading `molbuilder/jupyter.py`: *"why is jupyter.py not
following a data-driven design but rather handcrafted jibberish of code?"* —
then *"use .json or .jsonl or .toml to help clean this up. this is a
systematic design, not some hacking"* — then *"make sure that you don't have
other hackish code in the design, put this all in a consolidated plan."*

**The judgement is right and it is narrower than the words.** Parts of this
feature are already declarative (`_LAB_OVERRIDES`, `prepare_lab_home`'s
option→path map, `envs/recipes.py::_JUPYTER`). What earns the word is that
**the same kind of fact is expressed three different ways inside one
function**, that **40 lines of real Python live inside a string literal**, and
that **nothing has ever forced a seam on any of it** — the feature has no
test. § 5n.5 lists what is NOT residue, so the sweep can be checked rather
than believed.

**Row: W23.** This section is the contract; no code until it is agreed.

---

### 5n.1 The data file — format, location, and the admission rule

**TOML.** The project already answers this. Authored **rule tables** are TOML,
read with stdlib `tomllib` and gated by `persist.check_schema`:
`siesta/warm-files.toml`, `pyscf/warm-files.toml`,
`data/catalogue.template.toml`. JSON here is reserved for **fundamental
numeric tables** nobody annotates (`data/contact_distance.json`,
`data/fcc_lattice.json`). JSONL is a record stream, and this is a table. And
every row in this table exists for a reason that has to travel with the row —
TOML takes comments; JSON does not.

**`molbuilder/data/jupyter.toml`**, covered by the existing `data/*.toml`
package-data pattern. No packaging change, and this repository already
carries the incident where a `.toml` shipped in no wheel (`pyproject.toml`,
the comment above `data/*.toml`).

**Schema-stamped**, exactly as `warm-files.toml` is:
`schema = "molbuilder/jupyter@1"`, gated by `persist.check_schema`. An unknown
top-level section is refused **by naming the sections that exist** —
`warmfiles`' own refusal style, and the reason it is worth copying is that a
typo in a settings file otherwise disables a setting silently.

**THE ADMISSION RULE — what stops the file becoming a dumping ground.**

> A row belongs in `jupyter.toml` **if and only if its value is knowable
> before the process starts.** Anything that needs a runtime fact stays in
> code, and is emitted through the same writer.

That is a testable line, not taste. It cuts the current settings as follows:

| in the file | stays in code, and why |
|---|---|
| `ServerApp.port_retries = 0` | `ServerApp.ip` — a CLI argument |
| `MappingKernelManager.cull_idle_timeout = 1800` | `ServerApp.port` — derived, `jupyter_port(serve_port)` |
| `MappingKernelManager.cull_connected = false` | `ServerApp.token` — generated per start; a credential gets no home in the repo |
| `ServerApp.shutdown_no_activity_timeout = 3600` | `ServerApp.root_dir` — the projects root |
| `[lab.home]` — the three `LabApp.*_dir` options → their subdirectory names | `ServerApp.tornado_settings` — the `frame-ancestors` grant, **derived** from `serve_port_of(port)`. A security header stays a computation with its reasoning beside it |
| `[lab.overrides]` — all four plugin sections, verbatim from `_LAB_OVERRIDES` | `certfile` / `keyfile` — present only when `serve` has TLS |

**Lifecycle constants do NOT move.** `_PR_SET_PDEATHSIG = 1` is a kernel ABI
number; `_MARKER` is an identity string; `_STOP_GRACE_S = 4.0` is coupled to
`serve_daemon.stop_by_pidfile`'s `grace_s = 5.0` and splitting a coupled pair
across a data file and a module is worse than leaving both in code. *(That
coupling is asserted by a comment and by nothing else — see J12.)*

### 5n.2 The generated Python is a FILE, not a string

`_SERVER_CONFIG` becomes **`molbuilder/data/jupyter_server_config.py`**, a real
`.py` file copied verbatim at every start. It does **not** become TOML:
putting Python in a data file is the same defect in a worse place.

**Why not `inspect.getsource`**, which is this project's pattern for generated
code (`trajectory_log/emitter.py`, `pyscf/input.py`, `pyscf/vibration_emitters.py`
×2): that needs the class importable, and `NoCheckpoints` must subclass
`jupyter_server`'s `AsyncCheckpoints` **at class-definition time**. The
shepherd runs in the HOST env, which does not have `jupyter_server` and must
not need it. A file under `data/` is never imported, so it may name
`jupyter_server` freely, while pyflakes, an editor and `ast.parse` all still
read it as Python. One packaging addition: `data/*.py`.

### 5n.3 One emitter

`notebook_argv` stops holding a list literal. It builds a single
`{trait: value}` map — the file's rows merged with the runtime rows — and
emits `--{k}={v}`. The three mechanisms collapse to one, and a dropped row
like `port_retries` becomes **visibly missing from a table** instead of
invisibly absent from a list.

### 5n.4 The sweep — the other hand-built residue

Twelve found 2026-09-15 by reading the whole feature; **J13, J14 and J15 are the user's own, added the same day**.  Read: `jupyter.py`,
`web/blueprints/jupyter.py`, `web/static/jupyternb/`, `web/templates/jupyternb.html`,
the notebook half of `serve_daemon.py`, the `jupyter` CLI group, `config_dir`'s
four doors, and `envs/recipes.py::_JUPYTER`.

| # | what | why it is residue |
|---|---|---|
| **J1** | **Three mechanisms for one traitlets setting**, inside `notebook_argv`: 11 hand-written f-strings in a list literal, 2 of them interpolating module constants, and 4 more passed as a dict through a `--{k}={v}` loop **twelve lines below**. **15 settings, one shape** -- `--<Class>.<trait>=<value>` -- said three ways | § 5n.1 + § 5n.3 |
| **J2** | **`_SERVER_CONFIG` — 40 lines of real Python in a string literal.** No linter, no import, no test reaches it. **The cost is measured:** it was deleted by accident on 2026-09-14 by an edit that removed everything between two dead functions, and that shipped a `NameError` on every notebook start. `pyflakes` caught the sibling (`_LAB_OVERRIDES`, a name that IS referenced) and could say nothing about this one | § 5n.2 |
| **J3** | **The control routes are gated twice, on two different facts.** `@bp.post("/api/jupyter/start")` is registered inside `if os.environ.get(SUPERVISED_ENV) == "1":` **at import**, and the module's own docstring then explains that this flag is *not* the real precondition — `serve foreground` sets it while writing no pidfile — so a second, per-request gate `_supervised()` was added on 2026-09-14. Two gates, one fact. The import-time one also makes the app's **URL map depend on the environment `create_app()` happened to run in**, which is why no test can reach these routes | **Fix:** register always; the handler answers 404 when `_supervised()` is false. That is the behaviour § 6 of `jupyter.md` already specifies (*"404 rather than 403"*), with one gate, evaluated when the answer is knowable |
| **J4** | **Two hand-written 403 messages for one refusal.** `start` returns a three-line sentence naming `molbuilder.json`'s `admin` section; `stop` returns `"admin auth required"`. Same gate, same condition, two answers — and the short one tells a person nothing about what to do | One `_refuse()`, one sentence |
| **J5** | **The status payload has no author.** `jupyter.status()` returns 8 keys, the blueprint bolts on 5 more (`may_control`, `supervised`, `env_name`, `env_installed`, `install_command`), and `index.js` reads all 13 plus `ok`. Three files define one shape and no document states it — the two-authored-homes smell `engines/template.md` § 6.0 has a rule for | State the response shape once, in `jupyter.md` § 5; assemble it in one place |
| **J6** | **The tab's waiting is five ad-hoc timers, not a state machine.** `pollTimer`; `START_POLLS`/`startPollsLeft` at 1200 ms; `WEDGE_POLLS`/`wedgePolls` at 1500 ms; `ROOT_WAIT_MS`/`rootGaveUp` on a raw `setTimeout`; and three bare `pollSoon(600 \| 1500 \| 30000)`. Each arrived after its own incident — the comments are the record — and **each has its own clearing rule scattered through `render`**: `startPollsLeft` is cleared in one branch, `wedgePolls` at the top of the function, `rootGaveUp` never | **Fix:** one table of states, each row `{when, message, framed, pollMs, budget, onExhausted}`; `render` selects a row and one function applies it. **A JS object literal, not a data file** — the strings are UI copy and belong beside the page (`ui-contract.md`), and shipping TOML to a browser buys nothing |
| **J7** | **`jupyter.md` § 4 says "exactly four states". `render` carries **13** `return` statements and `refresh`'s `catch` adds a fourteenth outcome.** Four of the undocumented ones carry their own time budget AND their own button — wedged, *"asked for a notebook and none started"*, waiting for the projects root, and the fetch failure. **The states with the most behaviour are the ones the contract does not name** | Fixed with J6: the table in the code and the table in § 4 become the same list |
| **J8** | **A third hand-built unverified TLS context.** `jupyter.py` collapsed two copies into `_unverified_ctx()` on 2026-09-14, with a docstring saying *"One home, because this is a security knob"*. `cli.py:2525-2527` still builds the same three lines for `serve status` | One home means one home. `serve`'s, not the notebook's — but the same knob, and the claim is already written |
| **J9** | **`--port` hand-written six times across the `jupyter` verbs, while `_serve_flags` sits ~350 lines above** saying *"one decorator, so the two verbs cannot drift apart about what a server accepts"*. **The six have already drifted:** two carry help text that disagrees (*"the SERVE port -- the notebook's own is that plus one"* vs *"the SERVE port this notebook belongs to"*) and three carry none | A `_port_flag` the six share |
| **J10** | **THE FEATURE HAS NO TEST.** 1,610 lines across its own five files, plus the notebook half of `serve_daemon` and six CLI verbs. Nothing under `tests/` imports `molbuilder.jupyter` or touches `/api/jupyter/*`; the only mentions are an env name in `test_diagnostics` / `test_envs_install` and a layer entry in `test_layering`. **Every one of the nine defects fixed on 2026-09-14 was found by eye or by running the live server, and J2 shipped.** This is the root of the whole row: nothing has ever forced a seam on this code | § 5n.7 |
| **J11** | **`config_dir.jupyter_lab_home`'s docstring says "Two directories under it"**; `prepare_lab_home` creates **three** (`settings`, `user-settings`, `workspaces`) and writes a config file beside them. The door's own contract is behind its caller | One sentence |
| **J12** | **`_STOP_GRACE_S < serve_daemon`'s `grace_s` is asserted by a comment and by nothing else.** The two were both `5.0`, and reconciliation then reported *"(forced)"* for every perfectly polite stop. It was fixed by changing one number; nothing stops the other moving back | One assertion, in the test set below |
| **J13** | **SWITCHING TABS LOSES THE OPEN NOTEBOOK, and the kernel it belongs to is still running.** *(user, 2026-09-15: "when we switch tabs (without quitting server), currently the jupyternb tab reinit and does not open the original tab before the switching even though the kernel etc is still running … we would like to have this persistent when switching tab, but do not need this when server get shutdown and restarted.")*  Leaving `/jupyternb` destroys the iframe, so coming back always re-points it — that part is a page navigation and cannot be avoided in the tab. What decides whether Lab comes back where you left it is its **workspace** (which documents were open), and `labUrl` sends `?reset` on **every** load (`index.js:137`), which throws it away. **This supersedes a decision taken with the user on 2026-09-14** (`jupyter.md` § 4.1, *"it remembers no LAYOUT"*), whose reason was that a restore argued with the folder the projects sidebar had selected — a reason that has since weakened on its own, because the sidebar is no longer SHOWN on this tab and cannot be changed from it | **Fix — server-side, one home (preferred): `prepare_lab_home()` DELETES `workspaces/` at every shepherd start, and `?reset` is dropped from the URL entirely.** A new server then has no workspace to restore (clean, as asked); a tab switch or a page reload restores the one Lab has been saving (persistent, as asked); and no browser state is involved, so there is nothing to keep in sync. **Two things to MEASURE before the shape is fixed:** whether `/lab/tree/<path>` fights a restored workspace or merely moves the file browser, and whether Lab writes the workspace eagerly enough that a switch seconds after opening a notebook is caught. **Fallback if the measurement says no:** compare `st.token` to one remembered in `localStorage` and send `?reset` only when it differs — the token is regenerated by `run_shepherd` at every start, so it already IS the server-session identity. `localStorage`, not `sessionStorage`: closing a browser tab is not a server restart |
| **J14** | **`serve status` answers only about the port you guessed.** *(user, 2026-09-15: "the serve status should report all running server on different ports - should not only report only when the correct port is provided. if there are server running the pid and port is available in the log/state dir.")*  `--port` defaults to 8000, so a server on 8888 makes `molbuilder serve status` print *"not running"* and exit 3 — the one answer that is worse than no answer, because it is confidently wrong. The information is already on disk and already keyed by port: `runtime_dir()` holds `serve-<port>.pid`, one per server, and `pid_state` already says whether each is alive and really ours | **Fix:** `--port` on the two STATUS verbs becomes optional. Given, the verb behaves exactly as today (`serve status`: exit 0 answering / **4** up-not-answering / 3 absent — the codes it already returns, re-derived from the code on 2026-09-15 after this row guessed 1; `jupyter status` has only 0 and 1). Omitted, each **surveys**: every `serve-*.pid` (or `jupyter-*.pid`) in the runtime dir, verified with `pid_state`, probed for an answer, one line per live server — and, for `serve`, whether a notebook is held beside it. A miss on a named port now also names the servers that ARE up, which is the whole complaint in one line.  **Two things it must say rather than imply:** a `serve foreground` writes NO pidfile and therefore cannot appear, so an empty survey says that instead of *"nothing is running"*; and a stale file is reported stale, never signalled, which is the rule this module already keeps.  The probe (https then http, `/api/health`) is inline in `cmd_serve_status` today and becomes one helper, because the survey and the single-port path must not answer *"is it up"* two ways |
| **J15** | **TWO MOLBUILDERS ON ADJACENT PORTS COLLIDE BY CONSTRUCTION, and nothing says so.** *(user, 2026-09-15: "how about conflicts in jupyter port?")*  `jupyter_port(p) = p + 1`, so a molbuilder on 6006 wants 6007 for its notebook and a molbuilder on 6007 wants 6007 for its WEB server. Whichever starts second loses, and which one that is decides which symptom you get: the notebook refuses to start, or `serve start` fails to bind. **`--ServerApp.port_retries=0` already makes this loud rather than silent** — without it jupyter-server picks a random free port out of 50 while the runtime file, the `frame-ancestors` grant and `answering()` all keep naming `p+1`, and the tab reports "wedged" over a perfectly healthy notebook. So the remaining defect is not the crash; it is that **nothing warns when the clash is knowable in advance, and the report points at the wrong log** | **Fix, three parts.** ① **MEASURE FIRST** what jupyter-server actually does and prints at `port_retries=0` against a taken port — this row assumes the setting's documented meaning and nothing has run it. ② **Warn where it is computable**: `serve start` knows its own port and `ports_with_pidfile()` now lists every other live server, so `jupyter_port(port)` landing on one of them is a sentence printed before anything starts, and the `serve status` survey carries the same line. Not a refusal — the web server is perfectly startable, only its notebook is doomed. ③ **Name the right log**: `index.js`'s *"asked for a notebook and none started"* sends a person to the SERVE log, and a port clash is written in the NOTEBOOK log.  **NOT a change to the derivation.** `p+1` is one number to remember and the pairing has one home in both directions; moving to `p+1000` relocates the collision without removing it, and letting Jupyter choose freely makes the runtime file the authority for a port that four other places derive — which is the design `port_retries=0` was chosen against |

### 5n.5 What is NOT residue — so the sweep can be checked

* **The three lifecycle layers and their constants.** Order is the content —
  *"everything that can raise happens before the pidfile exists"* is not
  something a table can say — and each layer is measured, not argued.
* **`jupyter_port` / `serve_port_of`.** One derivation, both directions, with
  a stated security reason for the inverse existing at all.
* **`_LAB_OVERRIDES` and `prepare_lab_home`'s option→path map.** Already
  tables. They move into the file; their shape does not change.
* **`envs/recipes.py::_JUPYTER`.** Already declarative, and its packages carry
  their reasons.
* **`config_dir`'s four doors.** Consistent and reasoned — including
  `jupyter_lab_home` **not** being port-keyed, which the docstring states and
  justifies (*"the defaults do not differ between servers"*).
* **`serve_daemon`'s stringly-keyed `state` dict** (`"child"`, `"hup"`,
  `"term"`, `"jupyter"`, `"nb_busy"`, `"nb_pending"`). It is L1, it predates
  this feature, and reshaping it is not this row. **Out of scope, on purpose.**

### 5n.6 Order of work

Each step is separately green and separately revertible.

1. `molbuilder/data/jupyter.toml` + the loader (schema gate, refusal by name).
2. `notebook_argv` → one emitter; `_LAB_OVERRIDES` and the lab-home map read
   from the file. *(**Done 2026-09-15, and the line-count claim here was
   wrong**: `jupyter.py` went 681 → 744. The two literals were 78 lines and
   the loader with its refusals is ~110. The win is not fewer lines — it is
   130 lines of data in two files that a linter, an editor and `tomllib` can
   all read, and one emitter instead of three mechanisms.)*
3. `molbuilder/data/jupyter_server_config.py` + `data/*.py` in `pyproject.toml`.
4. **J3 + J4** — one gate, one refusal. (This is what makes the blueprint
   testable, so it comes before the tests.)
5. **J9 + J8** — the shared port flag; the third TLS context.
6. **J6 + J7** — the tab's state table, and § 4 rewritten from it.
7. **J13** — the workspace survives a tab switch and dies with the server.
   *(Done 2026-09-15. Measurement (b) answered — the workspace file carries a
   live mtime, so Lab writes it continuously and `?reset` was what discarded
   it. Measurement (a) was **not** made and is no longer needed: the shape
   chosen never sends a tree path and a restore together, so Lab's precedence
   between them decides nothing. The payload is 16 keys with
   `workspace_saved`, which step 10 documents.)*
   **Measure first** (does `/lab/tree/<path>` fight a restore; is the
   workspace written eagerly enough), then `prepare_lab_home()` wipes
   `workspaces/`, `?reset` comes out of `labUrl`, and `jupyter.md` § 4.1's
   *"it remembers no LAYOUT"* paragraph is rewritten to say what replaced it
   and why — a decision taken with the user is superseded in writing, never
   silently flipped.
8. **J14** — the two status verbs survey every server when no port is named.
   *(Done 2026-09-15. The backticks in this line were eaten by an unquoted
   shell heredoc when § 5n was written; restored with the J15 entry.)*
9. **J15** — measure the port clash, then warn where it is knowable.
   *(Done 2026-09-15. Measured: jupyter-server exits 1 with "the Jupyter
   server could not be started because port 6007 is not available", so the
   assumption held. `jupyter.port_clash()` is the one home; `serve start`
   warns before detaching, the survey warns per server, and the status
   payload carries it so the tab can name the cause. The payload is now 15
   keys, which step 10 documents.)*
10. **J5 + J11** — the response shape stated once; the stale docstring.
    *(Done 2026-09-15. `jupyter.md` § 5.1 is the 16-key table and § 5.2 the
    two control routes' four status codes; the blueprint builds the payload
    in one expression instead of six `st[...] =` mutations.)*
11. § 5n.7's tests, and **J12**. *(Done 2026-09-15 —
    `tests/test_jupyter_contract.py`, 7 tests / 11 cases, all seven
    mutation-tested. **The suite grows by seven**, and § 5n.7's "does not
    grow" line was wrong: it assumed hand-written assertions to remove, and
    there were none — the coverage being replaced is zero, which is J10.)*

### 5n.8 The fresh-eyes review of the review *(2026-09-15, same day)*

Three reviewers over the eleven commits, every claim re-verified here before
acting. **Nine real defects in work that was hours old**, which is the
argument for the pass rather than against it. All fixed in one commit; the
numbers below were re-derived, not taken.

| what | why it mattered |
|---|---|
| **`_forget_workspace`'s guard let `..` through** | `home / ".."` is a real directory whose `.parent` IS `home`, so the check passed and the loop would have emptied the state directory — logs, reports, run/. `../../../escape` was refused and `..` was not, and the comment claimed both. **Destructive.** Now compares resolved paths |
| **the workspace was shared by every server** | `jupyter_lab_home()` is not port-keyed, and J13 put session state in it. Starting B's notebook emptied A's layout → A's next tab switch lost the notebook whose kernel was still running: **J13's own bug, hours after J13**. Now `workspaces/<serve port>/`; `settings/` and `user-settings/` stay shared |
| **the survey stopped recording a wedge** | before J14 the no-argument `serve status` took the single-port path and appended the detection to the log. Making it a survey dropped that for the **common** invocation, against the rule `_note_wedge`'s own docstring cites |
| **`port_clash` asked one of the two directions it names** | `serve start --port 6007` beside a live notebook on 6007 warned nothing and failed to bind after `daemonize()`, with the terminal gone. Both directions now, and the first no longer globs the runtime directory to answer a single-port question |
| **`run_shepherd`'s stated invariant was false** | *"everything that can raise happens before the pidfile exists"* — `projects_root()` and `load_rules()` both sat below the write. Hoisting `prepare_lab_home` had fixed the measured case and left the claim wrong |
| **state 3 blocked on a folder it was about to discard** | with a workspace to restore the tree path is unused, yet `opening` still hid the frame for its full 8 s budget — on the headline J13 case |
| **`_serve_flags` still hand-wrote `--port`** | the eighth copy, in the decorator whose docstring forbids drift, and the only one with no help text. **The step-5 commit message claimed this was covered.** It was not |
| **`web/app.py`'s comment described the design J3 deleted** | and cited the reference the blueprint corrected — the restatement a reader of `create_app` actually sees |
| **test 4 asserted the class, not the wiring** | deleting one of the two `checkpoints_class` assignments — the plausible "looks redundant" edit — left it green while `.ipynb_checkpoints/` came back |

Smaller, same pass: three pure forwarders inlined (`_unverified_ctx`,
`_port_clash`, `_notebook_is_up`); `whenRootKnown`'s unreachable door check
removed; the last two bare `pollSoon` intervals replaced by the table's own;
the dead `"port"` key dropped from the runtime file and its docstring
corrected (it named the dead key and hid `"base"`, which IS read);
`_notebook_action`'s docstring stopped claiming a queue that cannot lose an
action, because a one-bytecode window exists and no arrangement of Python
statements closes it. Documents: nine corrections in `jupyter.md`, the 409 in
`web-api.md`, and `config_dir`'s not-port-keyed reasoning, which J13 outgrew.

**Left on the table, deliberately** — both real, both predating this
programme, both in code it did not open:

| # | area | item | state |
|---|---|---|---|
| **J16** | ops | **Three copies of "read a pidfile"** — `jupyter.read_pid`, `serve_daemon.read_pid`, and the same four lines inside `serve_daemon.stop_by_pidfile`. Exactly the shape J8 collapsed for the TLS context and `pid_state` collapsed for verification, with the *read* left at three | ✅ **done 2026-09-15** — `serve_daemon.read_pidfile(path)` is the one reader; `read_pid` is it at the serve address, `jupyter.read_pid` it at the notebook's, and `stop_by_pidfile` calls it. Every edge re-checked against the three it replaced (absent · empty · garbage · whitespace · a directory) |
| **J17** | ops | **`jupyter.py`'s `pid_path` / `log_path` / `runtime_path`** are one-line pass-throughs to `config_dir` — the pattern `serve_daemon` deleted as K-D3, whose comment says so in as many words. Worse, they RENAME: the notebook log is `jupyter_log` in one module and `log_path` in another. `pid_path` and `runtime_path` have no caller outside `jupyter.py` | ✅ **done 2026-09-15** — all three deleted, callers ask `config_dir`. The notebook log has ONE name now, so grepping `jupyter_log` finds every reader |

### 5n.7 Tests — the set, deliberately small

`feedback_tests_earn_their_place` applies: this must not become a test per
setting. Eight, and each one fails on a real mutation:

1. The schema gate refuses a wrong stamp.
2. An unknown top-level section is refused **by naming the sections that exist**.
3. Every `[server]` row reaches the argv — the assertion a dropped
   `port_retries` trips.
4. `data/jupyter_server_config.py` parses (`ast.parse`) and defines
   `NoCheckpoints` — the assertion J2's accident would have tripped.
5. `/api/jupyter/start` answers **404** with no supervisor and **403** to a
   non-admin with one — J3 and J4 in one test, and impossible to write today.
6. `_STOP_GRACE_S < serve_daemon`'s grace — J12.
7. `prepare_lab_home()` leaves no workspace behind — write a file into
   `workspaces/`, call it, assert the file is gone while `user-settings/`
   is untouched. That is J13's whole contract on the server side, and the
   mutation it catches (wiping the wrong directory) would silently discard
   a person's Lab settings.
8. **one server's start keeps another server's layout** -- added by the
   2026-09-15 review, which found the workspace sharing a directory across
   servers and re-creating J13's bug.

*(Written 2026-09-15 and the sentence here was wrong: there were no
hand-written assertions to remove, because there was no coverage at all.
**The suite grows by seven.** That is J10's whole point, and the usual rule —
unifying an API must REDUCE the count — does not apply to a feature whose
count is zero.)*

---

## 5o. Transport — the parameter surface is inert, measured against the binary *(2026-09-15)*

*(User: "check vigorously against actual transiesta manual and scientifically
how this computational process should be conducted. find the missing gap in UI
and parameter setting … and any inconsistency or potential issues. we need a
full solution for this particular task.")*

### 5o.0 How this was checked — the binary, not a memory of the manual

`molbuilder-siesta` ships **SIESTA 5.4.2** (`strings` on the binary). SIESTA's
fdf keywords are compiled into `siesta` and `tbtrans` as literal strings by
the `fdf_get` call sites, so **whether this installation can read a keyword is
measurable**, not a matter of recollection. Every claim below is a count taken
from `~/miniconda3/envs/molbuilder-siesta/bin/{siesta,tbtrans}`.

The method's one limit, stated: a label assembled at runtime from a prefix
does not appear whole — which is why `TS.ChemPots` counts 0 while `ChemPots`
and `.ChemPot.` are present, and the chempot blocks are *fine*. Each negative
below was therefore re-checked for the bare stem in **any** spelling.

### 5o.1 What is SOUND — so this review can be audited, not just believed

* **The five-stage ladder is the standard TranSIESTA recipe.** seed →
  electrode_L → electrode_R → device → transmission, with the electrodes run
  as bulk single-points and the device solved by NEGF. That is the workflow
  the method requires; nothing about the shape is wrong.
* **The electrode → device hand-off is correct and was measured live.** The
  electrode deck writes `TS.HS.Save true` **and** `SaveHS true`
  (`wizard.py:287`), a preflight refuses an electrode deck that would not
  write its `.TSHS` (`preflight.py:315`), and the device deck points tbtrans
  at the converged Hamiltonian with `TBT.HS <label>.TS.HSX` — that line
  carries a comment recording a live 5.4.2 measurement, and `TBT.HS` is in
  the binary.
* **Every per-electrode block key is the right 4.1+/5.x spelling:** `HS`,
  `chem-pot`, `used-atoms`, `elec-pos begin|end`, `bloch`,
  `semi-inf-direction`, inside `%block TS.Elec.<name>`, with
  `%block TS.Elecs`, `%block TS.ChemPots` / `TS.ChemPot.<name>`,
  `TS.Atoms.Buffer` and `TS.Voltage` — all present in `siesta`.
* **The consistency contract is the right physics.** Electrodes and device
  must share basis, XC and mesh or the lead self-energy cannot attach
  seamlessly; § 5 makes that an invariant and the contract fields are sealed
  at both doors. That is the single most important scientific rule in the
  workflow and it is correctly enforced.

### 5o.2 THE FINDING: ten of the twelve fields the tab renders cannot affect the run

| field | writes | in SIESTA 5.4.2? | effect |
|---|---|---|---|
| `transmission_emin_ev` | `TS.TBT.Emin` | **`Emin`: 0 occurrences in `tbtrans`, any spelling** | none |
| `transmission_emax_ev` | `TS.TBT.Emax` | **`Emax`: 0** | none |
| `transmission_n_points` | `TS.TBT.NumE` | **`NumE`: 0** (only the words "numeric") | none |
| `transmission_relative_to_ef` | `TS.TBT.Erange.RelToEF` | **`Erange`: 0 · `RelToEF`: 0** | none |
| `contour_n_circle` | *nothing* | — | **nothing** — it reached only the Methods paragraph (`transiesta.py:1015`), deleted 2026-09-17 with `methods_fragment`; see E12 |
| `contour_n_real` | *nothing* | — | **no consumer anywhere** |
| `contour_e_bottom_ev` | *nothing* | — | **no consumer anywhere** |
| `pyscf_functional` | *nothing* | engine not registered | none |
| `pyscf_basis` | *nothing* | engine not registered | none |
| `log_level` | `WriteVerbosity` claimed | **`WriteVerbosity`: 0 in `siesta`** | **no consumer anywhere** |
| `max_memory_mb` | runner `ulimit -v` | n/a (not fdf) | real |
| `num_threads` | `OMP_NUM_THREADS` | n/a (not fdf) | real |

**Two of twelve do anything, and both are shell knobs.** Every field that is
supposed to shape the *physics* is inert.

**Three distinct failures, not one:**

1. **Pre-4.1 keywords on a 5.4.2 binary.** `TS.TBT.*` and
   `TS.ComplexContour.NumCircle` / `NumLine` / `Emin` are SIESTA-3.x-era spellings.
   5.4.2 keeps exactly one legacy alias, `ComplexContour.NPoles`, which
   molbuilder does not use. The emitter's own header says "modern SIESTA
   4.1+/5.x NEGF syntax" — the device blocks are, the transmission window is
   not. **fdf ignores a label nobody queries**, so this is silent: the run
   completes and T(E) comes out on tbtrans's *default* energy grid, not the
   one the person asked for. A wrong answer that looks right.
2. **Fields wired to nothing.** Four have no consumer in the tree at all.
3. **A Methods paragraph that reports a setting the calculation never
   received.** `contour_n_circle` reaches only the manuscript sentence —
   *"NEGF density integration used a complex contour with N imaginary-axis
   points"* — for a deck that carries no contour keyword. That is the one
   finding here with a **publication** consequence.

*(No surviving artifact in `projects/` evidences the carbon-chain live walk
§ 8 cites, so I cannot say what that run's T(E) grid actually was — only that
these four keywords could not have set it.)*

### 5o.3 What a correct 5.4.2 run needs and the UI cannot say

Each verified PRESENT in the installed binary, so each is a real control
being left on the table:

| what | keyword | why it matters scientifically |
|---|---|---|
| the transmission energy window | `TBT.Contours` + `%block TBT.Contour.<name>` (`from … to`, `points`/`delta`, `method`) | **the actual replacement for the four dead fields** — the modern mechanism is a contour BLOCK, not scalars |
| transverse k for T(E) | `TBT.k` / `TBT.kgrid.MonkhorstPack` | T(E) needs a **denser** transverse grid than the SCF; convergence in it is the standard transport convergence study, and the tab cannot express it at all |
| broadening η | `TBT.Elecs.Eta`, `TBT.Contours.Eta` | sets the T(E) lineshape; too large smears resonances, too small makes them noise |
| tbtrans's own temperature | `TBT.ElectronicTemperature` | separate from the device SCF's; it is what enters the I–V Fermi functions |
| the SIESTA-side NEGF contour | `TS.Contours.nEq.Eta`, `TS.Contours.Eq.Pole`, `TS.Contours.nEq.Fermi.Cutoff` | the real controls the three dead contour fields were reaching for |
| what gets written | `TBT.DOS.Gf`, `TBT.DOS.A`, `TBT.DOS.Elecs`, `TBT.T.Gf`, `TBT.T.Bulk`, `TBT.T.All` | without these there is no PDOS or per-lead data on disk — which is what **W10**'s transmission inspector would have to read |
| lead treatment | `TS.Elecs.Bulk`, `bloch` (hardcoded `1 1 1`) | Bloch expansion of a minimal electrode cell is the standard cost saving; bulk-H in the lead region is a physics choice |
| spin | `TBT.Spin` | no spin-polarised transport path exists, and the junction chemistry check already detects open-shell metals |

### 5o.4 The rule that stops this recurring — and it is TESTABLE

§ 3.2's per-engine panels are the right frame, and the frame must carry this:

> **A field's `engine_key` names a keyword its engine's own binary accepts,
> and that is checked against the installed binary rather than asserted.**

The binaries are on disk in the env this project installs. A test can walk
every `engine_key` in a config dataclass, extract the fdf label, and grep the
engine's binary for its stem — skipping when the env is absent, the way
env-dependent tests already do. That converts *"the manual says this exists"*
— which is how seven dead keywords shipped — into a machine check, and it is
the only finding here that prevents the next one.

### 5o.5 Order of work

1. **W25 first, because it is a correctness bug, not a UI one.** Replace the
   four transmission scalars with the `%block TBT.Contour` the binary reads,
   keeping the person's window/points as the block's `from…to`/`points`.
   ✅ **Done 2026-09-15.** Three things came out of doing it that the
   investigation had not seen:
   * **Three tests were pinning the dead keywords**, including one whose own
     docstring describes the exact failure it was satisfied by — *"tbtrans
     falls back to its own defaults and the transmission curve … is reported
     over an interval nobody chose"*. A test written to catch this bug
     passed **because of** it.
   * **The fix's own comment resurrected the tokens.** Two of those tests
     assert a keyword by plain substring over the whole deck, so naming
     `TS.TBT.*` in an explanatory comment made them pass on prose. Caught
     within the hour; the emitter no longer spells them, and the rule is
     worth keeping: **a comment must not be able to satisfy an assertion.**
   * **`transport/record.py` holds the best evidence on the open question.**
     Its docstring pins the `AVTRANS` format *live on 5.4.2* and says E is
     relative to E_F *"when the deck said `TS.TBT.Erange.RelToEF T`"* — a
     keyword that can never have done anything, so what was observed was
     tbtrans's **default**. That points at *relative by default*, i.e. the
     retired switch was never needed rather than mis-spelled. Evidence, not
     settled: closed by reading one real `AVTRANS` beside its device `.fdf`.
2. The three contour fields become the real `TS.Contours.*` keywords, or go.
   **`contour_n_circle` leaves the Methods paragraph the same day**, whichever
   way it is settled — a sentence reporting an unset parameter is the one
   thing here that must not survive another commit.
3. `log_level` gets a real keyword or goes.
4. § 3.2's panel split (W24), with `TransportConfig` becoming TranSIESTA's.
5. The engine_key-versus-binary test, and the missing controls above added
   panel by panel with the test green at each step.

### 5o.6 The manual pass — the parameter inventory, and what it changed

*(User: "do thorough scientific investigation including transiesta's own
manual … a full list of parameter and settings … identify sources of data
user need to provide. check our framework and see where things are missing.")*

**Source.** The SIESTA **5.4.0** manual — the release note for the 5.4.2 this
project installs — pulled from the project's own GitLab and extracted
locally, plus the TBtrans reference page, cross-checked against the binaries'
compiled fdf labels. The inventory lives in `engines/transport.md` § 3.3,
where a person adding a field will look; this section records what the pass
CHANGED and what it leaves open.

**It answered § 5o's open question.** The manual states a TBtrans contour's
energy reference is **the equilibrium Fermi level by default**. So
`transport/record.py`'s live observation was right and only its attribution
was wrong: the retired `relative_to_ef` switch was **never needed**, not
mis-spelled. Closed.

**It explained the whole history.** tbtrans's own default contour is
`from -2. eV to 2. eV, delta 0.01 eV, mid-rule` — 401 points. molbuilder's
defaults are −2 eV, +2 eV, 401 points: **the same grid to the point.** The
four dead keywords therefore produced exactly the intended result for anyone
who changed nothing, and bit only somebody who changed a value. Latent, not
active — and latent by coincidence, not design. That is why a live walk
passed and why three tests could pin them unnoticed.

**One conformance defect, fixed here.** The manual: a `%block TS.Elec.<name>`
must carry `HS`, `semi-inf-dir`, `electrode-pos`, `chem-pot`. `elec-pos` sat
inside `if buffer_idx:`, so an ORDINARY junction got two electrode blocks
without it. Probably harmless — the junction is sorted so the electrodes are
the first and last atoms, which is where an omitted anchor would land — which
is precisely the problem: an undocumented default agreeing with the truth. Now
emitted unconditionally, with a test.

> **And mutation-testing that test caught a weak test of mine.** Checking
> that `elec-pos` is PRESENT passed the regression, because putting the line
> back inside the buffer branch made the `else` fire and gave the LEFT
> electrode `elec-pos end` — a wrong anchor a presence-check cannot see. The
> test now pins the PAIRING (z-min anchored by its first atom, z-max by its
> last) and catches all three mutations: the line moved, the line deleted,
> the anchors swapped.

#### What is missing, in the order it should be built

> ### ✅ **SEVEN OF THE NINE SHIPPED 2026-09-15.**
> The form went 12 fields → **21**, in seven sections ordered by the decision
> a person makes rather than by history: *Transmission* · *Transmission
> k-sampling* · *Broadening* · *Outputs* · *NEGF density contour* · *Leads* ·
> *Runtime*. **`TBT.k` is reachable**, so the T(E_F)-against-transverse-k
> study can be run from the tab. Every output flag is offered, so **W10** has
> data to read when asked for. The two PySCF fields are **gone** — the form
> contains the string "pyscf" nowhere — and the five electronic-contract
> fields have a section that names them correctly.
>
> **Emission policy, stated once and applied throughout:** where the manual's
> default is a NUMBER it is the field's default and is always emitted, so the
> deck is self-documenting and behaviour is unchanged; where the default is a
> FORMULA (`min[η_e]/10`, `5 k_B T`, *inherit the SCF grid*) the field
> defaults to 0/`0 0 0` meaning *leave it to the engine* and nothing is
> written — because a number there would silently replace a formula.
>
> **`log_level` got a real keyword rather than deletion:** `TBT.Verbosity`,
> an integer 0–10 defaulting to 5, is in the binary, and the three words map
> onto it (warning 2 · info 5 · debug 8). **The mapping is sourced from the
> TBtrans reference, not invented** — the same pass that retired
> `transmission_relative_to_ef` for want of a sourced mapping could not then
> invent one here.
>
> **Two are deliberately NOT knobs**, and that is the finding rather than a
> gap: `TS.Hartree.Fix` is derivable from the transport axis exactly as
> `semi-inf-direction` is, and the manual calls the boundary *"an intricate
> and important"* matter — it should be derived, never typed. And
> `TBT.ChemPot.<>.ElectronicTemperature` is per-chemical-potential, so it
> belongs with the bias scan, not the override lane.
>
> **Still open:** `TS.Elecs.Eta` (the TranSIESTA-side twin of a broadening
> that IS exposed), `TBT.T.Out` (meaningless below three terminals), and a
> spin-polarised device run wired end to end — `TBT.Spin` selects a channel,
> which is not the same as the SCF producing two.
>
> **Four tests failed on this change and every one was right to.** The
> Methods paragraph still interpolated the deleted `contour_n_circle`; two
> new sections had no description; the served section list is pinned; and a
> bias assertion searched the WHOLE deck for `"1.5000 eV"`, which
> `TS.Contours.Eq.Pole 1.5000 eV` now also matches — a loose assertion
> testing a number rather than a keyword, scoped to `TS.Voltage` lines now.
> The Methods sentence reports only emitted values, which closes the one
> defect here with a publication consequence.



Each verified present in the installed binary, with the manual's default.
**The framework for all of them is § 3.2's per-engine panel** — these are
TranSIESTA's panel, and none of them belongs in a shared dataclass.

| # | what | why it is next |
|---|---|---|
| **1** | **`TBT.k` — the transverse grid for T(E)** | It *inherits the SCF's*, and an SCF grid is routinely far too coarse for transmission. The standard convergence study is **T(E_F) against transverse k** — and it cannot be run from this tab at all. The single largest scientific gap |
| **2** | the `TBT.DOS.*` / `TBT.T.Eig` / `TBT.T.All` output flags (all default `false`) | a run writes transmission and nothing else, so **W10**'s inspector has no DOS or eigenchannel data to read *even in principle* |
| **3** | `TBT.Elecs.Eta` (1 meV) · `TBT.Contours.Eta` (min η_e/10) | broadening sets the T(E) lineshape: too large smears resonances, too small makes them noise |
| **4** | the real `TS.Contours.*` — `Eq.Pole` (1.5 eV), `nEq.Eta`, `nEq.Fermi.Cutoff` (5 k_B T) | what the three dead `TS.ComplexContour.*` fields were reaching for. **Retire or rewire those three in the same commit**, and `contour_n_circle` leaves the Methods paragraph with them |
| **5** | `TBT.ChemPot.<>.ElectronicTemperature` | tbtrans's own, inheriting `TS.ElectronicTemperature`; it is what enters the I–V Fermi functions |
| **6** | `TS.Elecs.Bulk` (true) · `bloch` (hardcoded `1 1 1`) · `TS.Hartree.Fix` | lead treatment and the boundary condition the manual calls *"an intricate and important"* matter |
| **7** | `TBT.Spin` | no spin-polarised transport path, while the junction chemistry check already detects open-shell metals |
| **8** | re-section the five contract fields out of **NEGF** | `basis_size`, `energy_shift_ry`, `xc_functional`, `xc_authors`, `siesta_mesh_cutoff_ry` declare `section: "NEGF"` and are the *electronic contract*. Hidden today, but a form-B citation renders them under a heading that misdescribes them |
| **9** | `log_level` → a real keyword or gone | it claims `WriteVerbosity`, which is **zero** in `siesta` |

**Two things the pass did NOT settle**, recorded rather than guessed:

* **What TranSIESTA does when a required electrode line is absent** — error,
  or silent default. It could not be determined without a live run, and the
  fix made the question moot rather than answering it.
* **Whether the live carbon-chain walk § 8 cites ever exercised a
  non-default window.** No artifact survives in `projects/`. Given the
  default coincidence above, a default-valued walk would have looked correct
  either way.

## 5p. Transport — the implementation, step by step *(2026-09-16)*

**The design is [`engines/transport.md` § 2a](?doc=engines/transport.md)**, agreed
2026-09-16. This section is the distance from here to there, in dependency
order. It does not restate the design; read § 2a first.

### 5p.0 THE RULE — this section is updated as the work happens

> **Every step updates this section when it lands, and every problem found
> while doing a step is recorded here before it is worked around.**
>
> Not afterwards, not in a commit message alone. A step that changed shape, a
> step that turned out to depend on something not listed, a defect discovered
> mid-step: each goes in, with what was measured. A plan that is only written
> once describes an intention; a plan kept current describes the work.

*(User instruction, 2026-09-16: "make it a rule that update of this plan is
mandatory after each step of implementation and/or modification when new
problems discovered in that process.")*

This is R3 applied to one programme — the open rows still live in § 2, and this
section keeps the *why* and the order (§ 0).

### 5p.1 The wheels that already exist — reuse, do not rebuild

**Measured 2026-09-16 by reading the framework, not from memory.** Nearly
everything § 2a describes is already built for the optimization kind. The gap is
not that the machinery is missing; it is that **transport does not use it.**

| what § 2a needs | what already exists | where |
|---|---|---|
| a shared baseline of values (Class A) | **the template** — `<label>.template.toml`, written from the catalogue narrowed to engine **and calculation**, carrying the config's values | `describe.build_description` → `template_with_values(cfg, engine=, calculation=)` |
| per-stage values (Class C) | **`varies` + `Stage.overrides`**, resolved as template ⊕ this stage's overrides | `resolve.effective_config`; `engines/stages.md` § 6.2 |
| a per-stage editing surface | **the stage table** — rows are stages, columns are `varies`, cells are `overrides`, and *an empty cell means "the template's value" and shows it greyed* | `web/task-setup.md` § 5, § 5.1 |
| a kind-aware parameter list | **already kind-aware**: the columns endpoint skips an item whose `calculations` excludes this folder's kind | `GET /api/task-setup/columns?calculation=` |
| the shared baseline read back | `GET /api/task-setup/template-values?dir=` — parses with `read_template`, the same reader `prep` uses | — |
| machine facts per stage (Class E) | `allocation` items + `Stage.execution` | `task-setup.md` § 6 |
| rendering from a description | `spec_for(struct, cfg, stage_token=, calculation=)` → `DeckSpec` → `prepare_deck` | `script_emit` |
| keeping another kind's rows out of a deck | **the kind gate** in the section walk | `script_emit._render_sections` (2026-09-15) |
| the attempt ladder | a launched attempt is never rewritten; re-prep opens the next | `materialize.resolve_attempt` |
| results moving between stages | the DAG copy with three gates + `.gathered-from` | `prep.gather_transport_inputs` |
| one submission walking an axis | the bias chain — `cd` per point, stop-or-continue by whether points depend on each other | `submit.submit_transport_chain` |

**The precedent to follow is the vibration kind**, which is a second calculation
on one engine's seam: one dispatch arm in `spec_for`, the kind's own deck module,
and its rows tagged in the one catalogue.

### 5p.2 The gap, measured

| # | what transport does instead | consequence |
|---|---|---|
| **G1** | **writes no template at all** — a transport folder holds `task.json`, the composed junction and pseudos, and no `<label>.template.toml` | Class A has **no home**. The stage table has nothing to read; `template-values` would find nothing; there is no shared baseline for a cell to inherit |
| **G2** | `config/transport.py` — a second vocabulary, with four parameters renamed from the catalogue's spelling | two names for one quantity; a projection is needed wherever they meet |
| **G3** | the tab builds its form from the dataclass, sectioned by **topic** | a section called "Electrodes" holds the device's bias; nothing says which run a control changes |
| **G4** | the hand-over puts **every** override on the `device` bag | a parameter the transmission owns never reaches the transmission deck — the live defect |
| **G5** | `config_for` fills from the cited `.fdf`, not from a template | the values are inherited rather than single; the first ruling in § 2a.7 cannot be honoured |
| **G6** | `_prep_transport` is a parallel arm that returns before the shared path | no `resolve`, so no `ParameterSet` and no provenance; `--pipeline-log` is a documented no-op |
| **G7** | four of five rungs still render from hand-written emitters | no read-back check, no `.validation.txt`, no check gate on those four |
| **G8** | three keywords the design needs have **no catalogue row**: `TS.HS.Save`, the equilibrium pole **count**, and the bias point `TS.Voltage` | they cannot be declared, defaulted, or reached from a description |
| **G9** | `kgrid` is one row holding three numbers whose components fall in **three** classes | no interface can explain it; § 2a.13 flags it |

### 5p.3 The one design addition, and it needs a ruling

Everything above is reuse. **Two declarations do not exist yet**, and both are
small siblings of markers the catalogue already has (`allocation` — the
scheduler answers this; `citation` — a cited run answers this):

| proposed | means | serves |
|---|---|---|
| **`role`** | *the stage's role answers this; nobody chooses it* | Class D. A `role` item is never a form field and never a stage-table column |
| **`stages = [...]`** | *which rungs may carry their own value for this* | Class C. It is `calculations` one level down, and it is what routes an override to the rung that owns it — the proper fix for **G4** |

**Not yet approved.** Everything else in this section proceeds without them;
the steps that need them say so.

### 5p.3a FOUND WHILE IMPLEMENTING — `citation`'s semantics contradict the ruling

*Recorded 2026-09-16 under R3a, before working around it.*

`template.py::template_with_values` writes a `citation` item into the template
**valueless**:

```python
value=(None if (it.allocation or calculation in it.citation)
       else getattr(config, it.name, it.value))
```

and `config_from_template` refuses one that carries a value. That is the
**sealed** reading — the item is declared but never answered in floor 2, and
`prep` fills it from the cited deck, exactly as it fills an allocation item from
what the scheduler granted.

**§ 2a.7's first ruling reversed that.** The cited relaxation *defaults* these
values; the person may change them, and a change applies to every stage at once.
Under that ruling the citation is **a source of defaults at `init`**, not an
answerer at `prep` — so the template must carry the value, and it must be
editable.

**What this changes, and where it belongs:** the marker survives and keeps its
name, because *"which items are harvested from the cited run"* is still exactly
the question it answers — but its behaviour moves from *valueless, filled at
prep* to *defaulted at init, thereafter an ordinary template value*. That is
**TR1's** substance (it is what makes the template carry a shared baseline at
all), so it is scheduled there rather than done here.

Not a defect in what shipped: the seed migration relies on the current
behaviour and is consistent with it. It is a consequence of the ruling that had
not yet been traced into the code, and tracing it is what R3a is for.

### 5p.3b LANDED — `role` and `stages` *(2026-09-16)*

*Approved by the user and built. Recorded here under R3a.*

Both are siblings of `allocation` and neither is a new mechanism:

| | means | what it buys |
|---|---|---|
| **`role`** — a list of kinds | *the stage's own role answers this* — the third answerer, after the scheduler (`allocation`) and a cited run (`citation`) | a `role` item is not a form field, not a stage-table column, and carries **no value** in a template of that kind, so no description can claim to have set it |
| **`stages`** — a list of rung names | *which rungs may carry their own value*; `calculations` one level down. **Absent means any rung may**, which keeps the optimization ladder exactly as it was | the basis on which an override reaches the rung that owns it — TR8's proper fix |

**Declared on the items § 2a.13 already classified**, transcribing the agreed
map rather than re-deciding it: `solution_method` and `wrap_into_cell` as
`role = ["transport"]`; the thirteen transmission/`TBT.*` items to the
transmission; `electrode_kz` to the two electrodes; the three NEGF items to the
device. The seed owns nothing of its own — it only obeys, which is the formal
statement of why it is skippable.

**They have a reader, deliberately.** `GET /api/task-setup/columns` now skips a
`role` item for that kind and publishes each item's `stages`. A declaration
with no reader is a control that does nothing — the defect `electrode_kz`
shipped with, and not one to repeat while fixing it.

**Verified:** six behavioural tests, the two load-bearing ones mutation-proven
(remove the valueless rule and the template test fails; remove the endpoint
filter and the column test fails). The column test asserts the other half too —
that `solution_method` is *still* offered for an optimization — because without
it the test would pass on an empty column list. Contract in
[`engines/template.md` § 6.4](?doc=engines/template.md). Full sweep of the
affected areas: 1348 passed.

**What did NOT change:** nothing reads `stages` to route an override yet. That
is TR8, and it is the point of the declaration rather than a follow-up to it.

### 5p.3c LANDED — TR1, transport has a template *(2026-09-16)*

**Three edits, and the third is what makes the other two mean anything.**

1. **`citation` flipped from sealed to defaulted** — the finding recorded at
   § 5p.3a, done here because it *is* TR1's substance. The template now carries
   a cited item's value and `config_from_template` no longer refuses one. The
   marker keeps its name: *which items are harvested from the cited run* is
   still exactly what it answers — only the harvest moved to `init`.
2. **`jobset init` writes `<label>.template.toml`**, defaulted from the cited
   deck (`transport/citation_defaults.py`, one function, called once). The k-grid
   is the one value not copied verbatim: the transverse pair carries over and
   the transport axis is forced to 1, at the point the value is born rather
   than corrected by each consumer downstream.
3. **The template WINS at prep.** `siesta_config_for` reads it as the baseline
   and drops from the projection exactly the items the `citation` marker names
   — asked of the catalogue, not listed in code. Without this the citation
   would overwrite the person's edit a line later and the ruling would be a
   fiction; the template would be a file nothing reads, which is the
   `electrode_kz` defect one directory up.

**Verified:** the test that can fail is *edit the template's basis to a value
the citation does not hold, re-prep, the deck must follow* — mutation-proven by
removing the drop. 1672 passed across the affected areas.

**Three tests failed and every one was informative:**

| | |
|---|---|
| I hand-formed the template path | `test_doc_claims` walks the AST to keep `<label>.template.toml` formed in **one** place — six call sites spelled it two incompatible ways until 2026-08-17, so a folder with two templates had the web tab and `prep` reading different files. Mine was offence seven, caught the day it was written. Now uses `template_path` |
| `MeshCutoff 250.0 Ry` vs `250 Ry` | The framework's syntax door formats from the item's **declared type** (float); the hand-emitter it replaces formatted the Python value it held (int). During the migration one rung says each, for a value they agree on. The deck assertion now compares numbers **by value** — asserting the spelling fails a test for a reason that is not the science |
| `assert written == ["task.json"]`, *"floor 2 is task.json ALONE"* | The sealed design encoded as a test, and the same phrase flagged earlier as never having been a ruling. A shared baseline needs a file to live in: transport's floor 2 is `task.json` **and** a template, like every other kind |

**What TR1 did NOT do:** the other four rungs still render from the citation's
values through the old emitters, because they do not go through
`siesta_config_for`. Only the seed reads the template today. That is TR5, and it
is why TR5 follows rather than being optional.

### 5p.3d LANDED — TR2 and TR3 *(2026-09-16)*

**TR2 — the three missing rows, with every spelling verified against the
binary rather than recalled.** `caps.env_prefix` answered this time (it
returned `None` earlier in this programme, which is why the check had been
deferred), so the keywords were read out of the installed SIESTA 5.4.2 through
the manager's own door:

| keyword | in `siesta` | in `tbtrans` | row |
|---|---|---|---|
| `TS.Voltage` | 4 | 2 | `bias_voltage_v` — Class C, the **device**'s, binding its own transmission point |
| ~~`TS.Contours.Eq.Pole.N`~~ | — | — | **Withdrawn 2026-09-16: there is no such keyword.** SIESTA has the string and never queries it; the pole count is derived as `N = E / (pi kT)`. The abort came from `negf_eq_pole_ev = 1.5` eV shipping as the default — 18 poles at 300 K. See `engines/transport.md` § 3.6 |
| `TS.HS.Save` | 1 | 0 | `ts_hs_save` — the **electrodes**', and marked `role`: a lead that omits it converges happily and produces nothing the device can attach to, so nobody is offered a switch |

**TR3 — smaller than this plan predicted, and the difference is the finding.**

§ 2a.13 said the `kgrid` row *"needs to become the transverse grid for
transport, with the transport axis owned separately."* Building it showed two
things that change the answer:

* **a separate `kgrid_transverse` row would be a second name for one physical
  quantity** — precisely the `TransportConfig` problem TR6 exists to delete. The
  cure would have been an instance of the disease;
* **the lead's transport axis already IS owned separately**, by `electrode_kz`.

So nothing was missing from the *rows*. What was missing was a **check**: a
person could set the third component in the template and the renderer would
quietly write 1 anyway — a control that appears to do something and does not,
which is the defect this programme keeps finding. Tier 2 in § 2a.4's terms, so
it is a refusal that names the reason, not a silent correction.

**It is registered as a KIND validator, and that distinction matters.** The
existing `_validate_transport` is keyed on the `TransportConfig` *class*, so it
stopped firing for any rung that moved onto the seam (the seed, 2026-09-15). A
rule that runs for one of two config classes is not a gate. This also closes
§ 2a.7's note that passing `calculation="transport"` read as though a gate
existed when `_KIND_VALIDATORS` held only `vibration`.

**Verified:** both tests mutation-proven; the refusal test has a second half
asserting the transverse pair still reaches the deck, without which deleting the
k-grid control entirely would pass. 1752 passed across the affected areas.

**Housekeeping noticed, not a defect:** two doc counts were back at their
pre-4b values because the per-stage-presets revert restored those files
wholesale, taking two unrelated count edits with it. The revert was slightly
wider than the change it undid. All three counts are now the measured numbers.

### 5p.3e LANDED — TR4, the transport arm resolves *(2026-09-16)*

`engines/transport.md` § 3.2 measured `_prep_transport` as **a second
conductor that decides** — it composed, gated, extracted and rendered, so a
transport run had no `ParameterSet`, no provenance, and `--pipeline-log`
printed *"not wired for the transport arm yet"*.

The seed rung now goes through `resolve`, floor 3's own step 2. The log is
real, and every value names its source:

```
STEP 2 · RESOLVE — the description becomes a ParameterSet
  in   template               T.template.toml
  out  elements               1 (a run, not a sweep)
  ⊕    basis_size             TZP          <- template
  ⊕    dm_tolerance           1e-05        <- template
  ⊕    kgrid                  (4, 4, 1)    <- template
```

**TR1 did not just make this reachable — it made it sufficient.** `resolve`
reads a template, and a transport calculation did not have one; that was the
whole blockage. Now that it does, and the template carries **every** item the
kind has (the transport-only rows included), there was nothing left for a
hand-built projection to add.

**So `siesta_config_for` is dead — 143 lines deleted**, `_TO_SIESTA_NAME` with
it. That two-vocabulary mapping was written three hours earlier for TR1 and was
scheduled to die in **TR6**; it fell out here for free, because the thing it
bridged no longer has two sides for this rung. A step that deletes a later
step's work is the shape to look for.

**Verified:** 1914 passed. Three tests, including the honest failure — a
description written before TR1 has no template and is **refused by name with
what to do**, rather than falling back to re-reading the citation, which would
be the sealed behaviour returning silently by the back door.

### 5p.3f THE SEAM QUESTION — WITHDRAWN, it was mis-posed *(2026-09-16)*

I asked whether *"a derived bulk lead is legitimately a `Structure`"* and put
TR5 behind the answer. **The question does not arise, and the design already
said so** *(user, correcting me)*:

> **One structure carries everything.** It holds the frozen electrodes at both
> ends and the relaxed bridge between them. Transport takes that one file,
> checks the labeled electrodes are the fixed atoms, **takes those atoms out**
> to build each lead's single-point run, and combines the bridge with the two
> self-energies. One file, so *"all the structure and facts involved in the
> calculation are consistently constructed."*

Nothing is derived from elsewhere. `compose._extract_and_gate_electrodes` calls
`extract_electrode_model(dev, REGION_LEFT_ELECTRODE)` — **a subset of the cited
structure, selected by label**. Same atoms, same relaxation. Asking whether it
is legitimately a structure was asking whether a selection of a person's own
relaxed atoms is a structure.

**So TR5 is UNBLOCKED and the answer is option (b):** each rung renders the
structure it describes — the extracted lead for an electrode rung, the junction
for the device — and the seam stays `(struct, cfg)`. `ElectrodeModel` already
holds everything a `Structure` needs (elements, positions, the lateral cell,
the bulk z-period); yielding one is mechanical, not a design question.

*The lesson is the one this programme keeps relearning: I had read § 4 (*"region
labels drive everything"*) and the extraction code earlier in this work, and
still posed a question the design had answered. Consulting the contract is not
the same as remembering that I did.*

### 5p.3g LANDED — the first validation *(W29, 2026-09-16)*

*Found and built the same day, under R3a.*

The design's **first** check — *"look for the labeled electrodes and make sure
they are the fixed atoms"* — did not exist. What existed was a different and
**later** one, and the distinction is the whole finding:

| | when | asks |
|---|---|---|
| already there — the frozen-unmoved gate (`compose.py`) | at compose, **after the relaxation has run** | did the electrode atoms *move*? Compares the cited deck's coordinates against the `.XV` |
| **added** — `check_electrode_labels_are_frozen` | at setup, **before anything runs** | are the electrode-labeled atoms *declared frozen*? |

Neither substitutes for the other. Label the leads, forget to freeze them, and
nothing objected: the relaxation ran, the leads relaxed, and the junction was
found unusable at compose — **a wasted relaxation, on a metal junction where
that is not cheap.** The closest thing was a hint inside an unrelated warning.

It runs in **both engines' settings gates**, beside
`check_unconsumed_region_labels`, because it is the same question one step
further: the labels say which atoms are leads, and a lead must survive the
relaxation untouched.

**A warning, not a refusal — and the line is deliberate.** A structure carrying
electrode labels is *heading* for transport but has not committed: a person may
relax the whole junction once before freezing the leads for the run that
counts, and refusing here would block that. The refusal belongs at compose,
where transport IS the intent, and it is already there. *Open for the user to
overturn: making it an error is a one-word change.*

**Verified:** six tests, and two earn their place specifically. *The bridge is
not required to be frozen* is the discriminating case — a junction's purpose is
that the bridge relaxes while the leads do not, so a check demanding every
LABELED atom be frozen would refuse every correct junction. *It reaches a real
preflight* is mutation-proven by unwiring it: a check nothing calls is a check
that does not exist, which is the `electrode_kz` lesson. 1913 passed.

### 5p.3h LANDED — TR5a, the electrode rungs on the seam *(2026-09-16)*

**The seam did not have to change, and the reason is the one-file design.**
`ElectrodeModel.as_structure()` restates the extracted lead as an ordinary
`Structure` carrying its own cell — the device's lateral vectors verbatim and
the bulk repeat along transport. It is a restatement, not a conversion: those
atoms came out of the cited junction by their region label, same atoms, same
relaxation. So a deck still describes a structure; there are simply **two
structures in one file**, and each rung renders the one it describes.

Measured: the lead deck went from a hand-written f-string to **423 lines** with
the engine's full section set, a `.validation.txt`, the check gate, and:

```
    4    0    0      0.0      <- transverse, shared with the device
    0    4    0      0.0
    0    0   40      0.0      <- the transport axis, DENSE: a lead is
                                 periodic there and its Fermi level is the
                                 reference energy everything is measured against
```

**A test predicted its own obsolescence and was right.**
`test_the_un_migrated_rungs_still_lack_them` asserted the electrode deck
LACKED the template's items, and carried an instruction: *"delete this when the
electrode rung is tabled — at which point it should fail, and that failure is
the migration being done."* It failed, and was deleted. Its job — keeping
*"this rung is on the seam"* from degrading into *"SIESTA decks tend to have
SCF settings"* — passes to `TestTheElectrodeRungIsOnTheSeam`, which asserts the
contrast the other way round: the one thing the lead must have that the seed
must not, the dense transport axis.

**Verified:** five tests, including *the lead deck is the LEAD not the
junction* (it must describe the extracted subset's atom count, or the others
would pass on a deck that quietly rendered the whole junction). 144 passed.

**Remaining in TR5:** the `negf` shape — device and transmission. Those
describe the junction, which the seed already proves the seam can carry; what
they additionally need is the region partition for `%block TS.Elecs`, and it
travels on the structure's own `regions`.

### 5p.3i LANDED — TR5b, every rung on the seam; and the contract consolidated

**TR5 is complete.** All five rungs render through `spec_for` → `DeckSpec` →
`prepare_deck`. Measured on the repository's own fixture: seed 488 lines, a
lead 419, device and transmission 584, each with a `.validation.txt` and the
engine's check gate, against the **13 keywords** § 3.2 measured.

The device and transmission share ONE layout but resolve their OWN configs, so
they share their shape and differ in exactly what a person tuned for the
transmission — which is what § 2a.7's ruling asked for, reached **without
anyone deciding which keywords `tbtrans` requires**. That question would have
needed verification against the binary; the per-rung config made it moot.

**Deliberately NOT tabled: the NEGF electrode block.** `%block TS.Elecs` and the
per-electrode blocks stay the pre-seam emitter's, wrapped in one `Block` with a
small projection at the boundary. TranSIESTA identifies each electrode by a
**contiguous atom range**, so an off-by-one computes transmission through a
region that is not the molecule *and converges while doing it*. That emitter has
been measured against a live 5.4.2 binary; a rewrite would have to earn that
again for no gain.

#### Two defects this step found in itself

**The check gate caught a duplicated keyword the day it was written.** The
device deck said `SolutionMethod diagon` from a section and `transiesta` from
the NEGF block — libfdf takes the first. The cause is general: **a `role` item
must not be written by a section**, because a section resolves a value from the
config and a role item rightly has none there. `_render_sections` skips them
now and each rung's block writes its own. The same lift-boundary rule that kept
`_emit_basis_and_xc` out of the seed layout, one level deeper — I had applied
it once and still missed it.

**A test fixture was geometrically invalid and nothing had ever looked.** Its
buffer variant put 44.5 Å of atoms in a 40 Å cell. It survived for as long as it
existed because **the device deck had no settings gate until it joined the
seam**: the first time anything validated a device geometry was the moment this
work put a gate in front of it. The gate is right; the fixture is sized from its
own geometry now.

#### The contract, consolidated *(user: "fully updated, consolidated, and checked")*

`engines/transport.md` no longer describes a design and a different
implementation:

* the § 2a banner says **built**, not *"none of it is implemented"*;
* § 3.2's floor-3 row is **closed**, with the measurement;
* the *"same bytes"* note is **resolved**, with why the per-rung config settled
  it rather than a keyword audit;
* § 3.6's outcome list is reconciled item by item — and item 5's own wording
  corrected: the values arrive from the **description**, not from the citation,
  which supplies only the defaults;
* **§ 2a.14** is new: what landed, a diagram of template ⊕ overrides → resolve
  → spec_for → prepare_deck, the two-structures-one-file table, the measured
  deck sizes, the lead's k-mesh as the physics made visible, and what did NOT
  land;
* § 2a.8's worked example was written before any of this existed, so it was
  **re-run against real decks** and now carries the result: one shared edit,
  three rungs follow.

**Still open and named there:** the bias has two homes (`task.bias` the axis,
`bias_voltage_v` the template row), `TransportConfig` survives only to feed the
lifted block, and net charge / gating are deferred by ruling.

### 5p.3j FIXED — the stage deck ignored the bias axis *(2026-09-16)*

**I raised this as an open design question and it was not one.** Adding a
`bias_voltage_v` catalogue row in TR2 looked like it had given the bias two
homes — the row, and `task.bias`'s list — so I asked which should win.
§ 2a.10 had already answered:

> *"single bias is the degenerate case of the bias axis — one point, normally
> at zero."*

**One mechanism.** The template declares the parameter (its range, unit and
help) and answers the one-point case; a scan is that same parameter taking
several values, which is the framework's ordinary precedence — *the template's
value ⊕ this point's*. Nothing to rule on. That is the **second** time in this
programme that a question I posed as open had been settled in a section I had
already read; the first was the seam (§ 5p.3f).

**The defect underneath it was real.** Measured with the template set to 0.5 V
and the axis at `[0.0, 0.2]`:

| | before | after |
|---|---|---|
| `04_device/T_04_device.fdf` | **0.5 V** — the template's own value | 0.0 V |
| `04_device/v0/…` | 0.0 V | 0.0 V |

§ 2a.11 describes the stage-directory deck as *"the same deck `v0/` holds"* —
it exists so the job row's script sits where every generic reader looks. It was
not that deck, and whichever of the two a person opened they would have
believed it.

Three tests pin it, including the degenerate case: with no axis the template
answers, and there is no `v*` level at all — § 2a.11's own rule, *a stage
carries a sub-level for each axis it varies over and none for an axis it does
not*.

**Recorded while fixing it: the row says volts, the deck says eV.** Not a
mismatch. fdf treats `TS.Voltage` as an energy, and the energy an electron
gains across V volts is exactly V electron-volts — same number, different unit
word. Written into the row's help, because the next person to notice will
otherwise "fix" one side to match the other and break it.

### 5p.3k LANDED — TR8, an override reaches the rung that owns it *(2026-09-16)*

**The defect this programme started from is closed.**

Every override went onto the `device` rung, whatever it was. So a person who set
the transmission's energy window had it written into the deck `siesta` runs —
where `TBT.*` keywords are inert — and **not** into the deck `tbtrans` runs,
which is the one that computes T(E). No error, no warning: you asked for ±3 eV
and got the default.

`route_overrides` asks the catalogue which rungs may own each item (the `stages`
declaration, `engines/template.md` § 6.4) and puts the value there:

| set | lands on |
|---|---|
| `transmission_emin_ev` | the **transmission** |
| `electrode_kz` | **both** electrodes — a junction has two leads, and routing to one would leave the self-energies built on different Fermi-level resolutions |

**Verified:** three tests, all mutation-proven — revert the routing and all
three fail, reproducing the original defect. One exists to make this a
*routing* test rather than a *delivery* test: it asserts the device does NOT
also get the transmission's parameter, since being inert there is precisely why
nobody noticed it was the only place it landed.

#### Found while doing it

**`config_for` was refusing a person's own lead k-density.** It validated
overrides against `TransportConfig`'s vocabulary, but an override now names a
**catalogue row** — what the template declares and `resolve` resolves — so
`electrode_kz` came back as *"not a transport parameter"*, which it plainly is.
The vocabulary is the engine's now, with `TransportConfig`'s accepted beside it
only while that class survives.

**And that function is close to vestigial.** It is called once, and its result
is handed to a pseudo-provisioning step that reads **no field from it**. What it
still uniquely guards is **identity** — `job_name`, the bias axis — which
`resolve` would let a stage override, because to `resolve` those are ordinary
schema fields. *That is what TR6 must preserve when `TransportConfig` goes*, and
it is written at the decision point rather than left to be rediscovered.

#### A holding position, stated as one

An override for a **shared** SCF control (no `stages` declaration) still goes to
the device. That is not an answer: those are items any rung may legitimately
own, and a flat form cannot say which was meant. **TR7's per-stage surface is
where the question becomes askable** — and the stage table already exists for
optimization, which is the whole point of § 5p.1.

### 5p.3l PART OF TR7 — the seal's REASON corrected; the interface half not built

**What was wrong.** Two doors refused a Class A field as a per-stage override
with the message *"the citation's to say — cite a relaxation that ran with the
values you want."* § 2a.7 reversed that rule, so by 2026-09-16 the advice was
**actively misleading**: it sent a person to redo a relaxation when they could
edit one line of the template.

**The refusal is still right, for a different reason**, and that is the whole
correction: a value shared by every rung cannot be a per-stage override,
because that is exactly how the device would come to disagree with its own
leads. So the guard stays and the reason changes — and the message now names
**where the value is actually changed**.

**And the form-A/form-B distinction went with it**, which is more than a message
fix and is recorded as such. The old rule OPENED these fields for a form-B
citation (a labeled pair, no deck) because the override lane was then the only
way to state a basis at all. Every transport calculation carries a template now
— form A filled from the cited deck, form B from the catalogue's defaults — so
that hole no longer needs to exist, and leaving it would mean **a form-B
junction could give its device a different basis from its leads**. Verified
rather than assumed: a form-B calculation does get a template, and stating the
basis there reaches the deck.

**Four tests changed, and how matters.** Three asserted the *reason*
(`"citation's to say"`); they now demand the corrected reason **and** that the
message names where to change the value. The fourth set a stage override to
prove a pair's basis was stateable — its concern was exact, *"there would be no
way to state the basis at all"* — so it proves the same thing through the
template. The concern survived; the mechanism moved.

#### NOT built, and it is the part that needs a decision

The tab still builds its form from `dataclass_to_form_schema(TransportConfig)`
and still **hides** the Class A fields. Hiding them was correct under the old
rule — *"a form field the door is guaranteed to refuse is a trap, not a
control"* — but under § 2a.7 they are the person's to change, so hiding them now
means **there is no way to change a transport calculation's basis or XC except
by hand-editing the template file.**

That is § 2a.6's Panel 0, and it carries a real question rather than just work:
the tab writes **stage overrides** today, and Panel 0 would have to write the
**template** — a different file, written by `jobset init` from the citation.
Either the tab edits the template directly, or the Class A values are edited
through **Task Setup**, where template values already have a home
(`/api/task-setup/template-values`) and the stage table already exists. The
second reuses more and touches the Transport tab less.

**Left for a ruling** rather than decided — it is the one remaining piece that
designs what a person sees rather than what runs.

### 5p.3m FIXED — a description made in the browser had no template

**A regression of mine, from TR1, live for several steps.**

TR1 made `jobset init` write `<label>.template.toml`. The **web describe door
does not write files — it RETURNS them** for the browser to write, and it
returned only `task.json`. So a transport calculation created in the browser
had no shared electronic description, and `prep` refused it by name: *"this
transport calculation has no template, so there is nothing to resolve."*

Fixed: the door returns both, the template's values defaulted from the cited
run.

#### Why four sweeps missed it, which is the part worth keeping

The test that would have caught it **immediately** is
`test_task_setup_tab.py::test_transport_describe_answers_the_finished_description`
— it posts to `/api/transport/describe` and asserted the response's file list.

**My sweeps selected files by NAME**: `-k transport` never matched
`test_task_setup_tab.py`. The tests that exercise a surface are not always in
the file named for it, and a filter chosen from names measures the names.
Corrected by running every non-browser file that *mentions* transport
(`grep -rln transport tests/*.py`), which is 12 files beyond the ones named for
it, and all pass.

#### Three tests in that unswept file, each saying something

| | |
|---|---|
| `== ["task.json"]`, *"floor 2 is task.json ALONE"* | the sealed design encoded as an assertion — **the same phrase flagged twice already in this programme**. Now asserts both files, and that the template is real: the reader `prep` uses accepts it and it carries the cited values |
| `"citation's to say"` in the refusal | the message § 2a.7 reversed. Now demands the corrected reason **and** that the message names where the value is changed — a refusal with nowhere to go is not a refusal, it is a dead end |
| the override lands on `device` | TR8 working: `transmission_n_points` goes to the **transmission** now. Asserts that, and that the device does *not* also get it |

### 5p.3n FULL-TEXT REVIEW — what four agents found, verified, and fixed *(2026-09-16)*

Asked for after *"I don't trust your work at all"*: read every document, every
backend file and the UI **in full text**, find redundant code, duplicates,
magic numbers, hacks, and doc-vs-code disagreement. Four agents read
~12,000 lines. **Every claim below I re-derived against the code myself before
acting, and two agent verdicts were wrong in the direction that matters.**

#### The live defects, each measured

| | what was measured | fixed by |
|---|---|---|
| `_prep_transport` opened **no attempt** | it ended at `prep_jobset`; the CLI opened one itself. `web/blueprints/build.py:1656` calls `prep_calculation` directly, so the browser's Prep button reported success and handed back a folder `submit._launch_dir` refuses — *naming the command that had just run*. The shared arm's own 27-line note calls this out: *"A verb that is only finished by one of its callers is not a verb."* It reintroduced exactly that | one `_open_attempts(js, base, stage, containers=…)` that **both arms** call; the scan's per-point ladders are DATA on the call |
| the resolved **resources were thrown away** | `_resolve_transport` returned `element.values` and the arm rebuilt the allocation by hand. `resolve` folds two riders onto `element.resources` — `continue_retries` and `use_gpu`. Measured: `SiestaConfig` defaults them `1` / `False`, `Resources` defaults both `None`, so **every transport wrapper rendered with no warm-retry loop** and `use_gpu` fell back to grepping the deck for `Diag.ELPA.GPU` — the re-derivation `gpu.md` G7 deleted. `resolve.py`'s A-5 finding, on a third road | return the element; use `element.resources` and `element.render_config()`; fold the allocation **before** the resolve, as the shared arm does |
| wrappers were written **past** the remote-activation guard | `prep run device --target sol` against a record stating no activation wrote one wrapper per bias point carrying *this* machine's activation, then refused. The refusal's whole premise is that those files must not exist | the guard moved above the write |
| `TS.Contours.Eq.Pole.N` **reached no deck** | a catalogue row with an anchor and a default that no section named. *(The fix recorded here — a `CONTOUR_SECTION` naming the row — was **wrong**, and a real run disproved it the next day: there is no such keyword. See § 5p.3o)* | superseded |
| the high-bias advisory **stopped firing** | `TransiestaEngine.preflight` is registered in `_ENGINE_VALIDATORS` keyed on `TransportConfig`; every rung now resolves a `SiestaConfig`, so it dispatches for nothing. Of what it carried, the region partition and atom order are `sort`'s own refusals and structural here, and open-shell runs from the siesta validator against the run's REAL spin treatment. The remainder was the \|V\| > 2 V advisory — and bias is the one axis transport exists to sweep | re-homed into `_validate_transport_kind`, beside the kz≠1 refusal. **And the dead checker itself is now GONE — 2026-09-17.** This row said it "dispatches for nothing" and left it standing, because ONE surface still built and validated a `TransportConfig`: `POST /api/transport/render`. Deleting that route (step 2c) removed the last reason, and the holistic review found two documents still calling the checker *"defense in depth"* for an ordering that is held by construction. `TransiestaEngine` is deleted; its tombstone names a live holder for every check it carried, verified one by one |
| the Methods paragraph named a basis nobody ran | under a docstring reading **"EVERY NUMBER HERE IS NOW ONE THE DECK CARRIES"**, the next line wrote `"a DZP basis, the PBE functional"` as literals. TZP/revPBE produced a paragraph claiming DZP/PBE — the same class the docstring calls *"the one defect with a PUBLICATION consequence"* | read off the config; verified both ways |
| `_sh` meant two things in one function | `_sh = shape_of(...)`, then `import shutil as _sh` **rebinds that function-local for the whole body**, so a later iteration hands the shutil module to `trial_work_dir` as a shape | `from shutil import copy2 as _copy2` |
| the electrode deck contradicted itself | a comment saying *"(Not `SaveHS`…)"* two lines above a layout that includes `OUTPUT_SECTION`, whose `write_hs` row has `anchor = "SaveHS"`, `value = true`. Rendered the lead: `SaveHS .true.` is there | the comment says what is true |
| `?contract=open` was a trap | the schema endpoint un-hid the seven contract fields for that lane while the describe door refuses them **unconditionally** — seven controls guaranteed to 400. `?contract=open` now selects nothing | the filter hides them in both lanes; the test that asserted the old rule rewritten to the shipped one, and mutation-tested |
| form persistence matched **zero elements** | `_restoreFormValues` queried `[name=…]`; `form-schema.js` sets `id` only, on all four builders | calls `formSchema.setValues`, the renderer's own door |
| the web door refused what the CLI accepts | `_known` checked `TransportConfig` only; `electrode_kz` is a `SiestaConfig` field and a catalogue row | widened to the union, the same vocabulary `prep` uses |
| dead code | `stages.render_stage_deck`'s unreachable refusal in `transport_spec` (9 lines), `if True:` wrapping 90 lines in `submit.py`, `contract_sealed`, six dead imports, two placeholder-less f-strings in emitted deck text | deleted |
| a **parameter that did nothing** | `config_from_template(..., calculation=)` — threaded by two callers, read by the body **never**, while `template.md` § 6.4 pointed at it as *the* enforcement site | deleted; its two tests became one |

#### Two agent verdicts I did not take

* *"`select(role=)` / `select(stages=)` have zero callers — dead."* True of
  production and the **wrong verdict**: `tests/test_template_role_and_stages.py`
  uses both as the reading API, and the alternative is every reader
  hand-filtering the catalogue. Kept. Six lines that earn themselves.
* *"Fold the DAG gather into `prep`."* I did, and **19 tests failed** — correctly.
  `gather_transport_inputs` refuses an upstream that has not CONCLUDED, so
  calling it at prep means you can no longer render the device deck to READ it
  before spending the queue. That is the split `prepare_attempt` names in its own
  words. Backed out; the browser-road gap is recorded below as a question about
  what `prep` MEANS, not a defect to close by widening it.

#### The documents

`template.md` § 6.4 is the `citation` marker's ONE HOME and still stated the
**sealed** reading § 2a.7 reversed — in three sentences, pointing at the dead
parameter above as the enforcement. Under this project's own convention every
other restatement points there, so the one home being wrong is what made the
other sites unfixable one at a time. Corrected first, then the restatements:
`transport.md` § 0.4 rows 1–2, § 0.5 (*"and no validation file… writes with
`write_text`"* — measured false: every rung goes through `prepare_deck` and
writes its `.validation.txt`), § 3.6a's three residue rows, § 6.1's diagram and
§ 6.1a's table (which named three renderers, **none** on the composite path, so
a reader fixing a keyword would have edited dead code), § 7's *"the basis / XC
block is hardcoded… edit the emitted `.fdf`"* — advice the read-back gate would
now refuse — and the module docstrings of `transport/stages.py`,
`transport/__init__.py` and `transport/deck.py`.

#### Open, and NOT closed by me

| | |
|---|---|
| the DAG gather runs on the CLI road only | a browser prep opens the attempt but never carries the upstream `.TSHS`/`.DM` in. Closing it means deciding what `prep` means — see above |
| `stages.render_stage_deck` (52 lines) and `config_for` (140) are production-dead | `config_for` still has three tests. Deleting it retires them; that is a call about what those tests are for |
| ~20 doc counts have drifted | "13 keywords / 45 items", "32 fields", "49 siesta items", "the 21 keywords". Historical measurements stated in the present tense. `test_doc_claims.py` pins some and not these |
| `wrap_into_cell` declares `role = ["transport"]` and nothing answers it | inert today because `_emit_geometry` never wraps. A decorative declaration |

### 5p.3o MEASURED ON THE ENGINE — the pole keyword is not a keyword, and our default aborted every device run *(2026-09-16)*

Asked to verify one thing (does the electrode rung need `TS.DE.Save`?), which
needed a real TranSIESTA run. The run answered that question and two others
nobody had asked.

**The method, because it is the point.** A 12-atom Au wire junction, decks
rendered through molbuilder's own emitters — not hand-written — then the lead
run on SIESTA 5.4.2, its `.TSHS` handed to the device, and the device run.

| question | answer | how |
|---|---|---|
| does the lead need `TS.DE.Save`? | **No** — and the reason given here first was the wrong one | the run showed it (device completed, exit 0, no electrode `.TSDE` present, `L/R principal cell is perfect!`), but a run is one configuration. **Settled from `m_ts_options.F90`**: the demand is guarded by `DM_init > 0` (`:1571`), which needs `TS.Elecs.DM.Init` = `bulk`/`force-bulk` (`:450-459`; the default chain lands on `diagon`) AND zero bias (`:460-468`; finite bias forces it to 0). This ladder writes none of those keywords. `TS.Elecs.Bulk` — what I first credited — is a different option, selecting whether the SELF-ENERGY uses the bulk Hamiltonian (`m_ts_elec_se.F90:49`). See `engines/transport.md` § 2a.15 |
| is `TS.Contours.Eq.Pole.N` a keyword? | **Yes — and it cannot act here.** | it is a real `fdf_get` (`m_ts_chem_pot.F90:113`). Our deck shape takes the continued-fraction branch (`:299`), where the count is overwritten from the ENERGY at `:319` and the branch default is non-zero, so the override always fires. Only the block-interior `contour.eq.pole.n` (`:263`) can set it. *(I first reported this as "not a keyword" on the strength of the run alone — see below)* |
| why did the device abort? | **our own default** | `negf_eq_pole_ev = 1.5` eV is 18 poles at 300 K, and TranSIESTA refuses fewer than 20. Measured: 1.5 → abort, 1.7 → 20, 2.0 → 24, 4.0 → 49, nothing written → 42 |

**And then the METHOD was wrong too, which is the lesson worth keeping.** I
established all of the above by poking the binary — running decks and reading
what came back — when the source was one `curl` away and the project's own rule
says *"let me try X and see about code I can open IS the violation"*. The user
said so, and reading `Src/m_ts_chem_pot.F90` changed two of the three answers:
the keyword is real (it simply cannot act on this deck shape), and the rule is
`N = int(E / (pi * kT))` with a `< 20` die at `:324` — which is what the
validator now cites, rather than the curve I had fitted to five data points.
The fitted curve happened to be right. It was right by luck, and a fitted curve
that is right by luck is indistinguishable from one that is not.

A third fact only the source gives: an energy written INSIDE the chempot block
means something different from the same energy written at the top level —
`:312` applies a 0.7 factor *"only up to 70% as they become sparse"*. No
amount of running our own decks would have surfaced that, because our decks
never write it.

**So the fix shipped the day before was wrong.** § 5p.3n added a
`CONTOUR_SECTION` emitting `TS.Contours.Eq.Pole.N`, on the strength of a
catalogue row whose help text asserted the diagnosis and a binary-strings check
that passed the label. Both were wrong in the same direction, and neither could
have been right: **a string in a binary is not proof that fdf queries it.**
That is now written into the guard's own docstring, because the guard passed
this keyword and will pass the next one of its kind.

The real defect was one line away from the one I fixed, and worse: a shipped
default that stops every device run after the queue wait, having read the
electrodes.

**What landed.** `negf_eq_pole_n` deleted — the field, the catalogue row and
the section. `negf_eq_pole_ev` defaults to **0 = let the engine choose**, the
established "0 means the engine's formula" pattern this very block already uses
for `TS.Contours.nEq.Eta`, and the emitter writes the line only when a person
names a value. The engine's choice scales with the temperature; a fixed number
cannot, which is the argument against simply raising 1.5 to 2.0.

And because the threshold is a RELATION, `_validate_transport_kind` refuses a
stated energy that gives fewer than 20 poles **at the run's own temperature**,
with the arithmetic in the message — 1.7 eV passes at 300 K and is refused at
1000 K. A deck rendered with the new defaults now runs to completion: 42 poles,
`Job completed`.

### 5p.3p TWO ERAS IN ONE PACKAGE — the composite's residue, and the layers below it *(2026-09-17)*

**The nature, in one sentence.** Transport was built twice. The FIRST build
(2026-06-10 to 06-27) assembled a junction **by hand** — you wrote a device
deck, derived an electrode deck, and a preflight compared the two because
*"humans break exactly these couplings"* (`transport/preflight.py`). The SECOND
— the composite, migrated 2026-08-29 (`workflow.md` § 7), finished by TR5b
(§ 5p.3i) — **derives everything from one cited junction**. The second shipped;
**the first was never removed.**

This section tracks that removal AND the layers underneath it that the removal
exposed. It is re-consolidated at every milestone (§ 5p.3p.5), because two
passes over it already changed the answer rather than refining it.

#### 5p.3p.1 The layer map, top down — contract, and state

Read this before touching anything here. Two of these layers were missed on the
first two passes precisely because the work started in the middle.

| layer | contract | state |
|---|---|---|
| the whole | `workflow.md` § 7 | ✅ records transport as the composite |
| tab + describe | `web/tabs.md`, `web/form-schema.md` | ✅ four routes, citation-driven |
| **CLI surface** | **`process/conventions.md` § 3** | ✅ **closed 2026-09-17.** Decision 7 + 34 (*"everything is a job set"*; `run` and `fdf` deleted as *"obsolete residue from the flat-dir design"*) now holds without exception: `transport` (steps 2c/9) and `pyscf` (step 8) were the two survivors of that shape and both are gone. **19** top-level commands, re-derived from `cli.commands`; no calculation KIND and no engine has a verb, and § 3 states that as a rule rather than leaving it to be inferred from the roster — see § 5p.3p.6 |
| **presenters** | **`web/presenters.md`** | ⚠ § 1's count CORRECTED 2026-09-17 (six register, four are results — `bench-summary` had been omitted). **Step 5 adds the composite row**; what it cannot fix is the LADDER (five directories, one run), which is § 5c.1's open question, not a presenter's. A module rename to `presenters` is declared pending (W15) and not started — the code is still `inspectors` |
| results tab | `web/results.md` | ⚠ § 2.3 now states the LADDER case and § 3 the real viewer count (2026-09-17). Still ⚠ because it describes a gap rather than a behaviour — closes with step 5 + § 5c |
| HTTP | `web/web-api.md` | ✅ four live transport routes; `/api/transport/render` deleted 2026-09-17 with its orphaned config builder |
| description | `engines/stages.md`, `execution/job-contracts.md` | ✅ five rungs, hierarchical shape |
| template | `engines/template.md` | ⚠ 485 facts in two homes; the mirrored guard restored 2026-09-17 |
| engine / deck | `engines/transport.md` | ✅ § 6 rewritten 2026-09-17 to name ROLES not function lists (§ 6a says why); § 5's holders named at 2c. The unbuilt transmission inspector is now stated as unbuilt, in `results.md` § 2.3 as well |
| **parse / directory** | **`model/parse.md` § 5** | ✅ `JobDirParser` → `RunDirResult` **BUILT + MIGRATED 2026-09-18** (§ 5c steps 1–4, **CLOSED**; proved 141/141 on the real tree before any caller moved). The discovery chain is single-homed in `parse/dirs/rundir.py`; `web/blueprints/watch.py` lost 166 lines. **Five of the six caller-map rows were withdrawn with measurements, not deferred** — they were name collisions, not duplicate readers. The Results picker was struck from the map, not left open: it is not a caller of this door — its question is the LADDER's, answered by `jobset_status`, and an HTTP surface over that belongs to § 5p.3p. § 7.9's absorption-site count corrected 2026-09-17 (four → two; the other two were the deleted transport readers) |
| **engine registry** | **`engines/overview.md` § 5** | ✅ *there is no registry* — spectra's went at P3 (2026-08-21), transport's 2026-09-17. § 5 told a new engine to `@register_engine` against it until corrected 2026-09-17; `spectra/methods.py` and `transiesta.py` carried the same claim in docstrings |
| **what the wrapper is HANDED vs re-reads** | **`execution/gpu.md` G7 + `execution/architecture.md` A8** | ⚠ **added 2026-09-17, and it is what stopped step N5d being trivial.** G7: *"the value travels; the deck is not re-read for it"* — reached 2026-08-23 **by carrying the answer on `Resources`**, not by a declaration alone. A8: the allocation *"arrives whole"* — `render_wrappers` was cut from eleven loose kwargs to one record on 2026-08-17. So handing the wrapper a value means a `Resources` field or nothing; there is no third door |
| **validation dispatch** | **`science/validation.md`** | ✅ **added 2026-09-17, and it is where the day's biggest defect lived.** Two registries, and WHICH one a science belongs in is the whole question: `_ENGINE_VALIDATORS` keys on a **config class**, `_KIND_VALIDATORS` on `task.calculation`. A row in the first is only as live as the callers that CONSTRUCT that class — and transport's keyed on `TransportConfig` while every rung resolves a `SiestaConfig`, so it dispatched for nothing. Now two engine rows (SIESTA / PySCF) and two kind rows (transport / vibration), asserted by **equality** |
| execution | `execution/script-preparation.md`, `architecture.md` | ✅ |

#### 5p.3p.2 Redundant — tracked, with status

| the June piece | replaced by | evidence | status |
|---|---|---|---|
| `render_script` + `_emit_header` + `_emit_k_mesh` | `deck.py` (negf shape) | only caller was `/api/transport/render`; no browser POSTed there since 2026-08-29 | **DONE** |
| `render_electrode_fdf` + `electrode_wizard` + `format_models` | `_electrode_layout` via `as_structure()` | `prep.py:1528` is `as_structure`'s only caller | **DONE** |
| `molbuilder transport electrode` | `jobset prep` | a deck from flags — what `molbuilder fdf` was deleted for | **DONE** |
| `engine_base.py` Protocol + registry | a direct call | 4 declared members, 1 implemented, 1 engine, 1 caller | **DONE** |
| `results.py` + `sidecars/transport.py` + `parse/sidecars/transport.py` (697 lines) | `record.py` | `dump_transport_json`: **zero production callers in every revision**; the reader claimed only `schema_version`, the writer emits `schema` | **DONE** |
| `preflight_files` + `format_report` + `transport preflight` | `compose` + one resolved contract | gates compare two decks for drift two decks from one config cannot have | **DONE** (2c) |
| **`TransiestaEngine` + `TransiestaEngine.preflight`** | `_KIND_VALIDATORS["transport"]` + `sort`'s refusals + construction | **registered in `_ENGINE_VALIDATORS` under `TransportConfig`, and nothing validates a `TransportConfig`** — every rung resolves a `SiestaConfig`, and the two sites that build one build it as a projection for the lifted NEGF emitter. It stayed live only through `POST /api/transport/render`, deleted in 2c; every check it carried has a named holder, verified one by one | **DONE 2026-09-17** |
| `TransportConfig` behind `_legacy_view` | — | § 5p.3i: *"survives only to feed the lifted block"*; 3 orphan fields, all already in `UNRESOLVED_FIELDS` | **step 3, bookkeeping** |

**NOT redundant, so not deleted by association:** `extract_electrode_model` +
`ElectrodeModel` (`compose.py:570`); `_emit_transiesta_block` + `_emit_geometry`
(`deck.py` reuses both; § 5p.3i ruled the NEGF block stays — an off-by-one in an
electrode range converges while computing the wrong thing);
`_compute_cell_from_extents`, `_find_electrode_regions`,
`electrode_hs_stem`; `preflight.parse_fdf_params` (**six production import
sites**, re-measured 2026-09-17 — `citation_defaults`, `compose` ×2 symbols,
`parse/contract`, `web/blueprints/transport`; it said four);
`cell.detect_layers` / `bulk_z_period` — this same pattern **done correctly**,
moved out of the wizard so `add_slab` and the extraction share one copy.

#### 5p.3p.3 The Results gap is a MISSING LAYER, not three defects

The tab is a dispatch shell; all dispatch is client-side. `file-picker.js`
enumerates a directory, keeps what an `isResult` presenter matches, applies
`absorbs`, groups by `resultCategory`. Point it at a finished five-rung junction:

1. **The result is invisible** — `<label>.transport.json` matches no `isResult`
   presenter, so `pickResult` returns null and the picker drops it.
2. ~~**The ladder lists as a pile of optimizations**~~ — **MEASURED WRONG
   2026-09-18, and it was repeated all day before anyone checked.** The claim
   was that every rung's `.out` and `.molwatch.log` are claimed by
   `trajectory.js` under *"SIESTA optimization"*, beside the result.
   `file-picker.js` calls `projects.listDir(dir)` — **ONE directory, not a
   walk** — and transport runs `--shape hierarchical` (§ 3, the worked CLI),
   so each rung is its own folder. The calculation root holds the record and
   the rungs are one level down; they never appear beside it. *The claim is
   true of the FLAT shape, where stages share a directory, and was written
   without asking which shape transport uses.*
3. **`absorbs` cannot express a ladder** — it collapses siblings in ONE
   directory; a ladder is five directories that are one run.
4. **From the sidebar it is a text pager** — and by `presenters.md` § 1's own
   table, *a `.json` gets a plain text pane*. **The code conforms; the contract
   has no row for a composite result.**

**The cause is one door that was never built.** `model/parse.md` § 5 specifies
`RunDirResult` with `engine` (which engine ran), `files` (what is here),
`openable` (**what a VIEWER should load**) and `active` — § 5.1 draws exactly
the distinction this menu needs and calls conflating them *"the trap this
section exists to mark"*. The picker guesses all of it from filenames in JS
because **there is no server door to ask**: `/api/results/contract` answers
`info.calculation`, one of six fields.

**The Results picker is a SEVENTH consumer and § 5c's caller map does not list
it** — every row there is server-side. Transport is not the cause; it is the
first result that is neither a trajectory nor a spectrum, so the guess stops
producing a plausible answer.

**One question is genuinely new**: `openable` is one answer per DIRECTORY; a
ladder is five. Neither `absorbs` nor `RunDirResult` can say so. It reaches every
multi-rung calculation and needs a home before code moves.

#### 5p.3p.4 The steps — each one validates against its contract before moving

**Every step has the same three parts: the change, the contract it is checked
against, and what to do when the check disagrees.** A disagreement is not a
blocker to route around — it sends the step back to § 5p.3p.5.

**Step 1 — delete the results chain.** *(DONE 2026-09-17)*
*Contract:* `model/parse.md` § 4 (the registry), `job-contracts.md` § 6.1 (the
record's owner). *Check:* the parse registry builds and no longer lists
`transport-json`; `record.py` still writes. *Result:* passed — 163 tests in the
parse layer green; 697 product lines and their tests gone.

**Step 2 — delete the cross-deck comparison.** `preflight_files`,
`format_report`, the `transport preflight` verb; keep `parse_fdf_params` and
`_BOHR_ANG`. *Contract:* `engines/transport.md` § 5 (the invariant set) — read it
first and record, invariant by invariant, **which are now held by construction
by `compose` + one resolved config and which are not**. *Check:* every invariant
either has a construction-time guarantee named, or a reason it still needs a
runtime gate. *If the check disagrees* — an invariant with neither — the verb
does not go; the invariant gets re-homed first and § 5p.3p.2 is corrected.

**Step 3 — settle `TransportConfig`.** *Contract:* `engines/template.md` § 2.1a
(the two-homes debt) and § 5p.3i's open item. *Check:* re-measure the orphan
fields. *Measured 2026-09-17:* three (`log_level`, `num_threads`,
`transmission_relative_to_ef`), all already in `stages.UNRESOLVED_FIELDS` or
documented at `stages.py:110` — the shim leaks nothing. *So this is bookkeeping,
not a defect*, and the step is to record that, not to act.

**Re-measured after the `TransiestaEngine` deletion, same day — and the surface
got smaller.** `TransportConfig` now has **no validation role whatsoever**: the
`_ENGINE_VALIDATORS` row keyed on it is gone, so nothing anywhere validates one.
What remains is exactly two jobs, and both are real:

1. **the lifted NEGF emitter's input vocabulary** — built as a projection at
   `deck.py::_legacy_view` and `stages.py::config_for`, consumed by
   `_emit_transiesta_block`. This is § 2a.14's *"survives only to feed the lifted
   block"*, unchanged;
2. **the transport tab's form shape** — `web/blueprints/transport.py` renders
   `GET /api/transport/schema` from the dataclass and its
   `_form_section_order`.

So the retirement condition (TR6) is now stateable precisely: **`TransportConfig`
goes when the NEGF block is tabled AND the tab's schema comes from the
catalogue** — two things, both already named elsewhere, neither blocked on this
section.

**Step 4 — correct the engine contract** — **DONE 2026-09-17, and it was not
two lines.** *Contract:* itself, plus every document its § 6 is restated in.
*Check:* every symbol the Results section names resolves in the tree.
**The check found nine documents, not one**, and a shape problem under them:
§ 6's pieces table enumerated modules AND their function lists, so six of nine
rows named code deleted that week. Rewritten to name **roles**, with the
reasoning in the new § 6a — the rule being that a contract enumerates only where
the list itself is the guarantee (§ 5's thirteen invariants are; a pieces map is
not). Also corrected: `overview.md` § 5 (no engine registry exists — a new
engine is described and rendered through `spec_for`, it does not register),
`script-preparation.md` (its *"what this costs today"* block was entirely about
deleted code), `parse.md` § 7.9 (four absorption sites → two), `tabs.md` (both
"still open" items closed by deletion), `presenters.md` § 1 and `results.md`
§§ 2.3/3 (the six-vs-five count, and the ladder case), and the six documents
still naming `molbuilder pyscf`. `transiesta.py`'s class docstring and
`spectra/methods.py`'s module header both described registries that no longer
exist and were corrected in product code.

**Step 5 — the Results placeholder, and only that.** Register a transport
presenter matching `*.transport.json`, `isResult: true`, category `"Transport"`,
that reads the record through `ctx.readFile`, names it, shows the I–V table the
record already carries, and says what is **not** drawn.

> **Corrected at the step's own review, 2026-09-18.** This said *"states that
> the transmission surface is not built"*. Measured: every point in the record
> carries `energy_ev[]` and `transmission[]` beside `conductance_g0` and
> `current_a` — **the curve's DATA is there and only the CHART is missing.**
> Telling a person the surface "is not built" when the numbers are in the file
> they are looking at is the kind of claim this section keeps having to
> retract. Say the narrower true thing, and say how many points are in hand.
*Contract:* `web/presenters.md` § 2 (the presenter contract) — and **§ 1's table
gains a row in the same change**, because that table is the declaration of what
is presented. *Check:* the table's count matches the registry's; a finished
junction appears in its own menu. *If the check disagrees* — e.g. the six-vs-five
drift means the table has other gaps — fix the table wholly, not this row alone.

**Step 6 — kill the tests pinning the retired design**, with the code and not
before it. *Contract:* the admission question (default ZERO). *Check:* every
deletion names the product symbol it pinned. *Already done for steps 1 and the
June writers;* the audit of 2026-09-17 lists the rest, of which the ones that
matter are the tests that **cannot fail** (`test_siesta.py:559`/`:571`,
`test_pyscf.py:225`/`:235`/`:253`).

**Step 7 — `JobDirParser`, which is NOT this section's to do** — **the two
additions LANDED 2026-09-18**, having been written as *"made now and not
deferred"* on 2026-09-17 and then deferred twice while six other commits went
in. They are § 5c's caller-map row for the Results picker and § 5c.1 for the
ladder question. *The instruction to do it now was in this sentence the whole
time; I re-derived the same finding twice instead of reading it.* It belongs to
**§ 5c**. What this section owed § 5c: the **Results picker as a seventh consumer** with a server door to ask,
and the **ladder question** (five directories, one run), which § 5c does not
cover. Items 1–3 of § 5p.3p.3 close when § 5c lands and not before.

#### 5p.3p.6 The CLI surface — two survivors of a ruling already made

*(Added 2026-09-17 after the layer map was re-read top-down; the CLI was not on
the first three passes, which is why this is a new sub-section and the map in
§ 5p.3p.1 gained a row rather than this being a bullet at the bottom.)*

**The ruling exists and is general.** `conventions.md` § 3, decision 7 (user,
2026-08-11): *"everything is a job set. There is no `molbuilder run` — it is
deleted, not deprecated, because a second way in is a second way to lose your
results."* And decision 34, the same day: *"there is no `molbuilder fdf`"* —
user's words, *"obsolete residue from the flat-dir design"*.

**The verbs sort into four kinds**, measured 2026-09-17: structure construction
(`peptide` `dna` `rna` `smiles` `name` `modify`), the job pipeline (`jobset`,
`validate`), machine/ops (`envs` `serve` `jupyter` `checkpoint` `auth-setup`
`notify-token` `monitor`), and format utilities (`xv2xyz` `runtime-info`
`watch parse` `pseudo check`). **No calculation KIND has a verb** — there is no
`spectra`, no `optimization`, no `vibration`; a kind is described in `task.json`
and run through `jobset`. Two break the pattern.

**`transport` — a calculation-kind group.** `conventions.md` already names this
as a closed design problem: *"`jobset`, `bench` and `transport` each ran
calculations their own way"* (2026-08-11). `bench` got the full closure —
folded into `jobset`, **its four verbs deleted, the group removed**. `transport`
got half: the `bundle` verb went 2026-08-29, the group stayed. Its two remaining
verbs are the June hand-assembly workflow's UI: `electrode` *made* an electrode
deck from flags, `preflight` *checked* it against a device deck. **Both are that
era.** With `electrode` gone (§ 5p.3p.2) and `preflight` going (step 2), the
group is empty and follows `bench`'s precedent. *(User ruled 2026-09-17:
removing the transport verb is correct.)*

**`pyscf` — an engine verb, and `fdf`'s surviving twin.** Same shape: a
structure plus every engine field as a flag → a finished deck, skipping the
description. `cli.py` states the exemption itself — *"`pyscf` keeps its own only
because its ladder runs inside one emitted script"* — and **the same file
refutes it two comments later**: *"THIS COMMAND WRITES ONE DECK, AND A LADDER IS
N DECKS … there is no `--stages-json` / `--stage-strategy` here … `jobset init
--engine pyscf --stage-strategy …` is the one door that writes one."* The
exemption describes PySCF's *script shape*, not a property of the verb — and
`--stage-strategy` was taken off this command on 2026-08-18 for exactly that
reason.

The framework already covers it: `prep.py:651` builds a full `EngineSeam` for
PySCF, so `jobset prep` renders it through `spec_for` → `prepare_deck` like any
other engine. `render_script` is *"a thin call over `spec_for`"* and `convert`
is *"STEP 3, WHOLE, IN ONE CALL — the same call `prep` makes."* The verb
duplicates no mechanism; it is a second **way in**, which is the thing decision 7
names.

**Step 8 — delete `molbuilder pyscf`** — **DONE 2026-09-17, and it grew.** *Contract:* `conventions.md` § 3
(decisions 7 and 34). *What goes with it, measured:* `cmd_pyscf` +
`_make_pyscf_options_decorator` (~75 lines); **`add_dataclass_options`
(~168 lines), whose only production consumer is this command**; `pyscf.convert`
(no other caller); `pyscf.render_script` if nothing else points at it; and the
`test_cli.py` group that tests the BRIDGE rather than any behaviour. *Check:*
`conventions.md` § 3's roster and its count are corrected in the same change —
the table currently lists 13, names `serve` (now a group) and omits
`notify-token`, so the count is re-derived, not decremented. *The check found no consumer outside
`cmd_pyscf`, so the bridge went too.*

**The review found a TWIN nobody had noticed.** `siesta.input.convert` — the
same single-shot "read a file, write a deck" worker — **has had no production
caller since `molbuilder fdf` was deleted on 2026-08-11**. One month dead, in
the engine whose command went first. Both `convert`s are deleted here; the
symmetry is the finding, and it is why this step is filed as one change rather
than two.

**`render_fdf` / `render_script` STAY**, and the distinction matters: 43 test
files call them and *that is not the reason*. They are thin calls over
`spec_for` and `engines/siesta.md` / `engines/pyscf.md` name them as each
emitter's public surface. A test never justifies code; a CONTRACT does.

*Rosters re-derived, not decremented:* `conventions.md` § 3 said **13**
top-level commands, listed `pyscf`, and omitted seven others — it is 19 now,
measured from `cli.commands`. Both engine contracts and the index carry the
`convert()` tombstone.

**Step 9 — remove the `transport` group** — **DONE 2026-09-17 with 2c.**
`molbuilder --help` lists **19** top-level commands and no calculation-KIND
verb; `conventions.md` § 3 records the rule explicitly rather than leaving it
to be inferred from the roster. *(This line said 20. Re-derived from
`cli.commands` during step 4's review: 19, the same number step 8 recorded two
sub-sections above — a count written from memory one step after it had been
measured correctly.)*

#### 5p.3p.8 Steps 10 and 11 — the two fixes the 2026-09-17 sweep OWES *(added 2026-09-17)*

That sweep found nine documents wrong and corrected all nine. **Correcting them
is not a fix** — they were correct when written and decayed the same way the
next nine will. These two steps are the mechanism, and both are small because
the repo already proved the pattern.

**Step 10 — put the three surviving enumerations under a MEMBERSHIP assert.**

*Why these three and no others:* an enumeration earns an assert when a reader
ACTS on it — picks a route, adds a presenter, runs a command. Prose that merely
counts something does not (`test_doc_claims.MEASURED` already covers the counts
that matter).

| # | the list | assert it against | today |
|---|---|---|---|
| 10a | `web-api.md` § 3's endpoint index | `app.url_map`, **set equality**, `static` excluded | only the COUNT is asserted — and on 2026-09-17 it passed at 97 while the index carried a deleted route AND omitted a live one. Two errors, cancelling |
| 10b | `presenters.md` § 1's viewer table | the `register()` calls in `lib/inspectors/`, **set equality** on presenter name + `isResult` | nothing. The table said five viewers / three results for as long as `bench-summary` had been registering |
| 10c | `conventions.md` § 3's command roster | `cli.commands`, **set equality** | nothing. It said 13 when there were 19, and named a verb deleted that day |

*Model to copy:* `test_doc_claims.py::test_the_documented_L1_index_is_the_enforced_one`
— it reads the documented set out of the table, reads the enforced set out of
the code, and asserts **both directions** with a failure naming each side's
extras. It caught `ref`'s deletion on the first run after it.

*Done when:* deleting a route, a presenter or a command **fails a named test
that quotes the document and the line**, and each test is mutation-tested by
doing exactly that.

*The rule each one encodes:* **membership, never a count.** A count is satisfied
by any two errors that cancel, which is not a hypothetical — it is what 97 was.

**Step 11 — the deletion protocol, because no assert can catch what this
missed.**

Step 2c deleted `POST /api/transport/render`. That route was the **last live
caller** of `TransiestaEngine`, so the deletion silently made a whole class
unreachable — and for a day two contracts went on describing it as a live gate,
one of them calling it *"defense in depth"* for an ordering held by
construction. No membership assert would have caught that: the class was still
imported, still registered, still green.

**So when a step deletes a route, verb, module or public symbol, the same step
does three things — not the next review, the same step:**

1. **Name what the deletion makes UNREACHABLE one layer away.** Ask it
   explicitly: *what was this the last caller of?* `/api/transport/render` was
   the last thing that built a `TransportConfig` **and validated it**, which is
   the only reason `_ENGINE_VALIDATORS[TransportConfig]` still dispatched. The
   question was never asked, and the answer was a 316-line class.
2. **Sweep the documents for the identifier AND its prose name.** A symbol grep
   for `electrode_wizard` does not match *"the electrode wizard"*; `render_script`
   does not match *"the render endpoint"*; `engine_base` does not match *"a
   registered engine"*. **Every miss in the 2026-09-17 sweep was of this shape** —
   the symbol searches had already been run, four times, and each time the prose
   survived them. Search both spellings, or the sweep is not done.
3. **Re-run step 10's membership asserts**, which is one command once they exist.

*Done when:* the protocol is written into `process/code-audit.md` as a numbered
rule beside D1–D4 — it is a code-audit discipline, not a transport one, and it
is the third time this exact failure has been paid for (`molbuilder fdf` 2026-08-11,
spectra's registry 2026-08-21, transport's 2026-09-17).

#### 5p.3p.7 Step 2's review — the invariant set, one by one *(2026-09-17)*

`transport.md` § 5 declares thirteen invariants and says `transport preflight`
*"turns the prose Golden Rule into automated gates — the single biggest
correctness lever."* Step 2 proposed deleting that verb. **The review says
not yet**, and found a defect while checking.

| | invariant | held today by |
|---|---|---|
| I1 | XC functional + authors | **construction** — every rung resolves from one template |
| I2 | pseudopotentials | **construction** — the lead's atoms ARE the device's, extracted |
| I3 | MeshCutoff | **construction** — one template |
| I4 | PAO.EnergyShift | **construction** — one template |
| I5 | basis tier | **construction** — one template |
| I6 | lateral cell (a, b) | **construction** — `extract_electrode_model` takes the device's `lat_a`/`lat_b` verbatim |
| I7 | transverse k commensurate | **construction** — `citation_defaults` carries the transverse pair to both |
| I8 | device kz = 1 | **live gate** — `validation/__init__.py:528`, error |
| I9 | electrode kz dense | ⚠ **only `preflight.py:357-361`** |
| I10 | electrode geom = device frozen layers | **construction** — the extraction clones them |
| I11 | thickness ≥ principal layer | **`compose.py:596-640`, and BETTER** — it reads real orbital ranges from the citation's `.ion` files and refuses with the numbers. Its own comment retires the preflight's 12 Å floor as *"a GUESS made before anything was read … wrong about exactly the leads this measurement exists to judge"* |
| I12 | z-vacuum ≈ 0 at the leads | ⚠ **only `preflight.py:404-408`** |
| I13 | electrode writes its HS | **construction** — `TS.HS.Save` is a role item on the electrode rung |

**Eleven of thirteen are held without the verb, and one is held better.** Two
are not held at all once it goes.

**~~And I12 exposed a live defect~~ — RETRACTED the same day, 2026-09-17.**

I reported that `cell.vacuum_thin` fires on a transport device and advises
≥ 8 Å per side on the transport axis — the edit I12 says severs the lead. That
was **wrong**, and it is recorded here rather than deleted because the way it
was wrong is the point.

The check already filters on `axis_kind == "isolated"`, and `compose.py:777` /
`sort.py:326` both carry `axis_kind` through. Measured: the same atoms declared
as a junction (`periodic, periodic, transport`) produce **no vacuum finding at
all**. It fired only on a bare test fixture that carries no cell and no
`axis_kind` — a structure that declares itself an isolated molecule, which the
check then correctly described.

**I measured a fixture and reported it as the product.** Had the "fix" gone in,
it would have broken correct advice for every isolated molecule — nearly every
other calculation — to repair nothing. *(User caught the cross-task risk before
the edit: "make sure this is not conflicting with other tasks — these
suggestion is correct in those context.")*

What survives is smaller and real: **nothing requires a transport structure to
declare a transport axis.** `_validate_transport_kind` gates bias, net charge,
the pole energy, `tbt_k_grid` and `kgrid`, and not this. A junction with no
declared axes still renders — `_lattice_block` fabricates a vacuum box and says
so loudly IN THE DECK, but no validation Issue is raised, so nothing reaches the
form. That is a candidate gate, not a defect, and it is I12-adjacent rather than
I12 itself.

**Revised step 2, in three parts:**

2a. ~~Fix the vacuum advice~~ — **VOID**, the finding was retracted above. The
    check is correctly gated on `axis_kind` and needs no change. *Optional, and
    separable:* a gate requiring a transport structure to declare a transport
    axis, beside I8 in `_validate_transport_kind`. It is not a prerequisite for
    2b or 2c.
2b. **Re-home I9 and I12** — **DONE 2026-09-17.** Both now live in
    `_validate_transport_kind`, beside I8, keyed on the calculation KIND so they
    fire on every prep rather than on a command somebody remembers to run.
    I9 → `config.electrode_kz` (error at 1, warn below 20); I12 →
    `cell.transport_vacuum` (warn above 3 Å). Both thresholds are
    `preflight.py`'s own numbers, kept so the re-homing changed no verdict.
    *Check, passed:* seven tests in `tests/validation/test_transport_kind.py`,
    each carrying its discriminating half — the shipped `electrode_kz` default
    says nothing, a seamless cell says nothing, and **neither rule reaches a
    non-transport calculation**, which matters because I12 is the REVERSE of
    what `cell.vacuum_thin` tells an isolated molecule. Mutation-tested:
    disabling either gate fails its test. `transport.md` § 5's rows now name
    the live checks, and its "single biggest correctness lever" claim is
    corrected in the same change.
2c. **Delete the comparison and the verb** — **DONE 2026-09-17.** Gone:
    `Check`, `PreflightReport`, `preflight`, `preflight_files`,
    `format_report`, `molbuilder/transport/_cli.py`, and the `transport` group
    itself. Kept: `parse_fdf_params` (four production callers), `_parse_fdf`,
    `FdfParams`, `_BOHR_ANG` — reading an fdf and COMPARING two of them are
    different jobs, and only the second lost its subject.
    *Check, and it FAILED first:* § 5's table named `preflight.py`'s check-ids
    for eleven of thirteen rows, so "every invariant has a named holder" was
    false the moment the module went. Every row now names its real holder —
    seven **construction**, I8/I9/I12 `_validate_transport_kind`, I11
    `compose.py`. **Step 9 lands with this**: the group is empty, so it goes on
    the `bench` precedent, and `conventions.md` § 3's roster is re-derived
    (it said seven sub-groups and named `bench`, deleted 2026-08-17).

#### 5p.3p.5 The standing rule — consolidate, do not patch

**Four passes over this section have each CHANGED the answer rather than
refining it**, and every correction came from opening a contract the previous
pass had not:

| pass | what it said | what the next contract showed |
|---|---|---|
| 1 | *"no transport inspector exists"* | true, and the least of it |
| 2 | the picker drops the record and mis-files the ladder | `presenters.md` — the code CONFORMS; the contract has no row |
| 3 | three Results defects | `model/parse.md` § 5 — one unbuilt door underneath all three |
| 4 | the transport package | `conventions.md` § 3 — the CLI layer was never on the map |

That is the argument for the protocol below. It is not process for its own
sake; it is what four wrong answers cost.

##### The milestone review — run it BEFORE each step, and record the result

A step does not start until this has been done and its outcome written into
this section. It is five questions, top down:

1. **Re-read the contract that owns the layer being touched** — the row in
   § 5p.3p.1. Not the code first. The code is what the contract is checked
   against, never the source of the answer.
2. **Re-measure every claim § 5p.3p.2 makes about that layer.** A count in this
   plan is a hypothesis until re-derived. Anything that moved is corrected in
   place, with the date.
3. **Ask whether a LAYER is missing from § 5p.3p.1** — the failure that
   produced passes 3 and 4. The test: name the document that owns each layer
   the change touches; if a layer has no row, add it before proceeding.
4. **Check for drift** — does the code still do what the contract says, and does
   the contract still describe what shipped? Both directions. A disagreement is
   recorded as a finding here, not fixed silently in passing.
5. **If anything from 2–4 changed the picture, STOP and re-consolidate** —
   rewrite § 5p.3p.1 and § 5p.3p.2, renumber the steps if their order no longer
   holds. **Never append a bullet to the bottom.** A plan that grows by
   accretion is the thing this section exists to end.

##### The log — every review, recorded

| date | before step | what was re-read | what changed |
|---|---|---|---|
| 2026-09-17 | 2 | `presenters.md`, `model/parse.md` § 5, `conventions.md` § 3, `workflow.md` § 7, the contract index | CLI layer added to the map; Results gap re-framed as one unbuilt door; `pyscf` verb found obsolete by decision 34 (§ 5p.3p.6) |
| 2026-09-17 | **1, 2b, 2c — RUN LATE** | `model/parse.md` § 4, `job-contracts.md` § 6.1, `science/overview.md` § 4, `conventions.md` § 3 | **These three steps SHIPPED WITHOUT THIS REVIEW**, against the rule two sub-sections above. Run retroactively when asked whether the rule had been followed; the honest answer was no. Step 1: clean, nothing to correct. **Step 2b: the check catalogue (`science/overview.md` § 4) had NO transport section at all** — seven shipped checks uncatalogued, including the two that step added. **Step 2c: five documents still named the deleted verb, and one of them was a claim step 2b itself had falsified** — `transport.md` § 2a.13 row 2 said the warn "lives only in the standalone verb, so a composite run never sees it", hours after the re-homing that moved it. All corrected. *The reviews that ran on time (2 and 8) each changed the step; the three that ran late each found drift the step had introduced and left. That is the cost, measured.* |
| 2026-09-17 | **2 — STOPPED** | `engines/transport.md` § 5 (the 13 invariants), `compose.py`'s gates, `validation/__init__.py` | **The check disagreed and step 2 does not proceed.** 11 of 13 invariants are held by construction or by a live gate, and I11 is held BETTER (`compose` reads real orbital ranges); **I9 and I12 are held ONLY by the verb step 2 would delete**. I also reported a vacuum defect here and **RETRACTED it the same day** — I had measured a bare test fixture and reported it as the product; the check is correctly gated on `axis_kind`. See § 5p.3p.7 |
| 2026-09-17 | 8 | `conventions.md` § 3 (decisions 7 + 34), both engine contracts, `cli.py`'s own comments | **Found a twin.** `siesta.input.convert` has had no production caller since 2026-08-11 — the same shape as the PySCF one, dead a month, in the engine whose verb went first. Both deleted together. `render_fdf`/`render_script` kept: 43 test files call them, but the reason they stay is that the engine contracts name them, not the tests |
| 2026-09-17 | **4 — RUN ON TIME** | `engines/transport.md` § 6, `web/results.md`, `web/presenters.md`, `engines/overview.md` § 5, `execution/script-preparation.md`, `model/parse.md`, `web/tabs.md`, `process/conventions.md` § 3 | **The drift is an ENUMERATION problem, and it is not transport's.** Four hand-maintained lists were wrong the same week, each missed by the sweep that correctly fixed the *rules* beside it: § 6's pieces table (6 of 9 rows named deleted code), `presenters.md` § 1 (**5 viewers / 3 results declared, 6 / 4 measured** — `bench-summary` omitted entirely), `results.md` § 3 (*"the three viewers"*), `overview.md` § 5 (told a new engine to `@register_engine` against a Protocol deleted that week). Recorded as `transport.md` § 6a, which also states the rule that follows: **a contract enumerates only where the list IS the guarantee.** Two claims I had restated without resolving turned out never to have existed — `render_checks` on transport's engine base (its Protocol declared `render_script`/`parse_output`/`preflight`/`methods_fragment`) and `molbuilder siesta`. Step 8's sweep was **incomplete**: `engines/pyscf.md` carried a `convert()` tombstone AND, eight lines below it, the live `convert()` docs plus a runnable `molbuilder pyscf` example — six documents still named the deleted verb. One product-code orphan found and removed: `_transport_config_from_params`, zero callers since the route went in 2c |
| 2026-09-18 | **5 (the Results presenter) — RUN BEFORE** | `web/presenters.md` §§ 1–2 whole, `transport/record.py`'s record shape, the six `register()` calls, `web/results.md` § 2.3 | **The record already carries the curve.** Each point holds `energy_ev[]` and `transmission[]` beside `conductance_g0` and `current_a` — so step 5's instruction to *"state that the transmission surface is not built"* is too broad: the DATA is present and only the CHART is missing, which is a smaller and more honest thing to say. Step 5's text corrected. Also re-checked: `presenters.md` § 1 says six viewers / four results and the registry has six and four (corrected earlier today, still true), so the step's *"the table's count matches the registry's"* check is already satisfied and the row is a pure addition. No layer missing — § 5c's seventh-consumer row, which WAS the gap, landed 2026-09-18 |
| 2026-09-17 | **the fdf consolidation (N5a/b/d/f) — RUN BEFORE, on demand** *(user: "you are ignoring the rule we set for every step again")* | `engines/template.md` § 6.1 (`read_by`), `execution/gpu.md` G7, `execution/architecture.md` A8, `model/parse.md` §§ 1a + 4, `tests/test_layering.py`'s enforced sets | **It changed the step, which is the whole argument for the rule.** I had written N5d as *"two declarations on existing catalogue rows"* and was about to build it. § 6.1 says `read_by` **documents** a dependency and does not carry a value — G7 landed *"by carrying the answer"*, `resolve` putting `use_gpu` on `Resources`. Measured: neither `render_wrappers` nor `write_run_wrapper` receives the unsuffixed label, and **A8 forbids the loose kwarg** (that signature exists because eleven were removed 2026-08-17). So the label reaches the wrapper only via a new `Resources` field — identity riding a record that carries allocation/retry/notify — which is a **design decision awaiting a yes**, not a mechanical change. Also: § 5p.3p.2 said `parse_fdf_params` had *four* importers; it has **six**. Also: a LAYER was missing from § 5p.3p.1 — *what the wrapper is handed vs what it re-reads* — and it is precisely the layer that made the step non-trivial. **N5a (the move) is unaffected and stays first**, and it is what makes the cheap alternative to N5d possible |
| 2026-09-17 | **the § 5l retirement + a whole-document sweep — REVIEW RUN AFTER, and the user had to ask for it** | `plan.md` § 5l end to end, `architecture.md`, `science/validation.md`, `web/presenters.md` + `web/results.md` whole, `engines/transport.md` whole, `engines/pyscf.md` whole, `engines/overview.md` § 5, `execution/script-preparation.md` | **The finding is that SYMBOL-grep cannot see stale PROSE, and four passes of it had left the documents contradicting themselves.** § 8 of `transport.md` still shipped *"the electrode wizard"*, *"the `electrode`/`preflight` helper CLI"*, *"the render endpoint **remains**"* and *"arrives as a **registered engine**"* — every one deleted, none matchable by a symbol search. `results.md` § 7 told a reader to click a **Bundle** button § 5 of the same document records as deleted three weeks earlier. My own `presenters.md` edit that morning fixed ONE count and left **four** other places in that file saying five, and the presenter contract omitted `absorbs` entirely while claiming "two optional". **And the claim I had made that morning — that `ref.py` was named by no document — was FALSE**: `architecture.md`'s L1 index and `test_layering.py` both carried it as a bare `` `ref` `` in a list, which my grep for `molbuilder/ref.py` could not match. **The substantive find: `TransiestaEngine` was entirely dead** — registered under `TransportConfig`, which nothing validates; it had stayed live only through `/api/transport/render`, a route step 2c deleted without anyone noticing the consequence, and two documents were calling it *"defense in depth"* for an ordering held by construction. Deleted after verifying a live holder for every check. Two tests were holding the dead registration up; one pinned a `validate()` call **no production caller makes** and is gone, the other used `<=` so a dead row could hide and is now equality |
| 2026-09-17 | 6 (tests) | the repointed tests themselves, against the live deck | **A repoint is not a free pass.** Four checks were carried over from the deleted writer; three pinned deck TEXT with the emitter's column spacing baked in, and one -- `used-atoms` -- claimed to prove a count was "derived, NOT hardcoded" while asserting the literal `3`. Rewritten to parse the deck and compare against the structure's own regions; **mutation-testing then showed the rewrite STILL passed** against a hardcoded emitter, because the fixture's leads are 3 and 3. Now driven by an asymmetric junction (2 and 4), which is the only shape that can tell a derived count from a constant |

##### What "done" means for this section

**Five of the eight reviews logged above ran LATE**, and the pattern is now
measurable rather than anecdotal: *every review that ran on time changed the
step; every review that ran late found drift the step itself had introduced and
left behind.* 2c deleted `/api/transport/render` and thereby killed
`TransiestaEngine` without noticing — a day later the class was still documented
as a live gate in two contracts.

**The answer is NOT "run the review harder", and this section should stop
implying it is.** Five failures out of eight is not a discipline problem to
exhort away; it is a protocol asking for something at the wrong moment. A review
that runs *between* steps can only find what the last step broke, after it is
broken — and the reviewer is the person who just broke it, which is the worst
moment to ask.

**So the check moves INTO the step: § 5p.3p.8 step 11, the deletion protocol.**
Three questions asked while the deletion is being written, when the answers are
in hand — *what was this the last caller of?*, *what is this thing's PROSE name
in the documents?*, *do the membership asserts still pass?* Each of the three
corresponds to a specific thing this section paid for on 2026-09-17, and none of
them needs a review to be scheduled. **The review then does what a review is
for** — checking the plan against the contracts — instead of standing in for a
check the step should have made itself.

It is not "the steps are ticked". It is: every row in § 5p.3p.1 reads ✅, every
row in § 5p.3p.2 is DONE or has a recorded reason to stay, and the three
documents named in § 5p.3p.3 agree with the tree. Until then this section is
open, whatever the step list says.

### 5p.4 The steps

Ordered so each one is verifiable on its own and nothing depends on a later
step. **Every step ends with a check that can fail**, not with a declaration.

| # | step | reuses | done when |
|---|---|---|---|
| **TR1** | **Transport writes a template.** `jobset init --calculation transport` produces `<label>.template.toml` alongside `task.json`, its Class A values **defaulted from the cited relaxation** (§ 2a.7, ruling 1). The one adaptation needed: the description builder assumes a `StructureRef`, and a transport description has `slots` instead | `build_description` · `template_with_values(…, calculation="transport")` — already narrows the catalogue by kind | **✅ DONE 2026-09-16 — § 5p.3c.** the folder holds a template; `read_template` opens it; its values equal the citation's; changing one and re-prepping changes every stage's deck |
| **TR2** | **The three missing rows** (G8) — `TS.HS.Save`, the equilibrium pole count, `TS.Voltage` — added to the catalogue with their `SiestaConfig` fields | the one catalogue; `test_catalogue_agreement` | **✅ DONE 2026-09-16 — § 5p.3d.** the rows exist and agree with the fields; a device deck can state a bias from its description |
| **TR3** | **`kgrid` resolved for transport** (G9). The transverse grid and the transport axis stop being one row | § 2a.13's statement of the three classes | **✅ DONE 2026-09-16 — § 5p.3d**, as a CHECK rather than a row split. a description can set the transverse grid and the lead's axis separately, and the device's axis is not settable at all |
| **TR4** | **`prep` resolves transport through the one path** (G6). `_prep_transport` keeps the compose — genuinely new input — and hands to `resolve`, so a transport run has a `ParameterSet` with provenance | `resolve` · `effective_config` | **✅ DONE 2026-09-16 — § 5p.3e.** `--pipeline-log` produces a log for a transport prep; every value in a deck has a recorded source |
| **TR5** | **The remaining four rungs onto `spec_for`** (G7). **UNBLOCKED 2026-09-16** (§ 5p.3f): each rung renders the structure it describes — the lead EXTRACTED from the one cited structure by its label, the junction for the device — so the seam stays `(struct, cfg)` and `ElectrodeModel` need only yield a `Structure` | `transport/deck.py` (the seed arm, landed 2026-09-15) · `siesta/layout.py` sections | all five decks have a `.validation.txt`; all five pass the engine check gate; none is written by an f-string |
| **TR6** | **`TransportConfig` retires** (G2, G5). The shape is `SiestaConfig`; the projection introduced for the seed goes with it | — | `config/transport.py` does not exist; nothing maps one vocabulary to another |
| **TR7** | **The transport tab reads the catalogue** (G3). Its bespoke form schema is deleted; the parameter surface is the kind-aware catalogue route, and per-stage values are the **stage table** rather than a second invention | `GET /api/build/schema/siesta?calculation=transport` · the task-setup stage table | `dataclass_to_form_schema` has no callers; the tab shows one shared panel and a per-stage table |
| **TR8** | **Overrides route to the rung that owns them** (G4 — the live defect). The `stages` declaration landed 2026-09-16 (§ 5p.3b); the routing that reads it is what remains | the declaration, once approved | **✅ DONE 2026-09-16 — § 5p.3k.** setting the T(E) window reaches the transmission deck and nothing else |
| **TR9** | **Grouping** — the preparatory block as one submission (§ 2a.7). ⚠️ **Reconcile first** with `task-setup.md` § 1 (below) | `submit_transport_chain`'s shape — one submission walking a list | one command prepares and launches seed + both leads; the device and transmission stay separate |
| **TR10** | **Results reads the transmission** (§ 2a.12), with the treatment label and the provenance chain | `transport/record.py` | a curve carries how it was computed; a linear-response I–V says so |

### 5p.5 A contract tension TR9 must resolve, not ignore

[`web/task-setup.md`](?doc=web/task-setup.md) § 1 states, as contract:

> *"There is no 'run all stages' button, and that is deliberate. A stage is a
> long job, and a chain that continues by itself can spend a week refining a
> geometry you would have rejected in a minute."*

§ 2a.7's grouping ruling proposes seed + both leads as **one submission**. The
two are reconcilable and the argument must be written down rather than assumed:

* that rule is about a **ladder**, where each rung refines the last, and its
  harm is *continuing past a result you would have rejected*. Transport's three
  preparatory rungs are **independent** — none refines another, and there is no
  intermediate geometry to reject;
* it is one submission of three runs, **not** an auto-continuing chain. The
  precedent already exists and was accepted: the bias chain is one submission
  walking several runs;
* the thing the rule actually guards — *nothing starts the expensive stage for
  you* — is preserved exactly. The device still waits for a person.

**If that argument does not hold, TR9 is withdrawn rather than the rule bent.**

### 5p.6 Where each step's rows live

Per § 0, the open rows belong in § 2. This section holds the order and the
reasoning; § 2 holds what is outstanding.


---

## 6. Closed by consolidation — archived

*The provenance map of the nine archived plan documents and where each one's open items went. Several of the rows it points at have since been killed as untrue, so it is history: archived 2026-09-10.*

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

