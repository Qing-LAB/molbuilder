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

## 0a. THE WORK ORDER — milestones, each closed by a full code-text review *(2026-09-27)*

> *(user: "you need a consolidated persistent plan that get's updated at every
> milestone and reviewed with agent full code text review to validate
> results"; the order itself: "yes, that order works. go ahead")*

**This table is the order of work and its state.** The rows it names (W20,
W33–W39, …) keep the WHAT and the decisions; this table keeps WHEN and HOW FAR.
It is updated at every milestone, in the commit that closes it.

**How a milestone closes:**
1. its items are built — the contract first wherever an item changes a rule,
   then the code, then a test that drives the road (`jobset
   init/prep/launch/summarize`, or the page on a live server) and is broken on
   purpose once to watch it fail;
2. the targeted tests pass (the unit's own and what it touched);
3. **an agent reads the FULL code text** of the milestone's diff and of the
   contract sections it touches — not a grep — and reports each finding with
   its evidence;
4. every finding is re-read against the code before it is acted on, then fixed
   or answered;
5. this row records the commits, the review and its outcome — and only then
   does the next milestone start.

A full test batch (`tools/testrun.py`) runs at the end of M2 and at the end of
each programme after it, with nothing changing under it.

| # | milestone | items | done when | review | status |
|---|---|---|---|---|---|
| **M1** | W33 P3's last check | T3: export from Results → reload (§ 5q.4) | T3 passes, mutation-checked; W33's status line corrected — its dev-server check is done (2026-09-25) | **done 2026-09-27** -- read by an agent in full (verdict *yes with notes*), every finding re-read against the code: ① a second export route, the structure preview of SIESTA's own `<label>.xyz`, stated no frame, against § 6.0 -- **fixed at the framework**: the run's engine frame has ONE composer (`parse/dirs/atom_metadata.engine_frame_for_run_dir`, moved from `watch.py`), and the codec gives it to an engine's own file read where its run is recorded (`structure.md` § 2.4's invariant refined); ② the test pressed Export before MolView held its frames -- it waits for the viewer now (`support/results_export.py`); ③ a PySCF run's cell (the deck's record) was pinned by no test -- a case of its own; ④ the export offered " .log_frame6" and saved a hidden file -- the viewer names its install after the file it loaded (`web/molview.md`'s own rule); ⑤ this plan's stale rows -- fixed; ⑥ the reference now parses as the door does. Checked in a browser on the dev server on a project made fresh on both roads (`projects/claude-validate`: the web UI's New project, the CLI's `smiles`, `jobset init/prep/launch`) | **done** -- `45ff5089`, `164549ec` |
| **M2a** | the monitor says it started | W36 ① | the session log holds `starting` / `started`, and for a broken member file the error and no `started`; the test breaks a member, not the entry | **done 2026-09-28** -- read by an agent in full (verdict *yes with notes*), every finding re-read against the text: the start pair cited as run-reports § 2.3 lives in § 2.6 -- fixed in five places; the contract's reading rule had a false positive -- a run that ends inside the monitor's ~0.2 s load leaves `starting` with no `started` and NO error -- so the `ERROR` line is now the evidence of a failed load and the stopped-while-loading case is named (§ 2.6); the ending door prints the load error once per question -- job-contracts § 2.6 now has the row; the test pins the contract's line format, the `ERROR` line and the traceback (red with the traceback dropped); stale text in the launch block, `monitor.py`, `wrapper_log.py`, three test docstrings and run-reports § 2.5 / running-a-job's session-log row -- fixed; the test file renamed `test_monitor_start_is_logged_e2e.py`, since it holds both cases. **Open, for the user:** whether the wrapper should ask the bundle ONCE if it loads (the error printed once, and "the ending cannot be read here" said for it too) rather than once per question -- the review's option (b). Noted, not acted on: an unreadable zip, or an exception escaping `main`, reports in Python's own words, and a failing `ending` then exits 1, which the wrapper treats as 2 | **done** -- `e31e56f7`, `f3ae6e6b` |
| **M2b** | the small code fixes | W36 ② constants in the zip · ④ the ASE probe · ③ the GPU warning · ⑨ three silent fallbacks · ⑩ one `_mb_outfile` | each as W36 records it | | open |
| **M2c** | recipes and start-up | W36 ⑤ ⑥ | `ase` in the three job recipes (no install without the user's word); the conda listing and the CUDA probe at their use; the two comments | | open |
| **M2d** | one prep entry | W38 F7 | the page and the CLI call one prep entry returning its findings and decisions as data; the CLI prints and asks, the page shows and confirms | | open |
| **M2e** | the pipeline log, always | W20 | every prep writes it, from both doors; the flag is gone | | open |
| **M2f** | launched and finished | W38 F2, F3 | one door each, the same for flat and hierarchical, asked by every caller | | open |
| **M2g** | one restart-file list | W36 ⑧, the bias chain's list (W38) | every reader asks the calculation's list; Task setup states which is in effect; the shipped lists say how to customize | | open |
| **M2h** | the stage hand-over | W37, W38 F9 | the one contract section; the default, the warning, the explicit choice, one record read by all; *Continue from* on the page; `summarize` reads the record | | open |
| **M2i** | stage numbers, no on/off | W38 F4, F5 | numbers come from disk; no `enabled` (old files read by the agreed rule); a removed stage's files marked `.disabled` | | open |
| **M2j** | status owns the ladder | W38 F8, the queued attempt | every described stage in status; launched-and-unfinished attempts never hidden; the Results tab reads status | | open |
| **M2k** | what a job runs with | W36 ⑦, W38 F1 + the one queue record, F6 | one placement and one record; GPU request / claim / match; the precedence table in the contract | | open |
| **M2l** | the small two-door cases | W38 M1–M5, transport `restart` refused | as W38 records them | | open |
| **M2m** | customized parameters | W39 | the contract first; one API each side; the pane's section; the two hashes | | open |
| **M2n** | the text, then the batch | W36 ⑪ and the document sweeps left | the full batch green; then the Sol memory measurement and the sweep for tests that read text | | open |
| **M3** | the run record, the rest | W35: P2's remainder, P3–P6 | W35's own done-conditions | | open |
| **M4** | the engine offset, finished | W33 P4, P5 | § 5q.6; the fake-junction ladder resumes at rung 4 | | open |
| **M5** | transport | W27 floor 3 → W30 ③ → W25 → W24 → W10; W32 ②–⑤ once the single-frame ladder has run end to end | each row's own | | open |
| **M6** | charge and spin | W34, P1 on | § 5s | | open |
| **M7** | the spectrum view's API | W21 step 4 | W21's own | | open |
| **M8** | MolView sealing, the CSS | W15; W1–W6, W13 | each row's own | | open |
| **M9** | two science features | V1.26, V1.27 | each needs its decision first | | open |

---

## 1. The 2026-09-01 fact-check — ARCHIVED

*Nine plan documents read against the code; three headers were flatly false. The detail is history now and lives in [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md). The one live fact it found — bench § 2.2's `parse_util_bound` reads the VERDICT and not the numbers, so the item is open — is a row in § 2.*

---

## 2. OPEN — the one list

**Every open item, in one table** *(consolidated 2026-09-10 at the user's instruction: "consolidate plan and to do, archive finished things and untrue things, clean it up so we have one list")*.  §§ 3, 4, 4a and 5 held four separate tables of the same kind of thing and are now pointers — the numbers stay so links from other documents still land.

**Rows that were done, withdrawn, or measured untrue are NOT here.** They are in [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md), with what each one turned out to be. Read § 5a before acting on any row below: a row is evidence of when it was written.

> **The two decisions this box held are taken.** **D-1** — what declares that
> a value binds every rung: decided 2026-09-24, the sibling marker `shared`,
> and built (`engines/transport.md` § 3.8.6). **D-2** — where a mode's activity
> is classified: decided 2026-09-23, the classifier as built is the rule and
> `top_n` / `threshold` retire under V1.6 (`engines/vibration.md` § 6.6).

| # | area | item | from | state |
|---|---|---|---|---|
| **E1** | engine / science | **Benchmark iteration count, settable per calculation.** No field exists on `task.json` or `Resources` — confirmed 2026-09-07. **Bigger than the row says:** the archived design was a one-point `bench` entry overriding the pin, and that path is now explicitly closed — `_cli.py:1961` `pins = {**declared_pins, **_MEASUREMENT_PINS}`, with the comment that measurement pins *"must win over any declaration — one-point declarations and value-axis coordinates alike."* So this needs a written precedence rule reversed, not a field added | `bench-and-junction` § 2.1 | not started |
| **E7** | engine / science | **D7's cluster half** — the prep→submit→watch loop for a **run** through SLURM on Sol. **2026-09-10, the user: "we have tested it on Sol" — and the row must not contradict that.** What is actually on disk here: Sol slurm records exist (`optimization/sol/AuBDTAu-slabcorrected/01_coarse/bench/launch/slurm.62380919.out`, real ids) and every `kind: "run"` ledger entry in `projects/` is `workstation`/`direct`. That is the absence of a LOCAL record, which the earlier wording ("No `kind: run` submission exists on any Sol tree") presented as the absence of the WORK. A tree on Sol is not in this repo. **This row needs the user to say whether it is closed, not another file scan** | roadmap § 1 | needs the user |
| **E10** | engine / science | **NEEDS A RE-SCOPE, not implementation — re-derived 2026-09-07.** The detection is **built**: `parse/engines/_run_ending.py` is the one marker table (`abnormal_termination` → `stopped`, OOM markers, `SCF_NOT_CONV`), `scan_ending()` returns `(run_state, scf_converged, error_message)`, and `summarize.py:217` + `cli.py:2771` consume it — the zero-exit case being the whole point. What is NOT built is the monitor half, and **deliberately**: `monitor.py:138` says *"NO COMPLETION MARKERS HERE"* and `job-contracts.md:214` now rules the monitor follows the launcher's PID *"rather than guessing from output markers"* — which contradicts this row's own "belongs in `mb_monitor.py`". The real gap is narrower: **the monitor's finish report carries no convergence fact.** Decide whether it should before writing anything | roadmap § 4 | open (re-scope) |
| **T1** | engine / science | **A LIVE POLL THAT MUST REBUILD DOES NOT UPDATE THE MOVIE — found 2026-09-07.** The append path is fine; the full-rebuild path is not. Reproduced: move the frame at `oldLen - 1` (the one `_frameEqualAt` reads) and grow the feed 4 → 6, so `canAppend` refuses and `applyNewData` takes the `else` branch. The status line then says *"Loaded 6 … frames"* and the frame bar still holds **4** — the feed's count and the movie's disagreeing, which `core.js` itself names as bug **#35**. The trigger is ordinary: a frame that was still being written when the last poll caught it, and has since settled. **Two things hide it, each worth its own look:** `setStatus` is a no-op on `/results` (`if (!document.getElementById("status")) return;`), so `rebuildModel`'s *"Viewer failed to load the run"* reports into nothing on the page the inspector lives on; and *"Loaded N frames"* is written by `applyNewData` from the FEED's count while `rebuildModel` runs unawaited beside it, so the tab can claim frames it is not showing. Recipe at the foot of `tests/test_inspector_registry_e2e.py`; it blocks the last three source pins in `test_structure_info_bridge.py`, which is how it surfaced | found 2026-09-07 | open |
| **E13** | engine / science | **The first real junction walk — BDT–Au, workstation then Sol. DROPPED BY THE CONSOLIDATION.** Named in the roadmap and again in `transport-design` § 7's *"order of proof"* as the run that follows P6. plan.md has the machine-blocked infrastructure (E5/E7) and the browser and deck walks (W12/E11) but no row for the transport composite's first real science run | roadmap · `transport-design` § 7 | open |
| **E14** | engine / science | **Two bibliography keys — Reed 2006 and Stokbro 2003 — are cited nowhere in `docs/science/references.bib`.** Verified absent 2026-09-07. Mechanical, small, and dropped by the consolidation | roadmap § 4 | open |
| **V1** | engine / science / web | **The vibration calculation — one path on two engines.** The contract is [`engines/vibration.md`](?doc=engines/vibration.md) (§ 10 says what stands); the science is [`science/normal-modes.md`](?doc=science/normal-modes.md); the design and the first audit are archived (`archive/2026-09-24-*`). Built 2026-09-21 → 24: one harmonic path with the rank rule and its gate against PySCF, one mass convention, stationarity on the free atoms, the runs write the pair, the Hessian over the free atoms with the two measured corrections, the SIESTA arm (deck on a sorted copy, `atom-permutation.json` with its key, the `.FC` reader, `summarize run`, the warm-file section, the kind's start state), the equilibrium block optional, the wrapper's banner. **Open, the sub-rows:** | `engines/vibration.md` § 10 | open |
| ↳ | ~~**V1.0**~~ | **DONE 2026-09-24**: the file's geometry is the Hessian's (`positions_ang` from `COORDS_EQ_ANG`); the thermochemistry headline and grid as one quantity with the headline T on the grid and no `kT` in the held regime; `raman_route` / `raman_fd_step_ang` and the Methods text stating the Raman method one way; the reader's unknown-key gate and the partition check without a range the size of a lie; `THERMO_GRID_K` in one home. Verified one run at a time through jobset — the SIESTA road, all seven PySCF runs, the CO₂ read-back — and the 1006-test unit batch; two forward-compatibility tests asserting the old ignore-unknown-keys rule retired into the gate test | `vibration.md` § 4.2, § 4.6, § 4.7, § 6.7 | done |
| ↳ | ~~**V1.1**~~ | **DONE 2026-09-24** — **The Spectrum tab offers both engines**: an engine strip over two catalogue-built forms (the Structure-optimization tab's component; the one-choice `engine` item retired the same day — a real choice is not a parameter of the deck), each form from `/api/build/schema/<engine>?calculation=vibration`, the live checks and the hand-over sending the strip's engine, and `web/blueprints/build.py`'s hand-over gate admitting SIESTA for a vibration (the CLI road is open since 2026-09-23) | `vibration.md` § 3.1, § 10 | done |
| ↳ | ~~**V1.2**~~ | **DONE 2026-09-24** — **Task setup prints `--target` when a machine is chosen** (measured missing on the 2026-09-23 UI walk though the page's card said *Prepared for (this machine)*) **and, for a SIESTA vibration, `summarize run <stage>` as the last step** the rung tab teaches | `vibration.md` § 2.1; `web/task-setup.md` § 10–11 | done |
| ↳ | ~~**V1.3**~~ | **DONE 2026-09-24** — **The Results viewer draws a SIESTA file**: `null` MO block and `null` intensities as *not computed*; the equilibrium energy as a dash (today `Number(null).toFixed(8)` prints `0.00000000`); the Raman line said by route, not by `phase_raman`; the change fingerprint off the unwritten `phase_ir`; `removed_motions.count` and `hessian_scope` shown beside the result (R7's second half); the relaxation warning shown when the phase is disabled | `vibration.md` § 6.5, § 10; `web/spectra.md` § 9b.3 | done |
| ↳ | ~~**V1.4**~~ | **DONE 2026-09-24** — **The force unit**: `relaxation.max_force_eh_a` is Eh/Bohr; the key's `_a` and the viewer's "Eh/Å" are wrong — rename the key and the label together (the key is the deck's) | `vibration.md` § 4.3 | done |
| ↳ | ~~**V1.5**~~ | **DONE 2026-09-24** — `not requested` is the fourth, terminal phase state: the PySCF deck writes it for a Raman sweep not asked for and for the probe under `skip`, the SIESTA writer for the relaxation, the strengths and the probe; the viewer counts it finished, draws its chip and prints it. Was: **The phase flags gain a *not requested* state**: a run that asked for no strengths writes `phase_raman = 'complete'` on both writers, and the SIESTA file writes every phase complete | `vibration.md` § 4.9 | done |
| ↳ | **V1.6** | **Retire `top_n` and `threshold`** (decided 2026-09-23): the config field and its catalogue rows, `spectra/selection.py`, `spectra/methods.py`, `validation/spectra.py`, the emitter's ranking in `pyscf/vibration_emitters.py`, the tab's lock map in `lib/spectra/core.js`, and their tests — the count must come down | `vibration.md` § 4.8; `web/spectra.md` § 9a.1 | open |
| ↳ | **V1.7** | **`temperature_K` and `pressure_atm` reachable on SIESTA** (today the derivation sums at 298.15 K, 1 atm and says so; the items are PySCF's) | `vibration.md` § 5.5 | open |
| ↳ | **V1.8** | **The Methods paragraph reads the effective level of theory**: `_paragraph_vibrational` names `cfg.functional` and `cfg.dispersion` whatever `cfg.method` is, so a Hartree–Fock run's write-up cites B3LYP-D3BJ while its next sentence names `pyscf.hessian.rhf` (measured 2026-09-22 on a deck with `mf = _scf.RHF(mol)`, no `mf.xc`, no `mf.disp`); one function returning method + functional + dispersion + basis, consulted by the SCF construction and the prose, plus an advisory when a functional or dispersion is set under HF | `vibration.md` § 4.10 | open |
| ↳ | **V1.9** | **The structure identity rule** (settled in discussion 2026-09-22, written into no contract yet — this row is its full statement until it has one): **two hashes, one home** — a *geometry hash* over the atom lines as **verbatim text** (the numbers are already text, so there is no float question) plus the lattice and the per-axis periodicity from the sidecar, with `none` as a real hashed value when there is no lattice; and a *broad hash* over labels, regions and identity columns, for provenance. **Never the comment line** (a human title from one writer and a derived `Lattice=` from another; adopting it would promote a derived value into a stored one). **Out:** the cell origin (a gauge choice — shifting it moves no periodic image), labels, timestamps, the job name. **Joins check the geometry hash only**; a broad-hash difference is reported, never blocking. **Minted at three gates** — Modify → Save to project, Results → export, the CLI on request — then **carried and never recomputed**; **checked at three points** — prep, results load, joining two runs. **Derived structures** (a reordered copy) mint their own identity and record the parent's plus the permutation. **A run's output is a record, not a gate** — it becomes an input only by passing through one (ruled 2026-09-22). Today: `spectra.json`'s `structure_hash` has the job name as line 2 (the atom count is line 1) and never matches the codec's pair hash, which is over the document's bytes — a different scheme; and `workingcopy_structure.py:265`'s `keep_sidecar = (not _metadata_is_default(meta))` writes no companion for all-default metadata, so *Save to project* can still emit a bare `.xyz` at the very gate meant to guarantee the pair (both measured saves got one only because they carried labels) | `vibration.md` § 4.3 | open |
| ↳ | **V1.10** | **The pair writer renders both halves** (ruled 2026-09-23, the unification audit § 1.1a): `pair()` returns text, the deck splices the codec's own JSON (`molstruct.dumps`); the deck's sidecar writer is the third serialiser today | unification audit § 1.1a | open |
| ↳ | **V1.11** | **Untested and open**: the reduced Hessian with a GPU mean field; a composed permutation for a structure sorted for two reasons (ruled one record, no caller yet); `transport/compose.py` writing and reading its record through `write_permutation` / `read_permutation` and stamping its key (U10) | `vibration.md` § 4.4, § 5.2 | open |
| ↳ | **V1.12** | **Not done**: a mode-by-mode intensity cross-check against an external code (Gaussian / ORCA / Turbomole); absolute intensities carry the caveat until then | `vibration.md` § 9 | open |
| ↳ | ~~**V1.13**~~ | **DONE 2026-09-24** — **Labels that name one engine**: `presenters.md`'s *PySCF spectrum*; the Spectrum tab's own engine sentence and the `engine` item's help — with V1.1 | `web/presenters.md` | done |
| ↳ | **V1.14** | **From the UI walk of 2026-09-23, not this kind's**: the Molbuilder tab's save prompt doubles a typed suffix (`x.xyz.xyz`); the `#`-labelled provenance regions (`O#`) warned as unconsumed on every SMILES-built molecule — needs a ruling; the vacuum notice advising 8 Å on a gas-phase PySCF run | unification audit § 1.18 U3, U4, U7 | open |
| ↳ | **V1.16** | **A release note**: every held-atom spectrum and free energy computed before 2026-09-23 contains a non-vibration (the surviving turn was reported as a mode and, on the sign of noise, summed into the entropy); old and new runs disagree, and that is intended | `vibration.md` § 10 | open |
| ↳ | **V1.17** | **Four held systems through the whole road**, today rank rows only: acetylene with both carbons held (the collinear trap end to end), NH₃ with its three hydrogens held (nothing removed), an empty held list reproducing the free path *exactly* (what makes "the free case is the held case with nothing held" a fact), the water dimer with one molecule held (the over-removal guard) | `vibration.md` § 9 | open |
| ↳ | **V1.18** | **`_mode_count`'s results arm has no production caller**: the composer renders before the run with `results=None`, so the arm that would interpolate the run's own list and frequency span is exercised by its own fixture only — wire it into the load path beside `with_ir_route`, or delete it | `vibration.md` § 4.10, I13 | open |
| ↳ | ~~**V1.19**~~ | **DONE 2026-09-24 — the browser walk, both engines, UI → Task setup → `prep` → `launch` → (`summarize`) → Results**, water with O held on PySCF and H₂ with one H held on SIESTA, each through the printed verbs on this machine. Five defects found and fixed on the road: the Spectrum form never loaded (the generate side was gated on an element the two-panel rewrite dropped); the six electronic-structure-probe selectors sat on the *Convergence targets* card and were offered as *vary per stage* (now `profile`, `vibration.md` § 3.1); the SIESTA engine validator fired its region-label and *held during relaxation* notices on a vibration beside the kind's own (one fact, one finding — `science/validation.md` § 7); the PySCF kind's held-atom findings pointed at the retired `config.frozen_indices` (now `structure.regions`); Task setup's pickers fell back to SIESTA on a PySCF hand-over and its three vocabulary caches raced, so a PySCF description was seeded a rank sweep its own preflight refused (`web/task-setup.md` § 6.1) | `vibration.md` § 9; `web/task-setup.md` § 6.1 | done |
| ↳ | ~~**V1.20**~~ | **DONE 2026-09-24** — the catalogue gained `recommended = { <kind> = … }` (`template.md` § 6.3a): the form-schema door, `jobset init` and `template_with_values` all present a kind's recommendation as the item's value and default, so the vibration template carries the tight tier (`geom_gmax` 2·10⁻⁴; `relax_force_tol` 0.01 eV/Å) as a value the person can change, and the claim is true everywhere it is stated. Was: **Needs a decision — the tight-tier claim**: `vibration.md` § 3.1 and § 4.2 said the vibration template *defaults the geometry criteria to the tight tier*, but the catalogue carried one default per item and no kind-keyed default existed, so the template the hand-over and `init` wrote carried the catalogue's 4.5·10⁻⁴ | `vibration.md` § 3.1, § 4.2; `engines/template.md` § 6.3a | done |
| ↳ | ~~**V1.21**~~ | **DONE 2026-09-24** — the forces at FC step 0 read into `relaxation.max_force_eh_bohr`, `converged` against the description's own `relax_force_tol` (the catalogue's general default stood in for it until later the same day), the warning printed and shown; `already_relaxed` offered on SIESTA — a refusal while unmade at first, the person's explicit choice by the end of the day (§ 2.2). Was: **The SIESTA route judges stationarity** (R5 holds on PySCF only): read the forces at `FC step = 0` from the run's output into `relaxation.max_force_eh_bohr` over the free atoms and warn above the optimization's own criterion. Measured 2026-09-24: the H₂ fixture at 0.741 Å carried 1.27 eV/Å on its reference step through the whole road and nothing was said | `vibration.md` § 5.5; `science/normal-modes.md` § 4b.6 A | done |
| ↳ | ~~**V1.22**~~ | **DONE 2026-09-24** (recorded as `engine_metadata.fc_asymmetry_max_ev_ang2`; no threshold; on an axial block the off-diagonals vanish by symmetry). Was: **The asymmetry diagnostic**: `hessian_from_fc` symmetrises `½(B + Bᵀ)` and forgets `max abs(H_ij − H_ji)`, the number that says first whether `FC.Displacement` was too small or too large; record it beside `fc_displacement_ang` and warn above a stated fraction of the block's largest element | `science/normal-modes.md` § 4b.6 C | done |
| ↳ | ~~**V1.30**~~ | **DONE 2026-09-24** — one rule on both routes, by the ruling of `vibration.md` § 2.2: the largest absolute force component over the free atoms against the template's own tolerance (`geom_gmax` on PySCF, `relax_force_tol` on SIESTA), a plain warning above it. Was: **One stationarity rule for both routes** (needs a decision): PySCF warned above ten times `geom_gmax` on the largest force component; SIESTA judged once the catalogue's recommended tolerance | `vibration.md` § 2.2, § 4.3, § 5.5; `science/normal-modes.md` § 4b.5 A | done |
| ↳ | **V1.23** | **A δ-convergence report** (needs a design): two stages at δ and δ/2 are describable today (`fc_displacement` is a stage item); which verb prints the comparison of `ω_ν` and `e_ν` between them | `science/normal-modes.md` § 4b.6 C | open |
| ↳ | **V1.24** | **Mode matching across runs** (needs a design): the overlap of eigenvectors in the shared free subspace, mass-weighted — the active-region convergence test (Models A/B/C) and the PySCF-against-SIESTA comparison of one molecule are the same calculation | `science/normal-modes.md` § 4b.6 F, § 4b.3 | open |
| ↳ | **V1.25** | **The mode-displaced frame set** *(decided 2026-09-24, user)*: a generator that writes ONE multi-frame pair — the base as frame 0, then `R_A(Q) = R_A⁰ + Q·L_canonical` with `R_F` unmoved at the zero-point amplitude and its thermal growth — from a spectra file, for a named mode; the rule (run, mode, amplitudes) recorded in the pair's `info`. It is the interface to transport's frame axis (W32); a person's own script writes the same pair. Level one of `vibration.md` § 5.6 | `engines/transport.md` § 2a.9 · `model/structure-molstruct.md` § 6.1 · `science/normal-modes.md` § 4b.6 G | **contract written 2026-09-24**; generator open |
| ↳ | **V1.26** | **`Δρ_ν(r)` maps** (needs a decision): SIESTA's density grid at `±Q_ν`, differenced — the discussion's intermediate quantity before any oscillator strength on a metal | `science/normal-modes.md` § 4b.7 | open |
| ↳ | **V1.27** | **The PySCF probe** (needs a decision): five points per mode instead of two; the coupling per zero-point amplitude `g_ν = ∂ε/∂Q · √(ħ/2ω)` in meV written beside `ΔE/(2A)`; a molecule-projected frontier quantity for a cluster whose HOMO and LUMO are metal states | `vibration.md` § 4.8; `science/normal-modes.md` § 4b.5 F–G | open |
| ↳ | ~~**V1.28**~~ | **DONE 2026-09-24** — the record is a cluster of the structure's `info` store, `info.relaxation` (`model/parse.md` § 5b.1: engine, the run's force tolerance, the largest force left on the moved atoms, the held set, the final geometry's fingerprint), composed by `run_info_for_dir` beside `info.calculation` and recorded by the Results tab's structure inspector; both kinds' gates read it on the box's card (`validation/sidecar.py::check_relaxation_record`): absent and ticked is accepted with a hint, present is checked against this calculation's tolerance, level of theory and held set, never a refusal. Found on the way and fixed: `result_roles` offered the seeded molwatch log before SIESTA's `.out`, so every SIESTA relaxation opened its 613-byte seed on the Results tab. Was: **The structure carries its relaxation record** (needs a design and a yes) | `vibration.md` § 2.2, § 10; `model/parse.md` § 5b.1 | done |
| ↳ | **V1.31** | **The PySCF deck's own `_optimized.xyz` pair records `info.relaxation`**: the deck knows its criteria, its convergence and its held set when it writes the pair; the spliced pair writer would compute the fingerprint the same way (`Structure.geometry_fingerprint`, spliced like `structure_hash_text`). Until then a PySCF-relaxed structure carries the record only when exported from the Results tab | `vibration.md` § 2.2, § 10 | open |
| ↳ | **V1.33** | **The Task setup tab re-derives the vibration ladder in JavaScript** (`proposedFromHandover`, `_afterRunLines` spell `relax`/`freq` and the box rule by hand) while Python owns it (`pyscf/stages.py::vibration_stages`). A second copy of one rule; the hand-over or the folder answer should carry the proposed ladder computed server-side, and the after-run line should ask the description which rung is the force-constant one. Found by the 2026-09-24 review | `vibration.md` § 2.2, § 5.2a | open |
| **W33** | structure / engines / web | **THE ENGINE OFFSET — one placement rule for every engine** *(user, 2026-09-25: "always adjust it before sending to siesta or other engines that the coordinates of all atoms are centered inside the cell"; "we don't have to have special logic to treat isolated, periodic, transport axis_info differently")*. Every engine receives the design coordinates + `engine_offset` with the cell at the origin; the offset is computed from the cell and every atom (the fractional span centred, never re-wrapped), recorded in every deck, and stated 0 for an engine's own output, so the Results tab draws what the engine had. Retires `cell_origin` (an origin the person assigns is kept, stored as the offset), the derive-on-null corner and its per-axis rules, calibrate, and the two hand translations. Found by the fake-junction ladder: a TranSIESTA device refused on a flush corner, a Results box the engine never had, the molwatch step-0 jump (audit X4) | `model/structure-periodicity.md` § 6.0 · § 5q | **P0–P3 done** (P3's T3 on 2026-09-27, `tests/test_siesta_run_in_the_viewer_e2e.py`; its dev-server check on 2026-09-25)**; P4, P5 open** (2026-09-25; § 5q.6 is the per-phase record). **D1–D16 settled** (§ 5q.8). The fake-junction ladder is paused at rung 4 until P5 |
| **W34** | science / engines / web | **THE ELECTRONIC STATE — charge and spin as one answer per calculation** *(user, 2026-09-25: "put spin and charge setup in to a unified framework so that these information can be produced consistently and systematically for different engines"; "investigate holistically ... from template to validation ... a design gap/framework level investigation rather than a patch"; "documentation should have an explicit discussion on species with these properties")*. Four engine-neutral template items — `net_charge`, `spin_treatment`, `unpaired_electrons`, `method` — resolved once with what the structure adds (the charge's source, the electron count, finite or repeating) and read by every deck writer, check, hand-over, form and read-back; restricted-open explicit; a species-by-species chapter. Found by the transport ladder's rung reports (a gold lead and a formate ion both told "switch to open-shell") | `science/chemistry-correctness.md` §§ 2a–2b · § 5s | **P0 open** (2026-09-25): decisions 1–7 taken (§ 5s.2) |
| **W35** | parse / execution / web | **THE RUN RECORD — what ran, with what, and how it went, for every run; one SIESTA-family reader; the transport report** *(user, 2026-09-26: "include in the report of transport results ... all indicators that can provide scientific information, progress tracking and symptom of non-convergence"; "parameters used for calculating transport should be similarly reported ... check what we did for the other tasks, and find the best way to honestly and fully record what is the computation setup and scientific setup, and how result evolves in the record, and output report"; "fdf parser need unified upgrade"; "get the framework and code unified and finalized")*. One record per attempt, composed on read — computation, setup (default · asked · engine used), the deck, every iteration of every phase from one grammar, the verdict with its symptoms — read by a Run panel for every kind and by a transport report built from its rungs; the NEGF contour stated rather than left to TranSIESTA's fallback. Found by a device that diverged for eleven hours with every reader saying otherwise | `model/parse.md` § 5d · `engines/transport.md` § 2a.12 · § 5t | **P0–P1 done (`a4c1d28e`); decision 10 done — the monitor, the wrapper's endings and the report fields through the framework; P2: the record and the Run panel built (2026-09-27), the rest below** |
| **W36** | execution / packaging | **THE RUN-INDEPENDENCE REVIEW (2026-09-27), agreed item by item before any is fixed** *(user: "i need to go through all of them so we have agreement before i let you just go ahead to fix all")*. The review's finding: a job needs nothing from molbuilder at run time -- wrappers, decks and the monitor bundle checked. **Agreed:** ① *(built 2026-09-27, M2a)* *a monitor that fails to load says nothing* -- `runwrap._BUNDLE_MAIN` catches the import error and exits with no word, so a broken member file leaves no trace even with the monitor's stderr in the session log, and `_mb_ending` answers "cannot read" silently (no warm retry, no hint). The fix: the entry prints the error and its traceback, and writes a pair into the session log in `_log`'s line format -- *starting* (with the python it runs on) before the import, *started* after it -- so a missing half is the failure *(user: "your recommendation plus a more complete log")*; the test breaks a member file rather than the entry. ② *two constants typed by hand in a carried file* -- `molwatch_grammar.py` spells `HARTREE_EV` and `HARTREE_BOHR_EV_ANGSTROM_ASE` as literals because it travels in `mb_monitor.pyz`; `architecture.md` § 3 allows a second spelling in three places and this is none of them. The fix: `constants.py` (it imports nothing) joins `MONITOR_COMPANIONS`, the grammar imports it two ways like its `end_lines` import, and the literals go *(user: "yes, ship constants in the zip")*. ③ *a GPU warning that checks the wrong place* -- `validation/spectra._gpu_capability_advisories` imports `gpu4pyscf` / `cupy` in the server's env, where they never live, so it fires on every GPU vibration and advises a pip install into the wrong env with a hand-typed CUDA tag; G-5 (`engines/overview.md` § 3a) checks PySCF's GPU at run start. The fix: the advisory and its call go, and `test_vibration_render_gate.py::test_the_gpu_advisory_path_renders_and_speaks` with them *(user: "yes, delete it")*. ④ *an ASE import that guards nothing* -- `siesta/input.py:26-32` imports `ase.data` / `ase.io` as a probe it never calls, so every import of the deck writer (and, through `validation._register_default_engines`, everything that validates) loads ASE's readers and SciPy (`ase.io`: 0.5 s of the module's 0.8 s). The fix: the probe goes, validation takes `SiestaConfig` / `PySCFConfig` from `config/`; ASE stays a dependency, imported where it is used *(user: "ok, but i also suggest to add ase into molbuilder-siesta and pyscf env for future proof use" -- the env half is ⑤)*. ⑤ *ASE in the job envs, for later use* -- `ase` (conda-forge) joins the `_SIESTA`, `_SIESTA_GPU` and `_PYSCF` recipes, so switching the GPU on or off never changes what a job can import; it brings `numpy`, `scipy`, `matplotlib-base` (the SIESTA envs have none today). **Recipes only**: installing into the envs here (`conda install -n <env> -c conda-forge ase`) waits for the user's yes, never during a test batch; Sol is the user's to run *(user: "yes, include to both")*. ⑥ *two slow checks on every command* -- `cli.main` runs `diagnostics.initialize()` (`conda env list --json`, 0.7 s here, beside a comment saying ~50 ms), so a broken `molbuilder.json` fails even `--help`; and `envs/recipes.py` runs `nvidia-smi` at import (`_CUDA_VERSION`), which every command loads through `cli.py`'s `envs_group`. The fix: the snapshot is taken on first use (`diagnostics.get_capabilities`), with the same `Error: ...` for a broken file where it is read; the CUDA version is resolved when a recipe needs it; the two wrong comments (`cli.py:3583`; `jobset/_cli.py:175`, whose header is skipped only for the group's own `--help`) are corrected *(user: "yes, including recipes.py")*. ⑦ *a launched job inherits whatever the shell holds, and that can change its size* -- `running-a-job.md` § 2.0a says the job "inherits nothing"; `submit.py` passes the whole environment (`env={**os.environ, ...}` direct, `--export ALL,...` under Slurm, which also beats a site's `export` directive), and the wrapper takes `MB_NP` / `OMP_NUM_THREADS` from it ahead of the reservation. The fix: keep inheriting, guarded -- the job's size comes from what prep wrote and `launch`'s flags; molbuilder's own override names (`MB_NP`, `MOLBUILDER_*`) still work and the wrapper log names each one used; a bare `OMP_NUM_THREADS` no longer sets it; a site's `export` is honoured. **The contract gets a table: for each run parameter, what decides the value actually used, and which setting wins over which** *(user: "yes, keep inheriting but guard it. document to clarify what determines the actual parameter used, and priority of different settings")*. ⑧ *a customized restart-file list is followed by prep and ignored by the places that describe the files* -- `job-contracts.md` § 4.2a lets a calculation carry its own `warm-files.toml` beside `task.json`, and prep's carry reads it (`rules_for(..., base_dir)`); `warmfiles.inventory` / `carry_inventory` take no folder, so the run script's `Mode :` line (`runwrap.py:452`, `:466`), `jobset status`'s warm-files column (`runstatus.py:53`) and the "already under way" notice (`validation/identity.py:49`) read the shipped list only; and the door § 4.2a promises (describe or the UI copies the file in) was never built. **The root: two doors onto one list** -- `rules_for(engine, calculation, base_dir)` is told the calculation, `inventory(engine)` / `carry_inventory(engine)` are not, and the calculation's-copy rule was added to the first only; every caller of the second holds the calculation's folder and never passes it. The fix: ONE door -- the list in effect for this calculation, asked with its folder -- and the "what can carry" / "every restart file" views read off that one answer; the run script's list is filled in when prep writes the script, not at import; the Task setup page states which list is in effect and where a custom copy goes; both shipped lists carry a header saying how to customize (copy beside `task.json`, what each field means); § 4.2a describes the real door *(user: "keep customization, make every reader use it. provide information on the task setup web ui, and make sure the template for such list has enough comment where use knows how to and where to edit it")*. ⑨ *three places that quietly change the answer when a piece is missing* (latent -- ASE, RDKit and OpenBabel are all in the host recipe): `chemistry.add_hydrogens` returns a heavy-atom structure with only `warnings.warn` (R5) when neither OpenBabel nor RDKit is there, and the Build page's peptide path (`web/blueprints/build.py:352`) does not collect it; the PDB reader's two-letter element check (`structure.py:1556`) is skipped silently without ASE's table, reading "FE" as "F"; `projects.py:219`'s cycle guard cannot fire and would report `paths.projects` as unset if it did. The fix: "add hydrogens" with no engine is refused through the Build page's missing-builder door; the PDB reader asks `chemistry`'s one element door with no fallback; the guard goes *(user: "yes, fix all three that way")*. ⑩ *PySCF's two decks disagree on where outputs go* -- the optimization deck's `_mb_outfile` (`pyscf/input.py:427`) resolves beside the script, the vibration deck's (`pyscf/vibration_deck.py:182`, "THE one definition") against the cwd; the same under the run script, different when a deck is run by hand from elsewhere; `runwrap.py:4108` claims the script rule for both. The fix: one definition, emitted into every PySCF deck from one place, the beside-the-script rule (`absolute()`) *(user: "yes, one definition next to the script")*. ⑪ *comments and documents describing code that is gone* -- `job-contracts.md` § 2.5's walk-up and `$MOLBUILDER_ROOT` (nothing reads it; a job needs nothing from the checkout at run time); `repo_root`'s docstring (only `script_emit` calls it, for the git stamp); `siesta/makov_payne.py:194`'s `runwrap._config_dir_source` (now `companion_source`); ("output at `/dev/null`" in `monitor.py` and `configuration.md` § 2.3 -- fixed with ①, M2a); `MB_LAUNCHED_BY=jobset-submit` in `job-contracts.md:932` and `running-a-job.md:957-969` (the verb was renamed to `launch` on 2026-08-21, `0f489861`; the value is `jobset-launch` in both modes). The fix: each says what the code does *(user: "yes, fix the docs that way")*. **All eleven agreed 2026-09-27; ① built (M2a).** | `execution/run-reports.md` § 2.3 · `architecture.md` § 3 | **①–⑪ agreed 2026-09-27; ① built (M2a, `e31e56f7`); the rest not started** |
| **W37** | execution / front end | **A LADDER'S NEXT STAGE CAN BE PREPPED WITH NOTHING HANDED OVER, AND NOTHING SAYS SO.** Found 2026-09-27 on `projects/PDT/optimization/PDT_FIX_moleculeonly`: all three stages prepped within 10 s, before coarse ran; fine started from the input geometry (SIESTA's `XV file not found`, the wrapper's `initial-run (clean state)`) and converged; final is prepped the same way. The rule (`execution/job-system.md` § 1, *How a ladder advances*): prep one stage, look, prep the next *from a run you name*. What lets it slip: the Task setup page prints `--from` only for a stage whose row sets `restart: continue` by hand (`task-setup/viewer.js:2222`), while `continue` is the default since 2026-08-18 (`engines/stages.md` § 1.3), so a default ladder's Prep button writes a continuing stage with nothing carried in; `prep` accepts a continuing stage with no `--from` and says only "nothing carried in"; `--from` copies whatever the named attempt holds, without the *(finished, converged)* check § 5.3's example prints; the hand-over is recorded in `.continued-from` and `run.json` only -- not the decision ledger, not the Run panel; the deck header says SIESTA reads what the previous run left "in this directory" and prints `launch run 02_medium`, which the resolver refuses | `execution/job-system.md` § 1, § 5.3 · `engines/stages.md` § 1.3 · `project-layout.md` § 1.6 | **agreed 2026-09-27, not started** *(user: "newest finished by default, warn on unconverged, explicit choice will override - user dictate when explicit")*. One section of `execution/job-system.md` owns the hand-over and the other documents point to it. A continuing stage is prepped after the stage before it has finished, from that stage's NEWEST attempt, which must have concluded (the vibration stage's rule, `prep._vibration_stage_geometry`) -- refused otherwise, with the command to run first; an unconverged one is taken with a warning. **An explicit choice is taken as said**: `--from <attempt>` or `--cold`; prep states what it sees about a named run (still running, stopped, unconverged) and refuses only what cannot be done (no such attempt, no restart files in it). Prep prints the hand-over (`02_fine/run-0 continues from 01_coarse/run-0 (finished 14:39, converged): copied <label>.XV, <label>.DM`); the decision ledger logs it; `.continued-from` and `run.json` stay; the Run panel shows *Continued from*; Task setup shows the same answer before Prep; the deck header's text is corrected. The flat shape's line is settled when the section is written. **Added 2026-09-27** *(user: "yes, add it to W37")*: the hand-over moves into the ONE prep both doors use -- today it is a second `prepare_attempt` call on the CLI's path only (`_cli.py:2589`), so the page's Prep door can carry nothing -- and each continuing stage's Task setup card gets a **Continue from** choice: the previous stage's attempts with their state (default: the newest finished), or *start from the input geometry* (`--cold`); the printed command follows the choice and the page sends it to the same prep |
| **W38** | execution / front end | **THE MULTI-DOOR REVIEW (2026-09-27): one fact, two or more code paths that can answer it differently** *(user: "are there possibly other multi-door problem in the api/framework of jobset? ... we are looking for framework level unification and consistency")*. The calibration case is W36 ⑧. Found, to discuss one by one (each re-read against the code before it is brought): **F1** the queue a job goes to is decided twice -- prep's header (`runwrap._placement_for`: allocation, else the menu's recommendation, with that queue's ceiling as the wall) never reads `execution.domain`, which launch uses (`_cli.py:3106`) without adding a wall (`submit.py:236`), so a job lands in `public` carrying `debug`'s 15 min (verified 2026-09-27) -- **agreed 2026-09-27**: one placement decision, asked by the header, by launch and by the Task setup card, in the order `launch --domain` > `allocation.domain` > `execution.domain` > the menu's recommendation; when launch places elsewhere than the header, the command line carries that queue's wall and width (the bench path's way); the order joins W36 ⑦'s precedence table *(user: "yes, one placement decision in that order")*; and **ONE RECORD** *(user: "one record for queue")*: prep resolves the queue once, in that order, and writes the answer and its source into the job set; the `.sbatch` header, the Task setup card and `launch` read that record; only an explicit `launch --domain` changes it, and `run.json` + the decision log then say so; **F2** a flat stage's launch record (`<stem>.run.json`) is read by status but not by `materialize.was_launched`, so re-prep and re-launch of a queued flat stage are not stopped -- **agreed 2026-09-27**: ONE door for *was this launched?*, the same for flat and hierarchical (the reader status uses), asked by status, the run record and every gate; `was_launched` stops being a second answer; a flat re-launch gets the attempts' "may still be running" question *(user: "yes, one door for launched")*; **F3** "did this attempt end on its own" has three readers (`attempt_concluded`: the wrapper's marker only; `compose.classify_citation`: marker or `0_NORMAL_EXIT`; `parse/dirs/job.run_status`: the output's ending first) -- status can say finished where `prep run freq` refuses -- **agreed 2026-09-27**: ONE door for *did this run end on its own?*, the run record's rule (`_process_conclusion`: molbuilder's marker for that run first, it carries the rc; else the engine's own end-of-run mark where it can only belong to this run), asked by status, the freq gate, the transport gather, the citation, launch's re-submit question and W37's hand-over; *how it ended* keeps its own one door *(user: "yes, one door for finished")*; **F4** a stage's number is its list position at every prep (`prep.token_for`), not the token already on disk, so removing a middle stage makes `02_tight` beside `03_tight` -- **agreed 2026-09-27**: ONE stage-number door, the contract's (`project-layout.md` § 4.2): a stage that has a folder keeps its number, a new stage takes the next unused one; on the page, removing a produced stage becomes *disable* (the honest gap); before anything is produced, remove and reorder stay free *(user: "yes, one door for stage numbers")*; **F5** a disabled stage is prepped without a word (`resolve._stage_of`), where transport refuses it by name -- **superseded the same day** (the earlier "refuse" rested on a wrong reading: the web road has no default three-tier ladder -- a new optimization starts with ONE `coarse` stage, `viewer.js:2647`, stages are added by hand and filled from the tier presets; the three-tier ladder with a switched-off third exists only behind `jobset init --stage-strategy`). **Decided 2026-09-27: no on/off for optimization and vibration** *(user: "remove on/off for optimization and vibration")* -- the row's on/off button and `--stage-strategy`'s switched-off stages go. **Agreed** *(user: "yes, drop the seed switch too"; earlier: "do we really need this flag? ... why don't we add a suffix .disabled to the dir or to the script")*: no `enabled` field at all; removing a stage that left files marks them -- `02_medium/` -> `02_medium.disabled/` (hierarchical), its deck and run script `.disabled` (flat), outputs untouched -- its number stays taken (the stage-number door reads the disk), refused while its job is launched and not finished; a same-named stage added later takes the next number; transport's seed is skipped by removing it (the DAG reads the described stages, `prep.py:1967`) -- so **no `enabled` field anywhere**, nothing left needing it (36 `task.json` under `projects/`, none switches a stage off). Old files: `enabled: true` -- written on every stage by the page and `jobset init` -- is accepted and ignored; `enabled: false` is refused by name with the fix (the stage reader refuses unknown keys, `task.py` `_check_keys`, so removing the field outright would refuse all 36); **F6** "does this job use a GPU" -- the `.sbatch` header adds its own conditions (`.fdf` only), so a PySCF GPU job's header asks for no GPU; the GPU type has a second door too -- **agreed 2026-09-27, in the user's terms** *(user: "one door for GPU. make sure pyscf logically treated the same way ... gpu is (1) a resource claim for specific machine/domain, (2) a resource request from the task as explicitly specified by user, and the request will need a compatible resource or will generate error explicitly. that's the design")*: the REQUEST is the task's (`use_gpu`, and a type or count if stated), read by one door for both engines -- the header's `.fdf` condition goes; the CLAIM is each domain's (type and count, from the machine record, `scheduler.gpu.default_type` as the override), read by one door; the MATCH is F1's one placement -- a GPU request goes only to a domain whose claim fits, the header and the command line both carry it (`--gres`, the GPU partition), and no fit is an explicit error naming what was asked and what the machine offers; G-5's run-start check of the device stays; **F7** the page's Prep button skips the CLI's preflight, the "already under way" question, the agreement warning and their ledger lines, and refuses an axis-less bench the CLI preps -- **agreed 2026-09-27**: ONE prep entry both doors call, returning what it found and decided as data (the preflight's findings, *already under way*, the agreement warning, the hand-over, the pipeline log); the CLI prints and asks, the page shows them and asks the same question as a confirm; a bench with no axes is the machine's proposal on both (`generator.md` § 4.3a) *(user: "yes, one prep entry for both")*; **F8** (display) the ladder's complete / next-to-run is computed by status (prepped rungs) and again by the Results door (every described rung) -- **agreed 2026-09-27**: STATUS OWNS THE LADDER AND IS MADE CORRECT FIRST *(user: "yes, status owns the ladder. but you need that status to be correct")*: today `jobset_status` walks only the prepped jobs, so relax-finished-freq-unprepped prints "All stages finished. Nothing to resume." (`runstatus.py:340`) -- wrong; it walks every DESCRIBED stage (not prepped yet = `not prepped`), answers complete / next-to-run once, through the one launched / finished / stage-number doors (F2-F4), and the Results door shows that answer instead of computing its own; **F9** which relax attempt fed a freq run is picked at prep and picked again at summarize -- **agreed 2026-09-27**: the freq stage's geometry pick is a W37 hand-over -- the same default (relax's newest finished attempt), the same optional explicit choice (`--from 01_relax/run-N`, the page's *Continue from*, taken as said with what prep sees stated, e.g. *not converged: expect imaginary frequencies*), the same ONE record, and `summarize` reads the record instead of picking again; a relaxed result from elsewhere takes the existing *already relaxed* road (export the pair with its `info.relaxation`, freq only, `check_relaxation_record`) *(user: "yes, record F9 that way")*. **Minor:** M1 `template_path(base, label)` vs `find_template(base)`; M2 vibration `summarize` infers the stage directory by existence; M3 the run-shape preview probes the machine live where prep refuses to; M4 the transport record reads `run_status` without the launch record and picks the `.out` by mtime; M5 a sweep trial's wrapper is told the task label, not the trial's. -- **all five agreed 2026-09-27** *(user: "yes, fix all five that way")*: M1 one template door (the single template, refused by name when its file name is not the label's); M2 the shape is asked (`Shape.stage_dir`), never guessed; M3 the preview reads the machine record like prep, and says so when there is none; M4 the transport record asks the one run-state door with the launch record and finds files by run number; M5 a trial's wrapper is told the trial's own label. Also met: two more doors on the restart-file list (the bias chain's hand-spelled `{label}.TSDE`, `submit.py:1561`; the re-submit's stored `warm`). -- **decided 2026-09-27** *(user: "yes, fix all three that way")*: the bias chain's copy list is written into its script at prep from the one restart-list door (joins W36 ⑧); the re-submit's stored `warm` is prep's recorded answer and matches the deck it wrote -- no change; the task reader refuses `restart` on a transport stage by name (the catalogue gives transport none, and `transport/stages.py`'s declaration never reads it). **Decided 2026-09-27** *(user: "yes, keep the new attempt but never hide")*: re-prep over a queued hierarchical attempt still opens the next attempt (the queued job's files stay its own), but status lists every launched-and-unfinished attempt with its job id beside the newest; the re-prep question names the job and how to cancel it; launching the new attempt while the old one is queued or running asks first, and refuses without `--yes` when nobody can answer; a flat stage's re-launch marker check | review evidence re-read at each item | to discuss |
| **W39** | model / front end | **CUSTOMIZED PARAMETERS -- a user-made list of named values stored WITH the structure, for customized calculation setups** *(user, 2026-09-27: "in the meta data card/panel, provide an additional customization list so user can make a list of parameters as meta data to be stored in the \"customized\" category of json key, for use with customized calculation setup. this involves UI design and an update of api in the structure meta data handling (a new category other than info which is the calculation setup, now this is a category of customized parameter associated with the structure, and should be considered in the sha calculation)")*. What the design has to meet, read 2026-09-27: `info` is the open store that is NOT the structure -- no emitter reads it, it never enters `structure_hash` (`model/structure-molstruct.md` § 3, ruling 2026-08-29), and the Metadata pane shows it read-only (`web/molview.md` § 8.4a, *"display, never a mutator"*); the structural block is `structure.METADATA_FIELDS` (regions, cell, engine_offset, axis_kind, vacuum, annotations), validated and carried by every edit; three hashes exist and none covers that block (`Structure.geometry_fingerprint`'s docstring names them) -- the sidecar's `structure_hash` is the sha256 of the geometry document alone (`workingcopy_structure.py:280`). So `customized` is a new STRUCTURAL field (not an `info` key), the pane gains an EDITABLE section (a change to § 8.4a's rule, for that section), and which hash it enters is a decision | `model/structure-molstruct.md` § 1, § 3 · `web/molview.md` § 8.4a · `model/structure.md` § 2.2a | **to design -- contract first.** Decided 2026-09-27 *(user: "carrying and showing is enough for first version, but we also need a unified api for writing and reading them from structure, no handcrafted json operation for these customized parameters")*: v1 carries and shows them (sidecar, every edit, the deck's metadata block, the Results tab); ONE API each side, `info`'s pattern -- `Structure.set_customized(name, value, unit=, note=)` / `remove_customized` / a read-only typed view, the codec the only JSON translator, MolView's `viewer.data.customized.set/remove/list` the pane's only door. Found while designing: § 3's box says the hash is "geometry + the structural metadata" but `structure_hash` covers the geometry document alone (`Structure.to_xyz`: count, title, elements, coordinates). **Agreed** *(user: "yes")*: a row is name + value (number / text / bool) + optional unit + optional note; two hashes -- the pairing pin stays geometry-only, a new structure-identity hash covers geometry + structural metadata + `customized`, and § 3's sentence is corrected to say which is which |
| **W40** | execution | **`jobset launch --mode direct` exits 0 when the job it ran failed.** Found 2026-09-27 by M1's fixture: a PySCF run died at activation, the decision ledger recorded `"status": "failed", "returncode": 1`, and the command still exited 0 -- so a script (or a test) that trusts the exit status is told the run succeeded. No contract states launch's exit status for a direct job | `execution/running-a-job.md` · `jobset/_cli.py` | **needs a decision** |
| **W32** | engines / structure | **THE FRAME AXIS — a frame set is one multi-frame pair** *(user, 2026-09-24: "allow multi-frame … which shares the same .json file so meta data and labels are shared, checking of atom number and others can still be gated")*. The contract is `engines/transport.md` § 2a.9 (the set, the per-frame checks, `f000` the base) and `model/structure-molstruct.md` § 6.1 (one sidecar, many frames; a reader that does not ask for frames gets frame 0). **Nothing new is invented**: the codec already writes and reads the pair, and every existing door keeps working because it sees frame 0. **Order of work:** ① the contract — **done 2026-09-24**; ② the citation door classifies a multi-frame pair and checks the four per-frame promises, naming the frame; ③ `prep`: the device and the transmission carry the frame level (`f###`), the seed and the leads do not, the gather runs per frame, one bias for the group (§ 2a.9's ruling); ④ the generator from a spectra file (V1.25) — the vibration side's half; ⑤ Results: the family of curves and what is derived across it (§ 2a.9's deliverable, § 2a.12). **Gate:** ② and ③ land after the single-frame ladder has run end to end once (the run W30's status calls for) — a frame axis on a ladder that has never produced a curve would be measured against nothing | `engines/transport.md` § 2a.9, § 2a.11 · `model/structure-molstruct.md` § 6.1 | **① done 2026-09-24**; ② – ⑤ open |
| **W31** | front end | **Task setup's stage table offers the optimization tiers on transport rungs**: every rung's row carries the `preset… coarse / medium / tight` select, which applies `SIESTA_STAGE_PRESETS` (relaxation-driver values) to a rung whose role is fixed and whose items the `stages` marker routes. Found driving the transport road 2026-09-24. The control is offered only where every field of the tier may be a column of this kind — the columns route's own membership rule (`web/task-setup.md` § 9); the per-rung columns for transport are the rung's own items | `web/task-setup.md` § 9, `engines/transport.md` § 3.8.2 | **done 2026-09-24** — `/api/task-setup/presets` takes the kind and answers an empty menu for transport; the page draws none |
| ↳ | ~~**V1.29**~~ | **DONE 2026-09-24** — the mass-calibrated displacement per mode in the result: `zero_point_amplitude_amu12_ang` and `zero_point_displacement_ang` (`Q_zp · L_canonical`), derived at every serialisation, `null` for an imaginary mode — what the vibration-coupled transport step displaces along (user, 2026-09-24) | `vibration.md` § 6.3, § 6.6 | done |
| ↳ | **V1.15** | **Recorded, not in scope**: the transport connection (displace along a mode, then transport; the electron–vibration coupling from `FC.Save.dHS`) and Born-charge infrared on SIESTA — each a feature to design as one | `vibration.md` § 5.6 | not started |
| **E11** | engine / science | **A fresh live walk of the PySCF / spectra decks.** The 2026-08-28 review exercised them only through the guard suites and says so | audit 08-28 § 5 | open |
| **W1** | front end | **The document tier (step C).** `html, body`, `header`, `button`, `footer`, `textarea` genuinely differ per page; the `*` reset is already deleted. Blocked on a browser pass over all pages | `css-system` § 4C | partly |
| **W2** | front end | **One home per component (step D).** `.card`, `.status`, `header .tagline`. **One value to settle first:** `.card`'s padding is `var(--space-md) 18px 18px` and 18 is off the 4px grid the contract declares — moving it shifts every page by 2px | `css-system` § 4D | not started |
| **W3** | front end | **Per-page token/namespace passes (step E)**, one page per commit: `spectra`, `structure-optimization`, `transport`, `results`, `documents` | `css-system` § 4E | partly |
| **W4** | front end | **Guards 1 and 2 (step F)** — one home including elements; a page sheet contains only its own tier. Guards 3 and 4 landed. **Both remaining gaps are now provable, 2026-09-07:** guard 1 is absent *by an explicit skip* — `test_css_no_duplicate_selectors.py:150` reads `if "." not in norm: continue`, so element-only selectors are exempt by construction; guard 2 has no test at all | `css-system` § 4F | partly |
| **W5** | front end | **The inspectors module's appearance still lives in `results/style.css`.** **Re-derived 2026-09-07 and the number was prose:** 70 was `grep -c inspector`, which counts the file's 200-line comment header and a hierarchy diagram. Comments stripped and classified by who EMITS each class: **22 module-owned rule blocks**, and **6 of those are dead** — `.inspector-section`, `-section-header`, `-section-body`, `-section-hint`, `.source-body-error`, `.structure-error` have **zero emitters anywhere** in the repo and are deletable outright, which this row never said. Three sheets are already repatriated. *"Renders unstyled elsewhere"* is **latent, not reachable**: `registry.js` is script-tagged by `results.html` only, so the css-system doc's premise (it also loads on /molbuilder and /spectra) is stale. Also: `inspectors/bench-summary.css` is missing from the boundary guard's `MODULE_SHEETS`, so that guard treats a module sheet as a page sheet | `css-system` § 7.0 | partly |
| **W6** | front end | **The editor module.** The loader half is confirmed and accurate: `lib/codemirror-load.js` is the one loader, two of three surfaces import it, and `lib/inspectors/markdown.js` still hand-rolls its own pair — *definitions* at `markdown.js:31` and `:38` (this row cited only the call site). **The sheet number was wrong twice over, re-derived 2026-09-07: 21 rule blocks / 60 declarations, not 30 and not 40.** The original 25+4+1 was never reproducible as a block count either — `projects-sidebar.css` has held 16 CodeMirror blocks at every commit back to 2026-08-28. The caps (1500-line selection, 1 MB view-only) are on `preview.js` alone, confirmed | `editor-module` | partly |
| **W10** | front end | **Results transmission inspector** — the record exists, the reader does not | `structure-info` § 3 · roadmap § 2 | open |
| **W13** | front end | **Raw px/rem literals — re-derived a THIRD time, 2026-09-07, and the definition finally holds still.** 160 / 740 reproduce exactly, but only because the regex reads raw file text *including comments*. Counting literals **in declarations**: **133** across the eight page sheets, **650** in `lib/`. Two things the row hides: `lib/tokens.css`'s 44 literals ARE the scale definitions — the token layer, not violations — and `lib/molview/molview.css` alone is **252**, 39% of the whole `lib/` figure. So "lib/ carries 740" is really "MolView carries 252, and the rest of lib carries ~400". 777 → 384 → 160/740 → 133/650 are four scopes, not four measurements | roadmap § 7.4c | partly |
| **W15** | front end | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. ~~Plus routing the CLI through the shared codec~~ — **DONE 2026-09-22**: `xv2xyz` was the last CLI converter writing a lone geometry, and the writers no longer take a path at all, so the class is closed rather than swept (`model/structure.md` §§ 2.3, 2.4). Still open here: exercising the last annotation-channel kind. Confirmed 2026-09-07: `lib/molview/` has **no `_seal.js`** where `spectrumchart/` and `vibrationview/` both do, and seven inspector scripts on `results.html` are still classic `<script defer>` against two on `type="module"` | roadmap § 3 | partly |
| **W18** | front end | **"Modify functions (Molbuilder tab)" — item ZERO of `structure-info-plan` § 5.6's own priority order, annotated *"user calls it higher priority; not yet described."*** It never reached plan.md in any form, and is still described nowhere. Blocks nothing technically; it needs a description before it can be planned | `structure-info` § 5.6 | **closed 2026-09-27** *(user: "close W18, covered by the redesign")* -- the Modify tab's 2026-09-01 redesign (atom, transform, slab, cell, append) is what it asked for; the user's next Modify-side wish is W39 |
| **W19** | front end | **The Modify slab panel shows several findings as one sentence at one tone.** `/api/modify/lattice-from-run` answers with `notes` — the same `{severity, message}` rows every other door sends — and `modify/slab-panel.js:421` joins their messages with `·` into a single toast (re-measured 2026-09-23; the line was `:389`), and `:431` passes the joined string to one `notify.show`. Three findings become one line, and the panel has no list to put rows in. **The severity half is fixed** (2026-09-11: it folded any-warn→warn and drew an *error* in the info tone; it now takes the worst of the three). What is left is presentation, and it is a UI change on the Modify tab rather than a refactor: give the panel a findings list and render through `lib/validation-findings.js` like every other surface (`science/validation.md` § 4.1 R2a) | found 2026-09-11 | **done 2026-09-27** *(user: "option 1, show them as a list")* — the notes are rows under the lattice box, drawn by `lib/validation-findings.js`; the status line states only the measured value, in the info tone; the next measurement replaces the rows, and a value typed or picked from the table clears them (`test_molbuilder_e2e.py::test_a_measured_lattices_notes_are_rows_each_at_its_own_severity`) |
| **E15** | engine / science | **`--pipeline-log` is not wired for the transport arm.** `jobset/prep.py:1295` prints *"--log is not wired for the transport arm yet; prep proceeds without it"* — it says so rather than eating the flag, which is right, but a junction prep cannot produce the one file that answers *"how did this value get into this deck?"*, and the junction is where a ladder's rungs disagree | found 2026-09-11 | open |
| **W20** | front end | **A prep from the browser can never produce a pipeline log.** `pipeline_log` is off by default and only `jobset prep --pipeline-log` sets it (`_cli.py:2252`; re-measured 2026-09-23); the web door builds its kwargs at `build.py:1639/1663/1682` and calls `prep_calculation` at `:1685` without it (re-measured 2026-09-23). Measured: 2 `*.pipeline.log` in the tree, both from e2e fixtures, **none** under any real project. Not a defect — the flag is deliberately opt-in (`pipeline_log.py`: *"the log observes the pipeline, it is not a step in it"*) — but if a person prepping from the UI should be able to ask for one, the door needs a way to say so | found 2026-09-11 | **agreed 2026-09-27, not started** *(user: "always write it, drop the flag")*: every prep writes its pipeline log, from every door (the command line and the Task setup page); `--pipeline-log` and the `pipeline_log=` switch go -- the log is small (18-32 KB measured) and never fatal, and a record asked for in advance is missing when it is needed |
| **W21** | front end / science | **ONE spectrum view, every mode on it — IR, Raman and the silent ones.** *(designed with the user 2026-09-11, walking a real CO2 Raman+IR run.)*  **What is measured today.**  The run computes both channels into `<job>.spectra.json` — per mode, `raman_activity_a4_amu` AND `ir_intensity_km_mol` beside each other — and the viewer shows only Raman.  It touches `ir_intensity_km_mol` in exactly ONE place, `lib/spectra/core.js:1485`, inside the change-detection fingerprint: it reads IR to notice the data changed and never to draw it.  `results.phase_ir` is in the same fingerprint at `:1488` while the phase-indicator list at `:1354-1357` is Relaxation / Frequencies / Raman / Per-mode ES.  The modes table is `# · ω · Raman (Å⁴/amu) · imag? · ES? · HOMO · LUMO · Gap · ΔGap max` — **no IR column**; the chart's y-title is hardcoded `"Raman activity (Å⁴/amu)"` at `lib/spectrumchart/index.js:119` with the unit repeated at `:186`, though the module is otherwise quantity-agnostic (it takes a generic `m.activity`).  So ticking **Compute IR intensities** produces correct physics a person can only read by opening the JSON: the CO2 run gave 653.45 cm⁻¹ ×2 at **32.85 km/mol**, 1388.81 at **14.74 Å⁴/amu**, 2460.11 at **613.04 km/mol** — mutual exclusion exactly right, and the 613 asymmetric stretch is THE band of the CO2 IR spectrum.  **Three things are wrong and they are one thing:** the split into "a Raman viewer" is false — one Hessian, two property derivatives on one set of eigenvectors, so activity is an ATTRIBUTE OF A MODE, not a separate spectrum.  **The shape agreed:**  ① **a mirror plot** — Raman up on a left axis (Å⁴/amu), IR down on a right axis (km/mol), each axis coloured to its curve.  Mirroring is why: both channels peak at the same frequencies, so one half-plane makes them collide, and IR drawn downward reads the way absorption does.  ② **a rug of ticks along y=0** carrying EVERY mode, coloured by class (Raman-only · IR-only · both · silent).  Position only, never a height — a silent mode given a stick height is a lie in either direction, and the rug is the only place a mode active in NEITHER channel can honestly appear.  ③ **a relative threshold slider** in Results — *"above X% of the strongest peak in its own channel"*, one control for two incommensurate units — with its default declared in the Spectrum calculation tab.  **Two prerequisites, both server-side.**  (a) `spectra.json` carries **no activity classification** (measured fields: `index_1based, frequency_cm1, raman_activity_a4_amu, ir_intensity_km_mol, eigenvector_canonical, eigenvector_display, has_imag, electronic_structure`), and the zeros are NUMERICAL not exact — the CO2 run's silent entries are `4.8e-09`, `1.0e-08`, `7.5e-08`, `3.6e-09` — so *"is this IR-active"* is a decision needing ONE home, stored, never a magic epsilon invented in the viewer.  (b) the per-mode electron-structure probe **selects its modes by Raman brightness**: `pyscf/vibration_emitters.py:1416` ranks `key=lambda m: (-m['raman_activity_a4_amu'], …)` and `:1421` cuts on `> ES_THRESHOLD`, whose config label at `config/pyscf.py:1055` is literally *'Raman-activity threshold'* (Å⁴/amu).  Gap modulation is ∂ε/∂Q; Raman is ∂α/∂Q — different selection rules, and in a centrosymmetric molecule the filter is wrong in a *systematic* direction, keeping the gerade modes and dropping every IR-active one.  **Note the distinct thresholds**: `es_threshold` decides what is COMPUTED (2 SCFs per mode) and stays absolute; the new one decides what is SHOWN and is free.  **And a third curve is legitimate later, not a decoration**: ∂ε/∂Q against frequency is the **spectral density** J(ω) of electron-transfer / transport theory — for a junction project the one that governs IETS and vibrational broadening of transmission — which also settles the selector: rank by \|ΔGap\| once computed, or simply take `all` where 2N SCFs is cheap (8 for CO2) | found 2026-09-11 by a live UI walk | **the viewer half is BUILT** (2026-09-11, `b337f91c`: mirror plot, rug, display floor, IR column) and **prerequisite (a) is built** (`spectra/activity.py`, confirmed as the rule 2026-09-23, home `engines/vibration.md` § 6.6); **(b) decided** 2026-09-23 — `top_n`/`threshold` retire at the design's step 4, with the rest of the API shape. This row's *measured today* paragraph describes 2026-09-11 and is history |
| **D2** | doc drift | **Tests with no target, remainder.** The two files with zero test functions were **checked and left** — each is a signpost recording where retired coverage moved, which is a service, not residue. **Re-derived 2026-09-07 with the definition stated:** 417 test files, 6,529 test functions; 2 files with no test function, **0** empty test bodies, and **10** `Test*` classes that collect nothing — not the 5 recorded. Eight of the ten are in `test_results_state_contract_js.py` and its spectra sibling, stating pins in the present tense while holding nothing. Three named remainders still cannot fail: `test_doc_claims.py:92` (loop filters on a string that appears 0 times in its target), `test_monitor.py:342` (`assert callable(fn)` on a `def`), `test_vibration_form_honesty.py:34` (`STILL_OPEN = {}`, iterated empty) | `consolidated-cleanup` § 9 | partly |
| **D4** | doc drift | **The README screenshots are three tabs stale** — five captured, eight ship. Nothing can enforce this (no test can count tabs in a PNG); the *owner* of the count is pinned as of 2026-09-01 | `screenshots.md` | open |
| **R5** | run-decision round | *(priority P3)*  | **`"(this machine)"` means `LOCAL_TARGET` at the prep door and `None` at the bench-grid door.** Real asymmetry, but **the fix is not unification** — `None` is what lets the reader prefer the bundle's own snapshot, and forcing them together broke a live GPU test. The narrow gap: on an unprepped folder with named records, both fit blocks 400 and hide themselves. Fix the *surfacing*, not the value | tried and reverted 2026-09-02; the reasoning is in the code |
| **T4** | run-decision round | *(priority P1)*  | **ALL FOUR DONE** -- the fourth on 2026-09-03 (`b6953f9c`): the flat `cert`/`key` spelling was removed and is refused BY NAME with the `tls` section to write, so the call this row waited for was made; re-checked 2026-09-27 (no `_FLAT_ALIASES` in the tree). **THREE OF FOUR DONE 2026-09-02.** ✅ `Config = SiestaConfig` **deleted** — alias, both `__all__` entries, the two docstring examples that taught it, and the test, together (its only callers were those). ✅ the gcc pin: the test asserted the substring `gcc_linux-64=14`, which **`14.4` satisfies as well as `14.3`** — and 14.4's gfortran miscompiles SIESTA's `kpoint_t.F90` into wrong k-points, so the one thing the pin exists to prevent was indistinguishable from success; now a property check (three packages, one version, minor present), mutation-tested through `MOLBUILDER_GCC`. ✅ the envelope test: rewritten to the property that is still true (a stray top-level key changes nothing, **ignored not refused**, because a request body is not a config file) — `struct_from_body`'s stale docstring head, which still led with the retired flat shape as *canonical*, fixed with it. ⛔ **`_FLAT_ALIASES` is NOT a code shim and I did not remove it** — `cert`/`key` is a **config-file format users have on disk**, and the loader refuses unknown keys, so deleting it stops their server booting. The no-shims rule is about renames in code; this is a migration and needs your call. Was: **Four tests actively block a correct change**: `test_review_fixes.py:237` (`assert Config is SiestaConfig`) and the three `runtime_config._FLAT_ALIASES` tests pin **backward-compat shims** against the project's no-shims rule; `test_envs_siesta_gpu_recipe.py:89` pins `gcc=14` where `installation.md:202` reverses it; `test_structure_envelope_protocol.py:87` pins a deleted legacy branch — **and that one needs the doc fixed first**, since `web-api.md` still claims `/api/modify/*` accepts the old flattened shape | verified |
| **X1** | transport / cleanup | **Transport historical residue — LEADS, NOT A DELETION PLAN.** Five candidates, found 2026-09-22 by reference scan plus spot reads. **① `transiesta._compute_cell_from_extents`** (with `_TRANSVERSE_PAD_ANG` and its two callers, `_lattice_block`'s `cell is None` arm and `wizard.extract_electrode_model`'s lateral fallback) — fabricates a 30 Å transverse / `int(bbox_z+2)+1` box, which is the rule `Structure.resolve_cell()` explicitly forbids (*"a periodic axis needs a commensurate lattice … never a bounding box"*); unreachable because `compose` refuses a cell-less citation on **both** forms (`:814`, `:869`) and `ElectrodeModel.as_structure()` always states one. From `8c70e721`, i.e. it predates `cell.py`. **② `wizard.DEFAULT_ELECTRODE_KZ`** — the third home of `40`; measured equal to the catalogue row and the `SiestaConfig` default, and `engines/transport.md`'s own table says it became a catalogue row. **③ `stages.SEALED_TRANSPORT_FIELDS`** — a union production never reads; `stages.py:390/396` and `transport.py:415/420` branch on the two sets separately. **④ `stages.config_for`** (~90 lines) — the pre-TR4 projection, superseded by `jobset/prep.py::_resolve_transport` (`d6f0218d` *"the transport arm resolves, and 143 lines of projection die with it"*). **⑤ `TransportConfig.num_threads`** — no reader, hidden from the served form by a predicate filter at `transport.py:543`, superseded by `SiestaConfig.omp_threads`; **not a defect**, a dead field behind a working guard. **Every one needs the `process/code-audit.md` § 1d **step 0** pass before it is touched** *(user, 2026-09-22: "a full code review before you decide if your decision about a piece of function is correct")* — a reference scan cannot tell residue from a lost caller, and ④ was one read away from being the second  **① HAS NOW HAD ITS STEP-0 PASS (2026-09-23) and moved to X4 ③.** What it found: the reachability claim in this row is CORRECT (measured — the only renderer is `prep.py` and both structures it can hand over state a cell), but the reason this row gives is not the whole one, and the other machine's audit reached the opposite verdict off `wizard.py`'s stale I6 header. The contract settles it: § 7 calls a padded extent box wrong rather than approximate, and § 5 holds I6 by copying the device's vectors. **②–⑤ are still unexamined** and keep the warning below — and now a second one: the other machine published verdicts on all five, and its verdict on ① was wrong, so those are leads too, not answers. **The audit's own step-0 reads (2026-09-23), for the record:** ② `DEFAULT_ELECTRODE_KZ` residue — the commit that deleted its last readers edited `__all__` to keep it; ③ `SEALED_TRANSPORT_FIELDS` residue — production builds a different union inline, and its only reader is a test; ④ `config_for` part residue, part lost caller — the lost rule, filling the config from a form-B pair's recorded contract, was taken over by `764addd3` in `citation_defaults.py`, so the residue half remains; ⑤ `num_threads` not a defect, but `log_level` is the same shape and does reach the deck (`TBT.Verbosity`, W25). The two documents disagreed about ①, so these too are leads until re-read | found 2026-09-22 | **① passed, moved to X4; ②–⑤ still need the pass** |
| **X2** | transport / cleanup | **Closed 2026-09-22 — both items resolved, neither was a defect in the end.** **① A bare `.xyz` read loses `transport`.** Measured and real: `to_extxyz` writes a boolean `pbc=`, and `periodic` and `transport` are both `True`, so `from_xyz` cannot tell them apart — `(periodic, periodic, transport)` reads back `(periodic, periodic, periodic)`. **Ruled a non-issue** *(user, 2026-09-22: "no single .xyz will be used for transport calculation. always .json is required")*. A transport structure never travels as a lone `.xyz`; the sidecar is required and it carries `axis_kind` verbatim, so the lossy path is unreachable for the only case that would care. `from_xyz`'s own comment already says the boolean is not guessed from — that is the design, not a gap. **② The L/R swap hand-writes the sidecar.** FIXED on the other machine: `compose.py` now goes through `molstruct.load`/`save` inside `with_lock`, so the BOM'd-sidecar → `JSONDecodeError` → HTTP 500 path is gone | found 2026-09-22 | **closed** |
| **X3** | front end / science | **The Cell page commits a box and returns no seam verdict.** Verified 2026-09-22: `classify_seam` / `_seam_notices` have exactly one caller between them, `/api/modify/slab` (`web/blueprints/modify.py:749`); `POST /api/structure/periodicity` (`build.py`) runs `apply_edit` + `validate_periodicity` and neither calls it. So the canonical junction walk — build slab, see `collision`, set `c` on the Cell page — gets **no answer at the one moment the geometry is complete**, which is where `science/junction-cell.md` § 6.1 says the loop closes. Re-building to get a verdict would overwrite `c` with the extent again. **A report, never a gate** — the box is the author's to set. *(The CLI half of this was raised and DROPPED on the user's ruling, 2026-09-22: "people working with CLI would know what they're doing… leave that out." This row is the browser only.)* | found 2026-09-22 | open |
| **D5** | doc drift | **`engines/transport.md`'s deletion bookkeeping describes LIVE code as deleted — three instances, 2026-09-22.** Its *"deleted \| why"* table lists `SEALED_ALWAYS` and `CONTRACT_FIELDS` (live; read at `stages.py:390/396` and `transport.py:415/420`) and `dataclass_to_form_schema` (live at `_shared.py:625`, called at `transport.py:547` — `tests/test_issues_workflow_group.py:331` even records the contradiction in prose); `:1585` repeats the last one as *"has no callers and is deleted"*. **Same class, other documents:** `model/structure-periodicity.md:218/731/927` and `engines/transport.md:2266` describe `transport/_cli.py::_load_device` and a `--cell-fdf` flag — no such module, no such flag, zero hits in `molbuilder/`; and `structure-periodicity.md:578` states emission *"stamps the applied shift … (`frame_shift`)"* and offers re-anchoring from it, which nothing writes or reads. **Why this is not tidiness:** it is what made X1① read as a sanctioned live path instead of residue — the doc described the fabricated box as the transport reading door. The table must be checked row by row against the code before anything is trusted from it | **RE-MEASURED 2026-09-23, symbol by symbol — HALF IS NOW FALSE and the remainder is in a DIFFERENT FILE.** Fixed since this row was written: `engines/transport.md` § 3.7 now carries a *state, measured* column and a header saying the table is a PLAN in the present tense; `SEALED_ALWAYS` and `CONTRACT_FIELDS` are correctly marked LIVE; `dataclass_to_form_schema` is LIVE with one production caller (`transport.py:572` — the catalogue swap removed it and the revert restored it, both on 2026-09-23); the `--cell-fdf` sentence is now an explicit correction. **STILL TRUE, and all of it in `model/structure-periodicity.md`, not `engines/transport.md`:** `transport/_cli.py::_load_device` (no such module — the package holds `__init__ citation_defaults compose deck record sort stages transiesta wizard`) and `--cell-fdf` at `:254`, `:767`, `:963`; and `frame_shift` at `:614`, which nothing writes or reads anywhere. **Retitle this row to name that file** | found 2026-09-22 · re-measured 2026-09-23 | open — in `structure-periodicity.md` |
| **X4** | transport / metadata | **The transport structure's metadata seam — one root, five consequences, found 2026-09-22/23 by reading `engines/transport.md` end to end against the code.** **THE ROOT:** `transiesta._emit_geometry(struct, cell=None)` was LIFTED onto the seam, not migrated (§ 3.6a's lift boundary), so it **takes no config at all** and hand-derives values the framework owns; and `wizard.as_structure()` hand-builds the lead's `Structure` rather than carrying one. Every item below sits on one of those two. **① `species_order` — a Class A value with three answers and no holder.** § 2a.13 classes it SHARED, binding every stage, tier 2 *checkable*: *"it fixes the orbital ordering inside `.DM` and `.TSHS`, and two stages that order species differently write files the next stage cannot read correctly."* Measured: the catalogue row carries `engines = ["siesta"]` and **no `calculations` key**, so the narrowing rule offers it in every transport template — and **no transport deck can read it**, because the emitter takes no `cfg`. Worse, the two emitters disagree on the rule: `siesta/input.py::_detect_species` sorts by **atomic number** (H,C,S,Au → Au=4), `transiesta.py:305` sorts **alphabetically** (Au,C,H,S → Au=1). And it is derived per-structure, so the device ({Au,C,H,S}) and the lead ({Au}) derive independently — agreeing here only because "Au" sorts first either way. § 5's invariant table has **no row for it**, so tier 2 names no holder and nothing checks cross-rung agreement. **NOT claimed: a proven physical failure** — how TranSIESTA matches species between `.TSHS` and the device was not traced. **② `as_structure()` strips metadata silently.** It states elements, positions, title, cell and `axis_kind` and nothing else, so a lead reaches the renderer with no `regions`, `annotations`, `info` or identity columns. Dropping `regions` is right (a bulk lead has no partition). The other three are undecided, and `model/structure.md` § 2.2a forbids that: *"a rebuild that simply did not list the field is a **defect**, not a default"* — and `info` is where the recorded contract the citation warnings read lives. **A candidate rule exists** (proposed 2026-09-23, not adopted): *`info` travels when the derived structure becomes a `.xyz` + `.molstruct.json` pair on disk; where the artifact states the settings itself, the artifact is the record.* It was derived from this case — the junction's `info.calculation` holds a mesh cutoff and transverse k converged for the junction's cell, which the lead does not have, and extraction sets no `structure_modified` — and held on a second (the deck's in-script atom-metadata block carries no `info` and re-derives the contract from the deck); merges are § 2.2b's. By it a lead carries no `info`. Adopting it answers (A)'s first question and gives § 2.2a the test it lacks; `structure.py`'s `info` comment, `wizard.as_structure`, `script_emit.py:448` and `parse/dirs/atom_metadata.py:47` then cite it. `annotations` and the identity columns `as_structure` also drops are still unexamined. **③ `_compute_cell_from_extents`** — X1 ①, now with the contract behind it. § 7 calls a padded extent box wrong rather than approximate (*"padding fabricates an orthorhombic box that severs the periodic gold"*) and § 5 holds I6 by copying the device's vectors, which is the OTHER arm. Measured unreachable: the only renderer is `prep.py`, and both structures it hands over state a cell. (The audit's step-0 read, 2026-09-23, calls the same code **live**, reached by routes that bypass `compose`: `load_compose_record` has no `_unusable_cell` call and the engine seam has no cell gate, so a cell-less junction renders a 184-line deck with every atom outside the box — one code, two definitions of reachable. The ruling decides it either way, `engines/transport.md` § 2a.9: transport derives no cell, so the no-cell arms and `_compute_cell_from_extents` go, and `_validate_transport_kind` gains the no-cell row.) **Nothing pins its numbers** — 15 Å → 99 Å leaves all 26 tests in `test_transport_wizard.py` + `test_transport_cell.py` green. A deletion attempted on self-authored fixtures was **reverted 2026-09-22**; it needs a real cited relaxation first. **④ The settings gate tells transport its labels do not matter.** Observed firing on a device deck: *"this structure carries region label(s) ['L-electrode','R-electrode','bridge'], which the SIESTA run does NOT consume"* — in the `.validation.txt` beside every transport deck, where the partition is what the whole ladder is built on. § 3.6a already records it: both checks take `(struct, cfg)` and cannot see the kind. **⑤ `structure_hash` is written and verified nowhere.** `workingcopy_structure.py:273` writes it on every save; `molstruct.load` says *"NOT verified here (the caller compares it…)"* and no caller does. **The likely answer is DELETE, not wire up** — it guards a second actor editing generated files, which the single-user rule calls a fake problem. **But the 2026-09-23 handover records the user specifying the other shape**, and the audit the same day still listed it as needing a ruling: *detect* on load, never auto-repair, never refuse (*reading does not judge* — a file you cannot open is a file you cannot fix); *attest* through a separate door the person calls after checking the file, which re-stamps the sidecar only — named as an attestation (*"I have looked, and the labels still apply"*), never as a repair, because only the person who made the edit can say it. What it guards is that person's own hand edit months later: the labels are atom indices, and `info.calculation` was measured on coordinates that have changed. Its write gate also checks only `len >= 16`, so `'not a hash at all!!!'` loads (the audit's § 1.8c). If detect: the rule goes into `model/structure-molstruct.md` § 3 before any code. **⑥ `load_compose_record` conflates absent with incomplete** — a record missing one file returns `None` exactly as no record does; where the citation cannot re-resolve, prep then raises *"the composed junction record is not beside `task.json`"*, naming the wrong cause. Minor. **THE ORDER OF WORK, and it is not negotiable between phases. (A) DECISIONS, contract only, no code:** does a lead inherit the junction's `info` (② — § 2.2a demands an explicit answer either way); what holds `species_order` across rungs, and is it a transport parameter at all (① — § 5 needs a row, or § 2a.13's tier is wrong); is `structure_hash` a guard or residue (⑤ — the user is recorded as having specified detect-and-attest; confirm it). **(B) THE SEAM, and ① cannot be fixed before it:** § 3.6a's own rule — *"a keyword with a value is a SECTION ITEM; structural text is a Block"* — says the species ORDERING is a value and the species TABLE is structure, so they split; until the geometry block can see a config there is no path from the row to the deck. **(C) THE FIXES**, each now reachable: one species rule reaching every rung with a cross-rung check; `as_structure` carries or explicitly strips; ③ deleted; the gate made kind-aware; ⑥'s message. **(D) X1 ②–⑤**, never examined, each with its own § 1d step-0 pass — and **verified rather than inherited**: the other machine's audit verdicts on them were not re-derived here, and its verdict on ① was measured wrong. **THE GATE OVER (C) AND (D): nothing lands without a real cited relaxation composed end to end.** Fixtures authored by the person making the change is how the 2026-09-22 attempt failed. **DONE 2026-09-23, documentation only, no behaviour:** `transport.md` **§ 6.2** — *"what happens to the STRUCTURE, hop by hop"* — written, the view that did not exist (§ 6 follows files, § 6.1 follows scripts, nothing followed the structure, and that gap produced two reviews reaching OPPOSITE wrong conclusions about the same twenty lines); it carries the 8-hop diagram, what each hop does to the box and to the rest of the metadata, a worked Au(111) example, and the rule the whole confusion turned on — **the lateral vectors are COPIED and the transport vector is COMPUTED, so "derived" is not the suspicious word; "fabricated from atom extents" is**. Plus: `wizard.py`'s I6 header, the single line that misled both reviews, now names the copy as the mechanism and records that it was misread twice; `as_structure` states what it drops; `_emit_geometry` explains why it reads the cell RAW and the origin RESOLVED, and flags ①; `_lattice_block` marks its fabricating arm unreachable and **not sanctioned** | found 2026-09-22/23 | **A needs your decisions; B is a design change; C/D blocked on a real run** |
| **A1** | structure / validation / tests | **The structure-API audit's open findings — moved here 2026-09-25, when the audit was archived.** The audit (seven reviews of the structure API, 2026-09-22/23) held its findings and its cleanup order outside this list; they are the rows below and § 5r, and the audit itself is the evidence record, measurement by measurement: [`archive/2026-09-22-unification-audit.md`](?doc=archive/2026-09-22-unification-audit.md). **Every row is as measured on 2026-09-23 — re-derive before acting (§ 5a).** The origin-rule sites it found are W33's, not these | the unification audit, archived | open |
| ↳ | **A1.1** | **`molbuilder validate` with no `--engine` runs no cell check.** An explicit left-handed cell → `n_errors 0`, exit 0; with `--engine siesta` → `cell.left_handed`, exit 2. `cli.py:454–467` branches to `validate_geometry` | audit § 1.2 | open |
| ↳ | **A1.2** | **The documented CLI pipe destroys the pair.** `modify a.xyz - \| modify - b.xyz` writes `b.xyz` alone — regions, cell, axes gone, and now silently (no sidecar at all). Needs the CLI-stdout ruling first (A1.15): what a single stream can carry | audit § 1.4 | open |
| ↳ | **A1.3** | **`/api/structure/analyze` 500s on a file it should describe.** A latin-1 PDB → HTTP 500 (`build.py:236` `read_text()`); a BOM'd `.xyz` → 400 where `/api/build/load` → 200; the sidecar is never read. The doc half — `structure.md` § 2.4's second wrong statement — is not `web/`'s and goes first | audit §§ 1.5, 1.5a | open |
| ↳ | **A1.4** | **Two silent reads of a broken metadata block.** A malformed annotations channel escapes `StructureCodec.load` as a bare `KeyError('kind')` naming no path (`apply_atom_metadata` the same); and `_extract_atom_metadata_dict` (`script_emit.py:1953`) returns `None` on `JSONDecodeError`, so a corrupted fence reads as *no labels* with nothing said | audit § 1.7, § 1.18 | open |
| ↳ | **A1.5** | **The pair's doors re-derive one rule.** Eleven spellings of *"is this a structure path"* (`workingcopy_structure.py`, `files.py:257`, `selection.py:99` — dead, `siesta/input.py:1872,1902`, `build.py:240,722,813`, `cli.py:93,773`, `structure.py:94`) and a fourth in the browser (`task-setup/viewer.js:1144` derives `<stem>.molstruct.json`). Rename strands labels: `water.xyz → notes.txt` leaves `notes.molstruct.json`, and the reverse adopts a foreign sidecar unchecked. A `.pdb` source travels under three names — describe records `c.source.pdb`, the codec writes `c.source.pdb.xyz` (`files()` never passes `fmt`), prep looks for `c.source.xyz` → *"the structure this calculation describes is not here"* | audit §§ 1.8, 1.8a, 1.18 | open |
| ↳ | **A1.6** | **`Structure.replace()` still hand-enumerates.** Nine fields in a literal, five more from `_carry_nonatom()`. Completeness is pinned by a test iterating `dataclasses.fields()` — but its fixture has `annotations={}`, so a `replace()` that drops annotations passes it. Deriving the carried set from `dataclasses.fields()` makes it complete by construction; `frozen_atoms` stays excluded (a property over `regions`, no storage). Read `replace()`, `_carry_nonatom()` and `__post_init__` end to end first — it is the most load-bearing method in the model | audit § 1.16e; the 2026-09-22 handover | open |
| ↳ | **A1.7** | **The engine registry fails open.** With `molbuilder.siesta` unimportable the registry lists only `PySCFConfig`, and `validate(Au2, SiestaConfig())` runs no `config.*` check and says nothing. § 1.11b is done (`3382b851`); its sentence for `science/validation.md` § 7 is still owed | audit § 1.11 | open |
| ↳ | **A1.8** | **Rules re-derived at the call site.** `estimate_partial_charges` is label-blind (water `O1,H2,H3` → **0.0 D**; `_DEFAULT_EN = 2.20` is hydrogen's value); **twelve dead `axis_kind` fallbacks** (11 × `("isolated",)*3`, 1 × `()`, all unreachable, plus `validation/siesta.py:734–737` resolving the opposite way — and `transiesta.py`'s was `("periodic",)*3`) — pure deletion, first; the k-sampling hint measures the gap in two frames (hexagonal cell: the hint says ~5.5 Å, the perpendicular gap is 4.16, `_min_image_distance` 6.5) | audit § 1.12a–c | open |
| ↳ | **A1.9** | **`load()` restamps `schema_version`.** A sidecar written at 7 reads back as 9 through `molstruct.load` — so no reader can say what version a file was. W33 moved the schema to v10 and inherited it: `parse/sidecars/molstruct.py` stamps v10 on every read | audit § 1.13 | open |
| ↳ | **A1.10** | **Placeholders stored as facts, and two builder defects.** The backbone check keys on `rid − 1`, so a 5P duplex is refused (*"residue 4 O3' → residue 5 P 16.74 Å"*) with the blame on `$X3DNA`; rdkit-added hydrogens carry `(1, MOL, A)` and are persisted as real identity (7 of 12 atoms); `smiles.py:155,187` and `_common.py:35,40` spell index names `C1, O3, H4…` as stated identity, so every SMILES-built molecule ships a sidecar; `_amber.py:77–86` warns *"requested B-form … not enforced"* on every build. The placeholder must be carried apart from the data — a shape decision before a fix | audit §§ 1.15, 1.18 | open |
| ↳ | **A1.11** | **The ghost element `X` is a legal element.** `resolve_element('X')` → `X`, `atomic_number('X') = 0`, `atomic_mass('X') = 1.0`; it passes `check_species_labels`, contributes Z = 0 to the electron count, and `render_fdf` writes `%block ChemicalSpeciesLabel / 1 0 X` — the defect `chemistry.py`'s own docstring says it exists to end. An unstated rule to state with it: an element denotes Z ≥ 1 | audit § 1.18 | open |
| ↳ | **A1.12** | **The H/heavy-ratio check is wrong twice.** It counts `e == "H"` on RAW labels (`validation/geometry.py:89–90`), so labelled methane (`C1, H1…H4`) warns *"H/heavy 0/5"* and unlabelled does not — the owner is `chemistry.is_atom`. And it fires on every transport deck, where a metal junction has no hydrogens by construction (measured 2026-09-24/25, every rung) | audit § 1.18; the 2026-09-24 handover | open |
| ↳ | **A1.13** | **One rule, several enumerations — and two strictnesses.** ~~Three `_CONTAIN_EPS`/`_EPS` (one unread); `cell._contains` re-implements `Structure.cell_contains_atoms`~~ — gone with W33 (`6c705058`): containment is one distance tolerance, the hand-off's; `affine`/`concat` hand-list the columns; `vacuum=-5` is accepted by the model and refused by the gate; `annotations` accepts a `str` index `regions` refuses; `set_channel` installs then validates, so a refused channel stays and breaks `copy()`; `files('mol.pdb')` writes an XYZ inside; `apply_to_structure` on a partial payload resets the cell. Fix each at its owner, never at the instance | audit §§ 3, 4 | open |
| ↳ | **A1.14** | **The same shape, latent** (unreachable or harmless today). A second PDB reader (`builders/backends/_common.py:57–93`: `Mg → M`, `Cl → C` on a blank element column); a second deserialiser (`selection.py:131–173`; `_shared.py:124–165, 1349–1406`); eight metadata-dropping rebuilds (`add_hydrogens`, `protonate_phosphate_oxygens`, `_drop_overlapping_hydrogens`, `relieve_clashes`, `_strip_5prime_phosphate`, `select_chain`, `_patch_residue`, `_fix_methylene_hydrogens`); `describe.write_description(struct=None)`; `_reset_to_derived`'s own 1e-6 threshold; `_validate_transport_kind` reading the raw cell; `transiesta.py:601,705` bare `+ 1` into the deck; `pyscf/input.py:1575` serialising with `json.dump`; `chemistry.py`'s `_adjacency` on raw `"H"` and a second periodic table; `emit_atom_metadata` dropping a kind-less channel silently; `build.py:410/1296`; the backend set spelled in seven places | audit § 1.18 | open |
| ↳ | **A1.15** | **Stale comments, and rules nobody wrote down.** `structure.py:9–13` cites the deleted `molbuilder.load` and `docs/design.md`; `structure.py:892` asserts a `frozen_atoms` reader that refuses; `workingcopy_structure.py:28–30` says the CLI writes geometry alone; `validation/sidecar.py:3–7`. To state: what the CLI's single stdout stream carries; `cell: null` in a load body; a partial `apply_to_structure` payload; a `.XV`'s companion sidecar; whether *Delete file* pairs the sidecar; `Frame.lattice` against `Structure.cell` (which frame a run artifact is in is W33's) | audit § 1.18 | open |
| ↳ | **A1.16** | **The UI walk of 2026-09-23, not yet rows elsewhere** (U3/U4 are V1.14; U6 is W21's). **U2** the printed prep command lacks `--target`, which `task-setup.md` § 10 says the page puts in; **U5** `max_force_eh_a` holds Eh/Bohr and the viewer prints Å; **U7** `[cell.vacuum_defaulted]` advises vacuum for a boxless PySCF run; **U8** the hand-over gate admits PySCF only for a vibration while the CLI road opened SIESTA vibrations on 2026-09-23; **U9** a SIESTA `.spectra.json`'s nulls must draw *not computed*; **U10** `compose.py:1082,1175` writes and reads the permutation record by hand beside `write_permutation`/`read_permutation`, and its record carries no `key` | audit § 1.18 | open (held on `web/`, `transport/` when measured) |
| ↳ | **A1.17** | **The tests: the count must come down, and three rules are unpinned.** Unpinned: `write(struct, "x.pdb")` producing a readable pair (restoring the pre-2026-09-07 bug leaves 460 passed); geometry-before-sidecar, both-or-neither (reversed → 441 passed); the explicit-cell centring branch (deleted → 441 passed — W33 makes it the rule and pins it). Blind or shape-asserting: ~~`test_periodicity_gate.py:1126` (inverted — fails on cosmetics, passes on deletion), `test_cell.py:202` (`"a " in message`)~~, `test_cell.py:588` (a signature), ~~`TestDocMatchesTheDoor` (blind to a shrinking `OPS`)~~ — the struck three fixed or retired with W33 (`6c705058`), which also pins the explicit-cell centring branch as the rule (T5, T1). ~24 duplicates in twelve clusters — 16–17 tests carry *"the default isolated vacuum gap is 3 Å"* (a floor: re-derive with the mutant over the full suite). 15 fixtures hand-build a `structure_hash` no writer emits, three matching a different error than they name | audit § 5a | open |
| ↳ | **A1.18** | **The contracts' line numbers are 15 % right.** 6 of 39 `file.py:NNN` references resolve (re-derived exhaustively, 2026-09-22); seven symbol names have never existed; five retired concepts are written as current. `model/parse.md:355` already states the rule — a line number is a pin; the fix is one mechanical pass, then the behavioural list | audit § 2 | open |
| ↳ | **A1.19** | **The test harness can report a green suite that is not green.** In `tools/`, open and not held: `run lf` with nothing to rerun shouts NOT GREEN (exit 5 on a green last run — `exit 4 \| 0/0 ran` appears 302 times in history, so the guard gets trained away); `testrun.py failed` emits node-ids with a `[teardown]` suffix pytest refuses; the head line stops summing (`2/2 ran \| pass 1 FAIL 3`). Same class: `progress_plugin`'s `except OSError` silently disables the writer and `cmd_status` returns 0 for no-data; the env canary's *DISARMED* goes through `warnings.warn` with no hook, so a run whose canary proved nothing reads like one that proved everything | audit § 0c | open |
| ↳ | **A1.20** | **Residue, with the step-0 read done** (`process/code-audit.md` § 1d): five `_enumerate_files` buckets with no reader, built on the Watch polling path with four directory scans; three legacy shim classes and the five test-only names that depend on them (one decision); `sha256_of_file`, whose docstring calls it the `structure_hash` pin (X4 ⑤); `selection_rules`, a format field with no producer and no consumer; `sidecars.molstruct.load_text`, zero callers; the `*-electrode` convention two modules advertise and `sort.PARTITION_LABELS` cannot compose. Last in § 5r's order | audit § 5 | open |
| ↳ | **A1.21** | **Audit #2 — the rest of the tree, planned and not started.** ~70,000 of ~110,000 lines were outside the audit: `web/static/lib/` (33k — a language boundary), `jobset/` (12k — a process boundary), the rest of `web/blueprints/` (~9k), `runwrap.py` (5k), `runtime_config`/`template`/`monitor`/`checkpoint`/`task` (~9k), `config/` + `validation/` (~8k). Kept separate because the failure shape differs (one fact per side of a boundary, not one door per operation) and so does the evidence (a JS finding needs no Python fixture; a wrapper one needs a submitted job) | audit § 7c | planned |
| ↳ | **A1.22** | **Should a deck carry the atom-metadata block at all, or a pointer to its pair?** Two `.molstruct.json` readers by format are right — `apply_metadata_dict` for the sidecar, `apply_atom_metadata` for the deck's in-body block — and they now share the validator. Whether the deck's copy should exist was deferred by the 2026-09-22 session to the structure-I/O work, which then merged into the audit without taking it up | the 2026-09-22 handover | **decided 2026-09-27: the deck keeps its block** *(user: "yes, keep the block. the .json and .xyz where this task is setup may not be copied to the dir. the .fdf is all that information stays")* -- the deck is the run folder's own record of the labels it ran with; the pair at the calculation's root is the structure's. The two readers by format stay, sharing the validator; nothing to build |
| **B2** | run-decision round | *(priority P3)*  | **PARTLY DONE 2026-09-03 — and the number was the wrong instrument.** Of the four shapes named here, only one is mechanically decidable: a `Test*` class whose body is a docstring collects nothing. Five existed. **Three were empty promises** — `TestBuildSiestaHonorsSidecarFrozenAtoms`, `TestWorkspacePayloadRegionsAndFrozen`, `TestGenerateWritesToWorkspace`, each stating in the present tense that it pins something (*"Tests pin both layers"*) while holding no test, so a reader scanning for coverage reads yes. Each is replaced by a pointer at the file that DOES cover it. **Two are deliberate retirement markers** that say so and name their successor — the same call D2 already made for two zero-test files. The other three shapes do not survive measurement: `assert len(X) == 5` where the test BUILT X is a real check, and `m = re.search(...); assert m` is a precondition with the real assertions after it. **A list of ~45 that cannot be re-derived is not a finding anyone can act on** — what is left needs the file-by-file read, not a regex | 5 measured |
| **B3** | run-decision round | *(priority P3)*  | **CLASSIFIED 2026-09-06 — and the population is a fifth of what three earlier counts claimed.** 233, then 256, then 173 were three definitions, none written down. Measured now by `tools/classify_source_reads.py`, which states its definition and can be re-run: of **1,255** assertions over a file's text, **1,147 read GENERATED output** and are correct as text — a property of a real product, never a defect. **108 read hand-written source**, in 31 files. Of those, **59 stay** (51 lints, where text is the only instrument that can prove absence, and 8 vendored/data files) and **49 convert**. Full method, per-bucket file list and the mutation proof are **§ 5h** | 49, not 233 |
| **B4** | run-decision round | *(priority P3)*  | **MEASURED 2026-09-03; the envelope half is done, the fixture half is proposed and NOT applied.** The `_envelope()` count was seven, and only **three** were re-implementations: `test_pseudos.py` and `test_task_setup_tab.py` hand-listed the envelope's fields (so a field the envelope grows would never reach them) and both now go through the one builder — which immediately surfaced a real defect: a test built a 2-atom envelope and overwrote `elements` to three, leaving `atom_names` describing the old atoms, and the route's own guard caught it the moment the canonical dict was used. The third was `test_structure_envelope_protocol.py`, carrying TWO docstrings back to back (the second was dead). The remaining four are a delegating alias and one-line `struct.to_dict()` calls — not the hand-rolled XYZ parsers the helper was written against. **`flask_server`: DONE 2026-09-03, without touching a single scope.** 18 of the 20 now call one context manager, `tests/support/live_server.py::serve()`; each module keeps its own `@pytest.fixture(...)` line, because a scope is a decision about how much state a file's tests share and a de-duplication does not get to change it for them. ~230 lines and 18 now-unused `import threading` go with it. The two left alone pass a non-default app config, which is a real difference. **`_node_esm`: 24 of 47 `*_js.py` files drive it** (the row said 7 of 48), and 13 more shell out to `node` themselves | 3 done · 16 proposed |
| **S1** | architecture seams | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — four branches, `runwrap.py:420 / 446 / 645 / 728`, the same four counted 2026-08-19; 128 engine-name literals in the file |
| **S3** | architecture seams | **`runtime_config`'s untyped scheduler dicts + mixed concerns** | `backend-architecture.md` § 5 (**W3**) | **OPEN — verified 2026-09-06.** `runtime_config._validate_scheduler` still returns `Dict[str, Any]`; overlaps **S6** |
| **S6** | architecture seams | **The scheduler menu is handed out as plain dictionaries**, so the typed record and the code using it never meet — how `gpu_partition` came to redirect GPU work from inside an unexamined bag | roadmap § 7.6 phase 3 | **measured partly** — the *record* is typed (`Domain`, `Device`, `Topology`, `Site` in `scheduler/record.py`); the *menu* is not (`known_machines() -> List[Dict[str, object]]`, `Domain.to_row() -> Dict[str, Any]`). Phases 1, 2, 4, 5 are done — phase 2 landed as `scheduler/admit.py`, split out so the check cannot drift from the record it checks |
| **S7** | architecture seams | **The preparation layer against its contract** — **P1** the enforced floor map puts `runwrap` and `jobset/prep` on floor 5; **P3** nothing names the shared package (`jobset/prep._shared_for` globs); **P5** PySCF's seam entry. P2, P4, P6 closed 2026-08-18 | `execution/script-preparation.md` | **P3 CLOSED — verified 2026-09-06**: `_shared_for` calls `seam.shared_package(base)`, and the code names the glob it retired as *'an accident of which suffix the glob happened to name'*. P1 and P5 not re-derived |
| **S18** | ops / envs / config | **The 2026-09-12 env-installer and config-and-secrets session has its own hand-over file: [`2026-09-12-env-config-handover.md`](?doc=plans/2026-09-12-env-config-handover.md).**  80 items, each marked by how it was checked (RAN / READ).  It exists because that session's own commit messages are not reliable: three independent audits were told to FALSIFY them, 14 of 16 behavioural claims held, and the failures were overstatements of scope -- four documentation statements written that day are false, two of them in text `envs init-config` ships into a user's config directory.  Its § 1 is the verified DONE list and exists to stop the next session re-deriving settled work; § 2 is the work, grouped as defects introduced (A), false docs (B), an instruction not implemented (C), sweeps that stopped at the first instance (D), pre-existing finds (E) and decisions for the user (F).  **§ 0 states the TARGET first** -- the installer's two state machines and one runner, and config's one resolver / one name / one writer, as 13 checkable invariants T1-T13 -- so every item reads as a named deviation rather than a patch.  **§ H is the residue of the pre-state-machine design**, swept against those invariants rather than against any diff: one question answered in two or more places four times over, a string where the design says state three times, dead parameters, and a door that never sanitises the environment it dispatches into.  **§ I is the config/secret residue**, and its first lesson is that **A11's own text in `architecture.md` still names a pre-consolidation owner**, so the rule as written licenses the three `.parent` climbs it forbids -- fix the rule before the sites.  § G records what was checked and found clean.  **§ 3 is the MIGRATION PLAN** -- eight phases scoped to `install-env.sh`, the `envs` verbs, deployment and how config is placed and validated, each stating what it closes and which end-state row (Z1-Z9) it realises; § 3.0 states that end state so it can be checked, and § 5 records what is deliberately out of scope.  **Read § 1 before touching § 2, and A0 before anything** -- `envs install molbuilder --clean` currently deletes the env the process is running from, and `envs doctor` prints that command as its remedy for a failed host verify.  No clean full-suite result exists for the work yet (F3) | audited 2026-09-12 | open -- nothing in § 2 started |
| **S13** | architecture seams | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built, re-verified 2026-09-20.** Nothing in the tree names it at all now — even the `transport/wizard.py` comment that used to is gone, so the only record that it is owed is `engines/transport.md` § 8 and this row |
| **N10** | parse / front end | **A CALCULATION ROOT IS NOT A RUN DIRECTORY, and the Results tab has only one notion.** Measured on the real transport ladder: `jobset_status` answers 5 stages all `pending, prepped, not launched`; `run_status` -- which `/api/results/dir` calls unconditionally -- answers `running, no result file yet`, so the tab tells a person a calculation nobody launched is running. The tab offers that root its own INPUT structure as the result, and says nothing about the five stages. `transport.md` § 2a.12 has required the ladder's state, the curve with its treatment named, and the provenance chain since before the surface was built. **The predicate already exists** -- `checkpoint._is_bundle_root` -- and a second copy in `parse/dirs` would be instance 14 of § 8 | § 5c.3, `transport.md` § 2a.12 | **RE-MEASURED 2026-09-23 — the headline defect is CLOSED; c/d/e open.** § 5c.3's two premises are both stale: `/api/results/dir` no longer calls `run_status` unconditionally (`results.py:213` asks `calcdirs.container_or_run` first, and a container gets `status: None`), and the public owner that landed is a NEW module, `calcdirs`, **not** the `checkpoint._is_bundle_root` step (a) proposed — which is still private at `checkpoint.py:384` with its one caller. So (a) is superseded, (b) is done in substance but via `place == CONTAINER` rather than a `jobset_status` ladder (`jobset_status` has zero hits in `results.py`), (f) is DONE (`inspectors/transport.js:180-205` renders the provenance chain), and **(c) the payload and (d) the ladder view are DONE 2026-09-24** (`ladder` on `/api/results/dir`, the empty-state card's table, `results.md` § 2.4); **(e) the inspector's parser is not started** — `transport.js:336` still does `JSON.parse`. **AND A NEW, SMALLER QUESTION:** there are now TWO predicates for *is this a calculation root* on DIFFERENT evidence — `_is_bundle_root` tests for `task.json`/`job-set.json` existing, `container_or_run` reads `task.json`'s `shape`. The § 8 duplication § 5c.3 warned about is real in a milder form, found by a browser walk that 2,800 passing tests missed |
| **N9** | parse / run files | **The front door has no consumers — wire them.** `JobDirParser` composes `run_status`, `_enumerate_files`, `engine_of` and the discovery chain into one answer, and `RunDirResult` is its shape. **Both STAY**: zero callers is a migration that has not happened, not evidence the door is unwanted. *(This row said "delete the bundle" until 2026-09-18 — four unlike things under one word, which read as "delete the framework". User: "job dir parser is actually the framework we're developing".)* What IS a defect is narrower: a **directory** routes through the same `detect()` the file verbs call, which cost three CLI verbs their clean refusal — one a silent hang. That is the ROUTE, and those verbs already ask `answers_a_trajectory()` instead. **Open: wire the consumers; decide whether the route is `detect()`/`parse_dir` or a direct call** | § 5c.2, § 5.0, § 5.5 | **open — the migration** |
| **N8** | parse / run files | **The run-output vocabulary is stated in nine places and the directory door is blind to PySCF's.** `.pyscf.log` is where PySCF writes and not `.out`; `parse/dirs/` never searched for it, so **3 finished runs report `running`** — one for 97 days — with `Job complete in <N> s` sitting in the directory. Four sites hardcode the role list; fixing one (the status probe, 2026-09-18) left the viewer's discovery chain and the sidecar pairing blind. The catalogue (`runfiles.WRITTEN`) already owns the vocabulary and already ships derived views | § 5c.2 | **CLOSED 2026-09-18 — steps 0–i all landed.** The vocabulary is one catalogue column, `ending_of` dispatches on the ROLE, `openable_in` delegates, and the Results tab asks a door instead of guessing. What the work then FOUND is its own row: N10 (a calculation root is not a run directory) |
| **N5** | parse / run files | **The three defects § 5l measured, which outlived it** (§ 5l.a). **① LIVE:** every staged run loses its frozen atoms — `_sidecar.read_frozen_atoms` needs a label to strip the rung and three of four callers omit it, so *"Hide frozen atoms"* and `runtime_info["frozen_atoms"]` are empty for every laddered calculation. **② duplication, NOT a latent defect — re-measured 2026-09-18:** `identity.parse_stage_token` is a second reader of the stage token alongside `runfiles.parse`. They differ on `_geom_optim.xyz` (a declared underscore ROLE, which the second swallows into the stage name) and on `.runwrap-*.log`. **Neither shape is ever passed to it.** Its three callers feed decks (`materialize` ×2, `job.script`) and `.out` / concluded `.molwatch.log` (`parse/dirs/job.py::_detect_stage`); measured on all five real shapes, the two readers AGREE every time. *This row said "latent" and was reported to the user as a bug waiting to happen, with an invented `.xyz` example — user: "why would you fucking pass a .xyz to a parser and ask which step this run belongs to?" Nothing does.* What is real is one grammar with two readers, worth collapsing on the one-home rule and on nothing more urgent. **③** a phantom rung for an unstaged calculation, fixed by ②. *The migration framing is gone with § 5l — these are ordinary defects in `parse/` and `identity`* | § 5l's inventory, re-measured 2026-09-17 | **① FIXED 2026-09-17. ② and ③ open.** *The fix was already in the same module.* `_siesta_fdf_path_for` — the function `model/parse.md` § 5.3 names as the shape a companion lookup may legitimately take — solves the identical problem identically: try the exact stem, then *"fall back to a single `*.fdf` in the same directory"*. `read_frozen_atoms` never got that fallback. It has one now, asked through `sidecars.molstruct.sidecars_in` (the framework's own search, § 4.5) rather than a hand-rolled glob, and **guarded twice**: the lone sidecar's label must be a prefix of the artifact's on a `_` boundary (so an unrelated sidecar that merely happens to be alone is refused), and two candidates decline rather than pick. Licensed by `project-layout.md` § 1.4 — a run directory holds one invocation's output. **Strictly additive**: it runs only where the answer was already nothing. ② and ③ remain, and are one deletion: `identity.parse_stage_token` goes, its three callers (`parse/dirs/job.py:81`, `materialize.py:394`, `:433`) move to `runfiles.parse`, which is right on both shapes the two disagree about |
| ~~**N6**~~ | execution / web | ~~the group launcher, and `files.py`'s duplicated sidecar constant~~ — **CLOSED 2026-09-18, and one of the two was never a question.** *`files.py`*: the copy is deleted; it asks `sidecars.molstruct.sidecar_path_for`, the module that owns the pairing. It needed no decision and should not have been filed as one. *The group launcher*: **withdrawn — there is no defect.** `launch/` holds the GROUP's own machinery (the sequencer, its `.sbatch`, its log, SLURM's stdout) beside the trial directories so they are not mixed among them — a deliberate decision recorded at `submit.py:1199` (*roadmap 7.10, user 2026-08-24*). I read a folder holding files as a claim about the TREE'S LEVELS and manufactured a design question out of a ruling already made. | § 5l, re-read 2026-09-18 | closed |
| ~~**N7**~~ | paths standard | ~~delete the superseded surface in favour of the three verbs~~ — **WITHDRAWN 2026-09-17** with § 5l. There is no replacement surface to migrate onto: `ref.py` is deleted, and `runfiles` + `paths` ARE the framework. The ~40 functions § 5l counted stay as they are; whether that number is itself a defect is a question § 5k's rule never raised and nothing has measured since | § 5l | withdrawn |
| **W23** | front end / ops | **The JupyterNB feature is hand-built where it should be declared — fifteen items, one plan: § 5n.** *(user, 2026-09-15: "why is jupyter.py not following a data-driven design but rather handcrafted jibberish of code?" … "use .json or .jsonl or .toml to help clean this up. this is a systematic design, not some hacking" … "make sure that you don't have other hackish code in the design".)*  The settings a framed Jupyter starts with are expressed three ways inside one function, 40 lines of real Python live inside a string literal no tool can read, the control routes are gated twice on two different facts, the tab's waiting is five ad-hoc timers, and **the whole feature has no test** — 1,610 lines in its own five files, plus the notebook half of `serve_daemon` and six CLI verbs.  The contract, the admission rule that keeps the data file from becoming a dumping ground, the sweep of the other residue, and the order of work are in **§ 5n** | § 5n · found 2026-09-15 | **all fifteen shipped 2026-09-15**, then reviewed with fresh eyes the same day — nine further defects, one of them destructive and one re-creating J13's own bug. § 5n.8 has them and they are fixed; two are recorded as **J16** and **J17** below |
| **W24** | front end / engines | **The transport tab is one panel per ENGINE, not one badge per field.** *(user, 2026-09-15: "i am confused to see mainly pyscf settings on that page while the main design should be focused on transiesta … let's separate transiesta and pySCF engine completely … why don't we use tab of different engine to separate them rather than marking each parameters".)*  Measured: of the 12 fields the tab renders, **5 name PySCF** and the only two with an engine name in the LABEL are `pyscf_functional` / `pyscf_basis` — in the NEGF section, for an engine `registered_engines()` does not list and `engine`'s own `choices` excludes. They are neither sealed nor contract-locked, so they travelled into `task.json`'s device-stage bag and merged into a config where `engine` is hardcoded `"transiesta"` and nothing reads them — the trap the schema endpoint's own docstring refuses. And card 3 claims the advanced fields "stay collapsed"; `tier: advanced` sets `opacity: 0.85` and a bullet, and collapses nothing | **contract settled in `engines/transport.md` § 3.8.8** *(restored there 2026-09-24 — the 2026-09-23 consolidation had replaced the § 3.2 that held it, and this row was its only copy)* — the `index.html` pattern (one card, a sub-tab strip, one panel and one schema endpoint per engine, one config dataclass per engine, which is what actually separates them: `SiestaConfig` and `PySCFConfig` share no field name). A known engine with no backend is a DISABLED tab saying what would make it live (the user's choice against hiding it and against live fields). `TransportConfig` keeps its name — 14 modules and 16 test files reference it — and loses both `pyscf_*` fields; the override gate's vocabulary becomes the selected engine's, so a PySCF name is refused rather than ignored | not started |
| **W25** | engines / science | **THE TRANSPORT TAB'S PARAMETER SURFACE IS INERT — measured against the installed binary, not a manual.** `molbuilder-siesta` ships **SIESTA 5.4.2**, whose fdf labels are compiled into `siesta`/`tbtrans` as literal strings, so this is countable. **Of the 12 fields the tab renders, 10 cannot affect the run:** the four transmission scalars write `TS.TBT.Emin` / `Emax` / `NumE` / `Erange.RelToEF` and `tbtrans` contains **zero** occurrences of `Emin`, `Emax`, `NumE`, `Erange` or `RelToEF` in any spelling; the three contour fields name `TS.ComplexContour.NumCircle` / `NumLine` / `Emin`, all **zero** in `siesta` (only the unused legacy `ComplexContour.NPoles` survives); `log_level` claims `WriteVerbosity`, **zero** in `siesta`. Four of those have no consumer in the tree at all, and **`contour_n_circle` reaches only the Methods paragraph**, which reports a contour the deck never carried — the one finding here with a publication consequence. fdf ignores a label nobody queries, so all of this is SILENT: the run completes and T(E) comes out on tbtrans's default grid. **What is sound:** the five-stage ladder is the standard recipe, the electrode→`.TSHS`→device→`TBT.HS` plumbing is correct and was measured live, every `%block TS.Elec.<name>` key is the right 5.x spelling, and the shared-electronic-contract invariant is the right physics. **Missing controls, each verified present in the binary:** `TBT.Contours` + `%block TBT.Contour.<name>`, `TBT.k` / `TBT.kgrid.MonkhorstPack` (T(E) needs a denser transverse grid than the SCF — the standard convergence study, inexpressible today), `TBT.Elecs.Eta`, `TBT.Contours.Eta`, `TBT.ElectronicTemperature`, `TS.Contours.nEq.Eta` / `Eq.Pole` / `nEq.Fermi.Cutoff`, the `TBT.DOS.*`/`TBT.T.*` outputs **W10** would read, `TS.Elecs.Bulk`, `bloch` (hardcoded `1 1 1`), `TBT.Spin` | § 5o · found 2026-09-15 | **contour fix landing; the rest sequenced in § 5o.5** |
| **W26** | front end / engines | **The transport tab's ORDER, and the Task setup seam — browser walk 2026-09-15 on a real cited junction.** *(user: "i want the full framework of how to do transport calculation scientifically sound/complete, and UI design to be logical, in the right sequence … and works with the 'task setup' framework".)*  The assessment was `engines/transport.md` § 3.4 until the 2026-09-23 consolidation replaced that section with the template's shape; what it recorded as SOUND: the electrode is derived from the citation's labels rather than built by hand, and the three rules that make a lead self-energy trustworthy are enforced — device `kz = 1` as an **error**, electrode `kz` dense as a warning, transverse k matched — plus the sealed electronic contract. **The Task setup seam works**: `prep-plan` answers for a transport description with the five stages, the hierarchical shape, each deck named by the producer, and the same machine/queue card every kind uses.  **What is wrong:** ① the **bias sits in the Describe card**, beside the save button, when it is the experiment — and it governs the whole non-equilibrium half of the density contour, which is inert at zero bias; it belongs at the top of the physics card. ② card 1's atom list **traps the page scroll** (a wheel scrolled rows 38→53 of 444 and left the page still; hit three ways including `Page_Up`) and puts **444 checkboxes ahead of every transport control** in the interactive order. **Diagnosed, and it is MolView's not transport's:** `.molviewer-selection-list-wrap` is `overflow-y: auto` and the list is UNVIRTUALISED, so 444 rows is ~10 screens the wheel is correctly consumed by. Three options, written 2026-09-15 and dropped from § 3.4 by the consolidation, so they live here: (a) **virtualise the atom list** (render a window, not 444 rows) — fixes both symptoms at the source and touches SIX templates that mount the panel: Molbuilder, Modify, Spectrum, Transport, Results, molview-demo; (b) **cap the list's height** — cosmetic, the trap shrinks and does not go; (c) **transport only: do not mount the atom list in card 1 at all** — card 1 exists to CHECK labels, which are assigned on the Molbuilder tab, so a 444-row editor for something uneditable here is the wrong control; scoped to this tab, no shared module touched. The document recommended (c) and said it is a UI decision that waits for a ruling. **Ruled 2026-09-24 (user): none of them.** *"there is no editing capability of molview with or without the atom list"* — MolView is a viewer on this tab, the list edits nothing, and nothing is done for MolView. ③ Task setup's empty state names only the Structure-optimization tab as a source, when the Transport tab writes `task.json` by its own door. ④ ~~Task setup's "What gets written" promises `<label>.template.toml`~~ — **WITHDRAWN, my error**: `task-setup/viewer.js` already hides the template and structure rows when `calculation === "transport"`, with § 4.1's reason in a comment. I read the EMPTY state, where there is no description to read, and blamed the kind. A claim about a kind made without selecting a folder of that kind. *(And since 2026-09-24 the template row IS shown for transport — a transport description has a template, TR1, and the shared panel writes it; only the structure pair stays hidden.)* | found 2026-09-15 · the options are in this row | ① ③ **done 2026-09-15** · ② **closed 2026-09-24, user ruling: no MolView work** · ④ withdrawn |
| **W27** | engines / execution | **Transport is not on the seven-floor stack — put it there.** *(user: "fix your fucking plan by having first a correct top-down architecture"; "why the fuck is the emit not based on template based approach".)*  The architecture and the derivation are `engines/transport.md` §§ 3.2–3.7.  **THE ROOT, in the project's own vocabulary** (`execution/architecture.md` § 2): `molbuilder/transport/` appears in **none of the seven floors**, while that document's one mention of transport claims its stages "run INSIDE the job system (each an ordinary prep/launch rung)".  Four rules broken: floor 3 renders the text of every file from a `ParameterSet` through `prepare_deck` — transport's `render_script` concatenates literal f-strings; floor 2 holds what the person asked for — transport had no template, so `TransportConfig` became the definition; `prep` is the conductor and may never decide — `_prep_transport` is a second conductor that does; floor 2 must never name a machine — `max_memory_mb`/`num_threads` sit on it.  **WHY, from the history:** `transiesta.py::render_script` 2026-06-10; the pipeline landed 2026-08-19 (`refactor(prep): the seam carries the engine's FORM`) and migrated siesta + pyscf in one commit, leaving transport behind; the composite was then built outward from the unmigrated emitter.  `template.md` § 9.2 has recorded the missing arm all along, filed as one lost feature (USER-CUSTOM) rather than as *transport cannot render from a template*.  **MEASURED:** the seed deck `prep` renders carries **13 keywords / 4 blocks** against a template offering **45 deck-reaching items**.  **EVERY KNOWN DEFECT IS DOWNSTREAM:** the seed dying at 1000 SCF iterations (`MaxSCFIterations` cannot travel from the citation — no transport emitter writes it and no transport field held it); the device deck aborting on "the continued fraction method requires at least 20 poles" (the pole *energy* is written, never `TS.Contours.Eq.Pole.N`); `TBT.k` as a bare scalar the parser rejects; `tbt_k_grid`'s unguarded transport axis; the electronic contract as two frozensets and a twice-spelled predicate.  **ORDER OF WORK IS FLOOR ORDER**, § 3.6: (1) render through `spec_for`/`DeckSpec`/`prepare_deck`, (2) no value syntax by hand, (3) validation report + read-back check, (4) `_prep_transport` stops deciding, then the floor-2 items.  My first draft of this plan put the pipeline at step 6 of 7 because it was written from the parameters down; read from the floors down it is step 1.  **DONE:** 4a the `citation` marker (`template.md` § 6.4's sibling answerer, per-kind); 4b/4c transport's parameters as 17 catalogue rows + 7 shared rows tagged `citation = ["transport"]` + 9 relaxation rows tagged `optimization`, with matching `SiestaConfig` fields — which also made `electrode_kz` (invariant I9) reachable from a description for the first time. | `engines/transport.md` §§ 3.2–3.7 · 2026-09-15 | 4a/4b/4c **done** · floor-3 migration **open** |
| **W30** | front end / engines | **THE TRANSPORT PARAMETER SURFACE — one row, absorbing six.** *(consolidated 2026-09-23 at the user's direction: "consolidate the actual design and plan"; the contract is `engines/transport.md` § 3.8, which is now the single statement and marks what it supersedes.)* **WHY THIS ROW EXISTS.** The rules for this surface were written in six places that disagreed on three questions, and a seventh was added before the other six were read. An inventory read all of them in full: four documents, **15 contradictions, six competing orders of work**. The three real disagreements — when the citation's values arrive and whether they can be changed · where the shared values are edited · what generates the form — are settled in § 3.8.0, and every other difference was restatement. **THE DESIGN, in one sentence:** *the surface is TWO surfaces — one shared panel that edits the TEMPLATE and binds all five rungs, and one per-rung form that edits a rung's OVERRIDE BAG — and both are generated from the catalogue, narrowed by kind, with the markers deciding which value lands where* (§ 3.8.2). Every failed attempt to build ONE form produced either a form that hides the shared values with nowhere to put them, or a form that offers them and is refused downstream. **THE ONE OPEN DECISION, and nothing can be built before it (§ 3.8.6): there is no marker for SHARED.** `citation` says *who supplies the default*; Class A (§ 2a.13) is larger — `species_order`, `spin_treatment`, `spin_total` and the pseudopotentials bind every rung and no run answers them. Two shapes, the user's call: a sibling marker `shared = ["transport"]` beside `citation`, or widening `citation` to mean *shared, and here is who defaults it*. **Measured cost of treating them as the same thing:** the catalogue swap of 2026-09-23 filtered on `citation`, offered `system_label` / `species_order` / `spin_treatment` / `spin_total` as per-rung overrides, and was **reverted the same day** — a `system_label` override survives into three rungs' decks and breaks the `.TS.HSX` handover while being silently inert on the other two, and a `species_order` override gives the device one orbital ordering and the leads another, which `model/chemistry.md` § 3a was written that morning to make impossible. **`prep` HAS THE SAME HOLE and it predates the swap:** its shared-value refusal also gates on `citation=True`, so a stage override of `species_order` has never been refused — the swap did not create the hole, it made it reachable from the UI. One declaration fixes both doors. **ORDER OF WORK — this replaces the six.** ① **the SHARED declaration** (the decision above), then the form, the describe door and `prep` all read that one declaration — closing the prep hole and re-enabling the reverted swap in one step; ② **the shared panel** (§ 3.8.2, § 2a.6's Class A panel) — until it exists there is nowhere in the UI to state the electronic description at all, so a hand-built structure cannot be described; ③ **unanswered `citation` rows stay VALUELESS** at `init` instead of taking catalogue defaults (§ 3.8.3) — ② and ③ land together or the plain-structure road regresses between them; ④ **the deck viewer** (§ 3.8.4), which CONSUMES the existing calculation-root reader and does not write its own — **and that reader is `calcdirs.container_or_run`, not the `checkpoint._is_bundle_root` § 5c.3 names**: a new module landed 2026-09-19 instead, so § 5c.3 step (a) is superseded (verified 2026-09-23). An earlier draft of § 3.8 designed a parallel enumerator, which is the mistake § 5c.3 warned of in advance (*"instance 14"*) — the warning was right even though its proposed owner was not; ⑤ **the per-engine panel split** (W24), which is § 5o.5 step 4 and is unchanged by this row. **WHAT THIS ROW ABSORBS, so they are read here and not acted on separately:** § 5o.5 steps 4–5 (the panel split and the missing controls, whose steps 1–3 are done or tracked in § 5o.6) · X4's phases B–D for the UI half · § 3.8's own earlier phase list · W26 ② (the MolView scroll trap, which is a MolView defect the tab suffers). **NOT absorbed and deliberately separate:** § 5c.3 (a)–(f) is the RESULTS surface reading a finished calculation and says so itself; W25/§ 5o.6's keyword work is binary-correctness, not surface; W27's floor-3 migration is architecture and its remaining items are § 3.6's 2, 6, 7, 11, 12. **DONE 2026-09-23, and it is the framework half of ①:** `role` items are kept off every form by `catalogue_to_form_schema`, per kind — `solution_method` stays a legitimate control on the Build tab and is refused for transport | `engines/transport.md` § 3.8 | **① decided 2026-09-24 (option 1, `shared`) and built; ② built; ③ the template side built on both roads (one door, `transport_template_text`) — the deck's MARK waits on `template.md` § 6.6's mechanism; **the per-rung form is a TAB PER RUNG with foldable cards and a note per rung (§ 3.8.2a, user 2026-09-24), the describe door takes per-rung bags, the router is gone**; **the `shared` and `stages` declarations reach the Task setup doors too since 2026-09-24** (no shared column, a disabled foreign cell, prep's ownership refusal — § 3.8.9 is the marker-by-door table); ④ ⑤ open** |
| **TR1–TR3** | engines / transport | **Transport's description gains a template, and the catalogue gains what the design needs.** T1: `jobset init --calculation transport` writes `<label>.template.toml` with its Class A values defaulted from the cited relaxation — today a transport folder has **no template at all**, so the shared baseline has no home and the stage table has nothing to read. T2: the three keywords with no catalogue row — `TS.HS.Save`, the equilibrium pole **count**, `TS.Voltage`. T3: `kgrid` split, because its three components fall in three classes. **Reuses `build_description` / `template_with_values`, which already narrows the catalogue by `calculation`** | § 5p · `transport.md` § 2a | **ALL DONE 2026-09-16** — TR1 § 5p.3c, TR2/TR3 § 5p.3d |
| **TR4–TR6** | engines / transport | **Transport renders through the one path.** T4: `_prep_transport` keeps the compose and hands to `resolve`, so a transport run has a `ParameterSet` with provenance and `--pipeline-log` stops being a no-op. T5: the remaining four rungs onto `spec_for` — ⚠️ **blocked on a seam question**, what a composite kind hands its renderer, since `spec_for(struct, cfg, stage_token=)` does not carry the `ComposedJunction`. T6: `TransportConfig` retires | § 5p · § 2a | **TR4 done 2026-09-16** (§ 5p.3e); TR6 half-done — `siesta_config_for` already deleted; **TR4 and TR5 DONE**; TR6 all but done — one projection survives, feeding the lifted NEGF block, and goes with it |
| **TR7–TR8** | front end / transport | **The tab reads the catalogue, and an override reaches the rung that owns it.** T7: the bespoke dataclass form is deleted for the kind-aware catalogue route plus the existing **stage table** — rows are stages, columns are `varies`, an empty cell inherits the template. T8 fixes the **live defect**: every override currently lands on the `device` bag, so a parameter the transmission owns never reaches the transmission deck. **T8 needs the `stages = [...]` declaration** (§ 5p.3), not yet approved | § 5p · § 2a.7 | **TR8 done 2026-09-16** (§ 5p.3k) — the live defect is closed. **TR7 half done** (§ 5p.3l): the seal's reason is corrected and the form-B hole closed; the interface half — § 2a.6's Panel 0 — awaits a ruling on where Class A is edited |
| **TR9–TR10** | execution / front end | **Grouping and the deliverable.** T9: the preparatory block (seed + both leads) as one submission — ⚠️ **must first reconcile with `task-setup.md` § 1's "no run-all-stages button"**; § 5p.5 has the argument, and if it does not hold T9 is withdrawn rather than the rule bent. T10: Results reads the transmission with its **treatment label** and provenance chain, so a linear-response I–V is never mistaken for a finite-bias one | § 5p · § 2a.12 | not started |
| **W29** | engines / front end | **The first validation of a junction does not exist: nothing checks that the atoms labeled `L-electrode`/`R-electrode` are the FROZEN atoms.** *(user, 2026-09-16, describing the design: "it will look for the labeled left electrodes and make sure that they are the fixed atoms … this is the first validation or check".)*  What exists is a different, LATER gate — `compose.py` checks the electrode atoms did not MOVE, by comparing the cited deck's coordinates against the `.XV`, **after the relaxation has run**. Label the electrodes and forget to freeze them and nothing objects: the relaxation runs, the leads relax, and the junction is unusable at compose time — a wasted relaxation on a metal junction. The closest thing today is a hint inside an unrelated warning (`validation/sidecar.py`: *"If you meant those atoms to be held fixed, assign them to frozen_atoms in /modify"*), which suggests but does not check. Belongs in the settings gate on the OPTIMIZATION of a junction — the run being set up when the mistake is made. Tier 2: detectable before anything runs | § 5p.3g · found 2026-09-16 | **DONE 2026-09-16** — warns rather than refuses; **ruled 2026-09-27: the warning stays** *(user: "yes, keep the warning")* -- a structure carrying electrode labels has not committed to transport, and the refusal sits at compose, where transport is the intent |
| **W28** | tests / front end | **`test_build_e2e.py::test_commit_mounts_molview_card` is FLAKY, and it is a race in the test.** It waits for the MolView card and its canvas to mount, then *immediately* reads `.molviewer-selection-count` — which renders slightly later, so on a loaded machine the text has no `" of "` and the split raises `IndexError`. **Measured 2026-09-16, interleaved against a pristine-HEAD worktree under identical load: mine 3/4 pass, HEAD 2/4.** Both flaky, so it is not a regression — and the first, unfair comparison (one HEAD run against three of mine) read it as one. The fix is to wait for the count line as well as the canvas. Recorded rather than fixed: it is a false signal for everyone who runs the suite | found 2026-09-16 while doing TR1  **RE-VERIFIED 2026-09-23 and the fix is identified.** The test is `tests/test_build_e2e.py:345-372`, unchanged in shape: it waits for `#viewer-host` to hold `.molviewer-card` AND `canvas` (`:361-366`), then on the NEXT statement reads `.molviewer-selection-count` and does `count.split(" of ")[1]` with no wait (`:371-372`). The count line is written only by `drawList` (`lib/molview/ui.js:2044`) and is the EMPTY STRING until the structure arrives, while the element itself is created during card construction (`:1703`) — and the mount deliberately precedes the load (`structure-optimization/viewer.js:1230`: *"A viewer mounts before it has a structure"*). So card + canvas can exist with an empty count, and `"".split(" of ")[1]` is `undefined`. **Test-only fix, no product change:** extend the existing `wait_for_function` to require the count's text to contain `" of "` before reading it  **FIXED 2026-09-23** — the wait now requires the count line to contain `" of "`, i.e. it waits for the STRUCTURE rather than for the card. Test-only; the mount-before-load order is deliberate and unchanged | **done 2026-09-23** |

---

### 2a. From the env-config hand-over, when it was archived *(2026-09-20)*

[`2026-09-12-env-config-handover.md`](?doc=archive/2026-09-12-env-config-handover.md)
is spent — **86 of ~95 items DONE**, re-derived from the code, its invariants
(§ 0's targets, § 3.0's Z1–Z9) reached on eight of nine phases. These are what
was left. Each carries the measurement that proved it open, so none needs
re-deriving before it is acted on.

**Seven of the eight closed on 2026-09-21, and one row is left.** Each was
re-derived against the tree before its row was removed:

* **J2** — `scheduler/probe.py` named `write_config_scope`; `probe --write`
  writes `environment.json` through `write_environment`. Docstring fixed.
* **J3** — was never open here. `diagnostics.manager_info` became the one
  reader of `conda info --json` on **2026-09-13**, the day after the hand-over
  was written, which is why it was not carried into this list at all. The
  archived body still describes the old state and its header now says so.
* **B8**, **H8**, **H8b** — `envs` (`889a9ca9`): a comment naming a module with
  no such function, two `__all__` entries that made `import *` raise, and a
  strip the control flow could not reach.
* **D9**, **D10** — `jobset` + `scripts` (`f8ffd458`): the bare `isatty` that
  could raise on the very input it was written for, and a script that honoured
  two thirds of the host-env rule.
* **F2** — **closed as a decision, 2026-09-21.** The question was whether to
  restore a deliberately broken PyPI `pyscf-properties` in the HOST env so the
  2026-09-11 observation could be pointed at. No: the fact is permanent and
  readable from what is installed — the dist-info reads `0.1.0` while
  `direct_url.json` names a git commit, which *is* the version collision
  `force=True` exists for. `recipes.py`'s comment now states that reason
  instead of citing the vanished artifact, so nothing depends on the evidence
  being recreated — and recreating it would put pyscf back in the env it must
  never be in.

| # | item | measured 2026-09-20 |
|---|---|---|
| **F1** | the A-rules whose checker was deleted | ✅ **CLOSED 2026-09-21 — by building the one checker that looked viable and measuring it.** **A7** is partly mechanised and its row now says which half. **A8** was the candidate; the checker was written (§ 3's eleven classes' fields × every signature, all from the AST) and run: **two candidates, both correct code, zero violations** — so the rule holds, and the check cannot tell a `Stage`'s `name` from *"what the user called this calculation"*, which is § 3.1's own invocation-vs-job carve-out and is semantic. Rejected, and recorded in A8's row so it is not re-attempted. **A1 and A4** cannot be checked at all. Nothing left to decide |

Two more are recorded there and are **out of scope by that document's own § 5**,
not by neglect: `script-preparation.md:202` vs `runwrap.py:2295` on who reads
the rank count, and `oauth.py`'s two independent `expanduser` readings of
`client_secret_file`. One is **deferred by your ruling**: `recipes.py:431`
resolves the CUDA version at import, so `nvidia-smi` runs on every invocation.

~~**Three fresh residues the audit turned up, not on anyone's list:**~~
✅ **ALL THREE CLOSED 2026-09-21.** `cli.py`'s two unused imports are gone.
The GHOST advice was wrong in **both** halves, not the one recorded here:
`conda env remove -n <name>` does not clear a ghost either — it resolves the
name to a prefix and refuses with `EnvironmentLocationNotFound`, there being no
directory, which is what GHOST *means* (measured 2026-09-21). And `--clean`
does fix it, by the route the note missed: the REMOVE step is skipped, but
`can_resume` is PRESENT-only, so the create still runs and restores the
directory the entry names. The message says that now.

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

**What follows is a NEW consolidation. Steps 1 AND 2 are DONE (2026-09-18);
§ 5c.2 reopened the section the same day.**
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
| ~~**the Results file picker** — `lib/results/file-picker.js`~~ | **the BROWSER** | ❌ **STRUCK 2026-09-18 — NOT A CALLER OF THIS DOOR, and it never was.** Its question is *"these five directories are one run"*, which is the **ladder's**, not a directory's; `openable` answers about one directory by construction. `stages.md` § 6.7 puts the layout in `task.json` and forbids inferring it from data, so `JobDirParser` — handed a bare path — **cannot** answer per-rung and must not be extended to try. The ladder door already exists and already reads the declared shape: `jobset/runstatus.py::jobset_status`. **What was PREDICTED here did not happen, and the prediction is the interesting part.** This row said the picker needed *"an HTTP surface over that"* — over `jobset_status`. What shipped on 2026-09-18 is `GET /api/results/dir` over **`parse.dirs`**: the blueprint imports `openable_in`, `run_status`, `labels_in`, `read_back` and `engine_of`, and imports nothing from `jobset` at all (verified 2026-09-20 — its one `jobset` mention is a comment). The per-directory door turned out to be exactly what the picker needed, because the picker lists ONE directory at a time. **The same wrong sentence had to be corrected in `web/results.md` § 2.3 and `web/presenters.md` § 1 on 2026-09-19/20**, where it was actively telling a reader to go build a route that already existed — this is its third home. *(Listed here as "the seventh consumer" from 2026-09-18 until later the same day, which made a finished migration read as unfinished every time its status was asked.)* |

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
refuse. *(**That measurement was WRONG, and it cost three CLI verbs.** It said "no
call site in the tree passes a directory to `detect()`" and named three that
pass a concrete file. There was a FOURTH: `cli.py`'s `runtime-info`,
`watch parse` and `watch tail` hand `detect()` whatever path they are given
and then use the result as a trajectory. Registering `JobDirParser` turned
their clean `UnknownFormatError` refusal into an `AttributeError` traceback —
and the first guard written for it raised inside a poll loop that retries
`ParseError`, turning the traceback into a silent INFINITE HANG. Both fixed
2026-09-18. **Registering a parser is not additive: it changes what
`detect()` answers for every caller that does not narrow.**)*

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
   of this door. ~~**§ 5c IS CLOSED.**~~ **REOPENED 2026-09-18 by § 5c.2 (open list: N8)** —
   the map asked *who calls this door* and never *what the door looks for*,
   and the door could not see PySCF's output at all.

   The picker's work is real and is `§ 5p.3p`'s: an HTTP surface over
   `jobset_status`, the ladder door that already exists.

   *(Two method failures are recorded above rather than quietly fixed,
   because both would repeat: the map was built from FUNCTION NAMES, so it
   could not see `transport/record.py` answering the same question under its
   own names; and § 5c.1 was posed as "should `RunDirResult` learn about
   runs spanning several directories" when the answer was already on the
   books — something above it composes them, and that something is
   `jobset_status`.)*

### 5c.2 Migrate onto `model/parse.md` § 5.5

*Reopens § 5c, whose completeness check asked WHO CALLS the door and never
WHAT THE DOOR LOOKS FOR. Open list: N8 (this) and N9 (the bundle).*

**The contract is [`model/parse.md` § 5.5](?doc=model/parse.md).** It settles
the design; this section is the migration and the evidence.

#### Why — measured

PySCF writes its stdout to `.pyscf.log`. Nothing collects it and nothing
reads it.

| | |
|---|---|
| PySCF run directories in the tree | 12, **all carrying an end marker** |
| of those, reporting `running` | **3** — oldest **97.6 days** |
| of those, `openable_in` returns `None` | **3** |
| `.pyscf.log` files any registered parser claims | **0 of 13** |

**The 9 that answer correctly do so from the molwatch footer, never from the
engine's own stdout.** The 3 that fail are **spectrum decks, which write no
molwatch log at all** — so the only evidence of how they ended is the file
nothing reads. That is why it is exactly those three.

#### Two questions, not one — and this corrects an earlier draft of this row

*What is a run's output* is the catalogue's. *What can a person open* is the
registry's. `.pyscf.log` is output and no parser claims it; `.spectra.json`
is what a person should see in 2 of the 3 broken directories and is **not**
output. An earlier version of this section told the discovery chain to "ask
the catalogue", which would have handed the viewer a file `detect()` refuses
and still answered `None` for both spectrum directories.

**And the third directory's `None` is not a role-list bug at all.** Rung 3
composes `<stem><role>` **with no attempt counter**, so it looks for
`water_01_coarse.pyscf.log` while the file is
`water_01_coarse-run0.pyscf.log`. `runfiles.find()` already returns every
attempt sorted by run index; the chain hand-rolls `compose` + `isfile`, which
is the door `find`'s own docstring says it replaces.

#### Order of work

Tests before code at every step: the mutation that restores the original bug
currently leaves 417 tests green.

| # | do | proves |
|---|---|---|
| **0** | revert the uncommitted attempt — it is a SECOND catalogue, stating the vocabulary in `_run_ending` that `runfiles.WRITTEN` already holds. Keep the `test_siesta_use_gpu.py` first-vs-last correction; re-land its `_pyscf_ending` and marker constants in step 2 | tree is HEAD + one correct test |
| **a** | `runfiles`: the `output` column, 3 rows, `.out` gains `engine="siesta"`; `run_output_roles()` / `stdout_roles(engine)` / `engines()` | a declaration. Gate: `manifest(…, engine="pyscf")` stops naming `.out` |
| **b** | the emitters name their end lines as constants; `_run_ending` grows the PySCF reader **importing them**, and `ending_of(path)` dispatching on `runfiles.canonical_role`; assert `set(READERS) == set(run_output_roles())` | the end text exists once. **Mutation gate:** change the emitter's constant → the reader's test fails |
| **c** | `run_status` collects by `run_output_roles()`, reads by `ending_of`; the two conclusion readers collapse to one loop | **the 3 directories answer `finished`.** Gate: re-measure all 141 — the only differences are those 3 |
| **d** | liveness → freshest run-output mtime; the speaker rule unchanged | a killed run reads `running` while the progress log is fresh, `stale` once nothing grows |
| **e** | derive the engine→role map: `contract._STDOUT_SUFFIX`, `contract._ENGINES`, `_from_cluster`'s `.fdf→siesta`, `runwrap`'s `_ext`, **and `pyscf/input.py:261`'s banner** | one spelling of PySCF's stdout, for readers **and writers** |
| **f** | `runstatus._OUTPUT_ROLES` → `run_output_roles()` | it expands to **11** roles today, including `.parse.log` — so a queued rung flips to `running` the moment molbuilder reads its directory. Latent (0 in `projects/`), one Watch poll away |
| **g** | LANDED 2026-09-18 — `openable_in` answers by DELEGATION, not a ladder: the calculation (task.json) picks the role, the catalogue names it (`result_roles`), the registry vets it (`detect()`). Fixed on the way: `runfiles.find` instead of a hand-rolled `compose` with no attempt counter, and gather across every deck before offering (role decides WHAT, mtime decides WHICH) | **sweep: 5 of 109 change, all 5 intended** — both found spectrum dirs `None`→`.spectra.json`, the generated CO2 spectrum stub→its spectrum, and the 2 predicted `slurm.*.out` false positives→`None` (no file in those dirs is claimed by any parser) |
| **h** | **REWRITTEN 2026-09-18 — the previous row said "delete `parse_dir`, `RunDirResult`, `JobDirParser`, the `DirParser` ABC", which contradicts N9 and § 5.5.** The door STAYS. What this step does: **wire the consumers to it** (`web/blueprints/watch.py`, the spectrum surface — which today takes a file path and cannot open a DIRECTORY at all), and decide the route: `detect()`/`parse_dir`, or a direct call. `cli._trajectory_parser_for`'s directory guard stays either way — those verbs ask `answers_a_trajectory()` now, which is the correct fix | the door has callers; no consumer names a reader by hand |
| **i** | DONE 2026-09-18 — swept `model/parse.md` § 5.2 (the ladder was replaced, and the section still described four rungs under the title "unchanged in behaviour"), § 5.4 + `base.py` + § 7's restatement ("dispatch each file through the registry" — which the code stopped obeying for a measured reason, so the RULE was the wrong one), § 5.0's no-consumer note, and `process/testing.md`'s door map (`openable_in` is no longer a THIN caller, by that section's own rule) | rule first, restatements second — and where code and contract disagreed, the contract was re-derived rather than the code bent to it |

#### Validation — against runs we WATCH FINISH, not found directories

The 141 directories in `projects/` are a **regression sweep only**: nobody
established what state they are in, so agreeing with them proves inertness.
Correctness is checked against jobs driven through the real UI to a known
ending — **CO2 optimization and spectrum, in both flat and hierarchical
layout** — so every expected answer is one we watched happen.

| # | check | today |
|---|---|---|
| V1 | a finished run → `finished`, from its own stdout, with no molwatch footer | 3 say `running` |
| V2 | the same directory → `openable_in` returns the file a person should see | `None` |
| V3 | a killed run → `running` while the progress log is fresh; `stale` once nothing grows | cannot be asked |
| V4 | a run whose optimizer package is missing (`SystemExit`, **no traceback**) → `failed` | cannot be asked |
| V5 | deleting `.pyscf.log` from the catalogue makes a test FAIL | **417 stay green** |
| V6 | `set(READERS) == set(run_output_roles())` | neither exists yet |
| V7 | `run_status` median on the largest real PySCF directory | 21.6 ms — target < 3 ms |
| V8 | a third engine is two edits | fourteen sites |

#### Needs a ruling

**`openable`'s preference order below rung 1.** 9 of 141 directories change;
2 are today's false positives. The other 7 need a rule: does a directory whose
only parseable file is a `.XV` or a hand-made `.xyz` offer it, or answer
`None`? The generated CO2 jobs above are what this should be decided against.

*(Settled, previously listed here in error: `.pyscf.log` does **not** get a
FileParser. The frames live in `_geom_optim.xyz`, which already has one;
`.pyscf.log` is the stdout capture, and stdout capture answers how a run
ended — `ending_of`'s question, not the registry's.)*

---

### 5c.3 A CALCULATION ROOT IS NOT A RUN DIRECTORY — the transport ladder has no view *(2026-09-18)*

*Found by driving the real Results tab in a browser after § 5c.2 step (h)
landed. No test asserts any of it, because every test asks about a RUN
directory and this is about the folder above one.*

#### The measurement

`projects/Au-BDT-Au/transport/AuBDTAu-CT` is a transport calculation root: a
`task.json`, a `job-set.json`, the composed `junction.xyz`, and five stage
directories. Asked about that one directory, the two readers disagree
completely:

```
jobset_status  -- the LADDER reader
  complete=False  first_incomplete=seed
    seed          pending   prepped, not launched (no run.json)
    electrode_L   pending   prepped, not launched (no run.json)
    electrode_R   pending   prepped, not launched (no run.json)
    device        pending   prepped, not launched (no run.json)
    transmission  pending   prepped, not launched (no run.json)

run_status     -- the RUN reader, same directory
    running   (no result file yet)
```

`jobset_status` is right. `run_status` is answering a question the directory
was never asked — **and it is what `/api/results/dir` reports**, because that
route (§ 5c.2 step h) calls `run_status` unconditionally. So the Results tab
tells a person a calculation that has never been launched is *running*.

What the tab offers there, from the door:

```
openable : None                                  <- correct: no record exists
offerable: junction.molstruct.json, junction.xyz  <- the INPUT structure
           ...and 8 files it cannot read
```

The five stage directories are in the sidebar and **the Results tab has
nothing to say about them.** It offers the structure the calculation started
from as its result.

#### What the design already requires, and has since before this was built

`engines/transport.md` § 2a.12 names three things the Results surface reads,
none of which exists:

| | |
|---|---|
| **the curve, with its treatment NAMED beside it** | linear-response and finite-bias I–V "are different claims and look identical on a plot" (§ 2a.10). The record carries `treatment` for exactly this |
| **the provenance chain** | which device run, which lead runs, which relaxation the junction came from — "a transmission curve without its chain cannot be interpreted, reproduced, or compared" |
| **the ladder's state** | "a transmission that has not run yet is *pending*, never a failure of the calculation, and a reader needs to see which of five runs is the one still outstanding" |

The reader for the first two now exists — `parse/sidecars/transport.py`, added
2026-09-18, and `result_roles("transport")` names the record so the door
offers it. **The third has no surface at all.**

#### The one thing to decide, and it is not a layout question

**A calculation root and a run directory are different kinds of thing, and
the Results tab has only one notion.** Everything else follows from how that
is answered:

* a RUN directory holds one attempt's output → `run_status`, `openable_in`,
  one file mounted in an inspector;
* a CALCULATION root holds a *plan* and N stage directories → `jobset_status`,
  a ladder, and a deliverable that is written once at the root.

**The question already has an implementation**, and it is not in the parse
layer: `checkpoint.py::_is_bundle_root` — *"does `path` declare itself the
root of one multi-directory unit of work?"* — testing for `task.json` /
`job-set.json`, with both names imported from the modules that write them.
It is private to `checkpoint`. **Giving it a public home is the fix; a second
copy in `parse/dirs` would be instance 14 of § 8**, in the week that section
was written.

#### Order of work

Tests before code at every step, and **the validation is a browser walk, not
a directory sweep** — this defect was invisible to 2 800 passing tests.

| # | do | proves |
|---|---|---|
| **a** | give `_is_bundle_root` a public home and one owner; `checkpoint` asks it | one answer to *"is this a calculation root"*, not two |
| **b** | `/api/results/dir` asks that first: a calculation root answers with `jobset_status`'s ladder, a run directory with `run_status` as now | the transport root stops reporting `running` for a calculation nobody launched |
| **c** | **✅ done 2026-09-24** as `ladder: {complete, first_incomplete, stages: [{name, seq, state, detail, dir, attempt}]}` — the payload gains the ladder for a root, `null` for a run directory | the browser can render a ladder without a second route or a second scan |
| **d** | **✅ done 2026-09-24** — the empty-state card's ladder table: one row per rung, the bench summary's state chip, the rung to resume from in the title (`results.md` § 2.4) | § 2a.12's third requirement, and the one a person needs while a five-stage run is in flight |
| **e** | the transport inspector renders the record through `transport-json` rather than `JSON.parse`, with `treatment` NAMED beside the curve | § 2a.10 / § 2a.12's first requirement. The parser landed 2026-09-18; the inspector still parses in the browser |
| **f** | the provenance chain from `record.provenance` | § 2a.12's second requirement |

#### Deliberately NOT in this section

* **The frame axis** (§ 2a.9). The record's shape is already additive and
  § 2a.11's axis rule keeps the directory tree additive; a family of curves is
  a later row, not a rewrite, and pretending to design it now would be
  designing against no data.
* **Any change to the transport TAB** (W24–W27). This is the Results surface
  reading a finished calculation; that is the parameter surface setting one up.
* **A second status vocabulary.** `jobset_status` already answers in
  `_stage_state`'s eight words and `bench-summary.js` already tones seven of
  them. The ladder view reuses both or it is a third copy.

#### What this rests on, so it can be re-derived

Every number above was produced by running the code on 2026-09-18, not read
off a comment: the two readers' disagreement, the door's `openable: None` and
its two offerable files, and the absence of any `.transport.json` in
`projects/` (0 of 110 run directories, and none at any calculation root).

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

**The paths STANDARD — one address, three verbs — is retired**, and
`molbuilder/ref.py` + `tests/test_ref_address.py` are deleted with it. § 5k is
the paths framework and stands; this was a standard proposed on top of it.

The full record — what it was, the four APIs it meant to collapse, and the
lesson the user drew (*a module with no caller is not a green step, it is an
unmerged branch living in `main`*) — is
[`archive/2026-09-17-paths-standard-retired.md`](?doc=archive/2026-09-17-paths-standard-retired.md).

## 5m. The test screen — what the audit found, sequenced *(2026-09-09)*

**§ 5k is the paths framework (§ 5l's standard on top of it was retired
2026-09-17); this is the suite.** Two audits ran under
`process/testing.md` § 3b — a four-partition gate over ~3,000 test functions
(2026-09-08) and a header pass over the ten most-undocumented files
(2026-09-09). Their EVIDENCE lives in two records and stays there:

- [`archive/2026-09-08-test-audit-findings.md`](?doc=archive/2026-09-08-test-audit-findings.md) — the
  numbered defect ledger (§ 0a), the reproductions, and the unapplied verdicts.
- [`archive/2026-09-08-test-design-findings.md`](?doc=archive/2026-09-08-test-design-findings.md) —
  the protected class: where a science test would pass for a physically wrong
  reason.

**This section is the WORK.** It exists because a record with no plan row is a
record nobody executes — which is how `archive/2026-09-08-test-design-findings.md` shipped
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
| `wrap_into_cell` declares `role = ["transport"]` and nothing answers it | inert today because `_emit_geometry` never wraps. A decorative declaration. **Resolved 2026-09-25**: retired with the knob (W33) |

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
| 10a | `web-api.md` §§ 4–5's route catalogue | `app.url_map`, **set equality**, `static` excluded | ⛔ **BLOCKED — measured 2026-09-18, and the measurement is the finding.** See below |
| 10b | `presenters.md` § 1's viewer table | the `register()` calls in `lib/inspectors/`, **set equality** on presenter name + `isResult` | ✅ **DONE 2026-09-18** — `test_inspector_registry_dispatch_js.py::test_the_documented_presenters_are_the_registered_ones`, which since 2026-09-26 runs the registry in node and compares its `list()` with the table (it read the `register()` calls by regex before). The table gained a `Presenter` column, because it was keyed by file pattern and had no name to assert on. Mutation-tested four ways, including the `makePartialInspector` default-`true` case that produced the original wrong count |
| 10c | `conventions.md` § 3's command roster | `cli.commands`, **set equality** | ✅ **DONE 2026-09-18** — `test_the_documented_command_roster_is_the_shipped_one`. Mutation-tested both directions |

**10a — why it is blocked, and what the measurement says.** The set equality is
written and runs; the document cannot pass it yet, and the fix is **not** to
relax the test.

*Parsing it is solved.* Three things had to be handled and all three are:
§ 4's tables use brace shorthand (`/api/files/{roots,list,stat,…}`), which
expands; § 5 documents the un-owned routes in prose, so both sections count;
and **blockquote lines are excluded, because that is where this document keeps
its history** — without that, every tombstone reads as a live claim.

*What is left is a content problem, in both directions:*

* **11 routes are claimed in ordinary prose and do not exist** —
  `GET /api/checkpoint/config`, `GET /api/docs/list`, `GET /login`,
  `POST /api/admin/reload`, `POST /api/build/fdf`, `POST /api/build/pyscf`,
  `POST /api/results/bundle`, `POST /api/run/install-wrapper`,
  `POST /api/selection/atoms`, `POST /api/siesta/install-pseudos`, and
  `GET /api/docs/img/<path>` (a placeholder-spelling difference from the live
  `<path:img_path>`). **At least two are conditional, not dead**: `/login`
  registers only with an `auth` config, and the document already records that
  *"a production config with rate limiting on registers a few additional
  admin/auth routes"* — so the assert needs a declared conditional set, not a
  deletion.
* **9 live routes are claimed nowhere** — every tab page: `/documents`,
  `/jupyternb`, `/molview-demo`, `/results`, `/spectrum-calculation`,
  `/structure-optimization`, `/task-setup`, `/this-machine`, plus
  `/api/docs/img/<path:img_path>`.

***`web-api.md` itself forbids the silent fix***, and it is right: *"Each needs
checking for **retired** versus **renamed** before its row is deleted, which is
task #38's sweep, not a silent edit here."* Deleting ten rows I have not
individually traced would turn a stale index into a **wrong** one, and a renamed
route that quietly loses its row is exactly the drift step 10 exists to stop.

**So 10a lands with task #38**, and this measurement is its input: the parser is
written, the two directions are separated, and the conditional-route case is
named. *(The count test stays until then — it is weak, but it is not nothing.)*

*Model to copy:* `test_doc_claims.py::test_the_documented_L1_index_is_the_enforced_one`
— it read the documented set out of the table, read the enforced set out of
the code, and asserted **both directions** with a failure naming each side's
extras. It caught `ref`'s deletion on the first run after it. *(Retired
2026-09-27 with the layer scan whose copy it compared against; the pattern
stands for a set the RUNNING product states -- the routes the app serves, the
commands the CLI accepts.)*

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

**DONE 2026-09-18 — written into
[`process/code-audit.md`](?doc=process/code-audit.md) § 1d**, beside § 1c, which
is the other thing review must carry because a test cannot. The three actions
are unchanged; **point 2's stated reason was wrong and is corrected there**, by
measurement.

1. **Name what the deletion makes UNREACHABLE one layer away.** *What was this
   the last caller of?* `/api/transport/render` was the last thing that built a
   `TransportConfig` **and validated it** — the only reason
   `_ENGINE_VALIDATORS[TransportConfig]` still dispatched. The question was
   never asked and the answer was a 316-line class. *(A second case landed the
   same day: `render_electrode_fdf` was the last caller of
   `_emit_basis_and_xc`, which then outlived it by a day in code and longer in
   prose — four separate statements asserted callers it did not have.)*
2. **Sweep for the identifier — and READ every hit, including the ones in
   docstrings.** This said *"a symbol grep for `electrode_wizard` does not match
   'the electrode wizard' … **every miss in the 2026-09-17 sweep was of this
   shape**"*. **Re-derived 2026-09-18 against the actual misses: that is the
   MINORITY shape.** There are three, and a protocol written against the prose
   one catches almost none of them:
   **(a)** the prose name carries no symbol — nothing to match;
   **(b)** the grep *hits*, in a docstring, and the hit is dismissed as "just
   prose" — `transport/__init__.py` named three **deleted modules** and a
   nonexistent decorator, and a grep for any of the four **would have hit it at
   five lines**, every time, for a day;
   **(c)** the symbol never existed, so the grep returns one hit — the prose
   inventing it — and *absence* reads as *nothing to do*: `JobMonitor`,
   `transport.transiesta.validate`, `deck.py::_transport_view`.
   The rule is therefore **not "grep harder"**: a hit inside a docstring is a
   CLAIM, and a claim gets read. Cheapest reliable form — open the package
   `__init__.py` and the module header of every file the deletion touched.
3. **Re-run step 10's membership asserts** — one command, and 10b/10c exist
   now. **Necessary, not sufficient**, which is why § 1d sits beside them: no
   membership assert would have caught `TransiestaEngine`. It was still
   imported, still registered, still green.

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

## 5q. The engine offset — scope and order of work *(W33, 2026-09-25)*

*The rule, the name, the operations and the checks are the contract's:
[`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.0.
This section is the scope and the order of work, and restates no rule; its
row is **W33** in § 2.*

### 5q.0 Why — four measured facts, one cause

All four were measured on 2026-09-25, on the fake-junction ladder
(`projects/claude-vib-ui`).

1. **TranSIESTA refused the device deck** — *"Electrode: L lies outside the
   unit-cell"*, 4 s in, before any SCF. The junction's box was anchored at its
   lowest atom by the electrode builder, deliberately (`modify.py:872–881`,
   *"the padding opens at the TOP"*). The stored corner was `−17.355` against an
   atom at `−17.355016`, so that atom sat 1.6e-5 Å below the face in every deck. The
   relaxation and the seed ran with the same atom, because plain SIESTA treats z
   as periodic; only TranSIESTA checks. molbuilder's own `.validation.txt`
   passed the deck.
2. **The Results tab drew a box the engine never had.** `watch.py::_run_periodicity_json`
   sends no corner — so the server derives one — and takes the axis kinds from a
   `.source` pair searched in the **run** directory. A ladder keeps that pair at
   the calculation root, so none was found, the axes defaulted to `isolated ×3`,
   and the derived corner (`−0.48, −0.48, −1.177`) drew the atoms centred.
   SIESTA's corner was `(0,0,0)`, and every file SIESTA wrote is flush.
3. **The same kind of junction was centred on 2026-08-29 by accident.** Before
   `20f1cca0` (2026-09-21), transport form A left every axis `isolated`, so the
   derived corner centred it (the `Au-BDT-Au` device deck: 1.2004 Å each side).
   That commit correctly stated z = `transport`, and a transport axis's corner
   rule is `bbox_min`, which is flush. Nothing caught it because no device had
   run since.
4. **The provenance never existed, and the trajectory mixes frames.** `frame_shift`,
   the stamp `structure-periodicity.md` § 6.1 clause 5 promises, is written and read by no code. The molwatch
   log's step 0 is in the design frame while the deck from the same prep call is
   in the engine frame (measured by the structure-API audit, § 1.18: a 45 Å jump between step 0 and step 1).

**The cause, once:** placement is re-derived by each reader, with rules that
depend on the axis kind, instead of computed once by one rule and recorded
where it was applied.

### 5q.1 Data structure

| | today | becomes |
|---|---|---|
| `Structure.cell_origin` | a stored field (`structure.py:132` `METADATA_FIELDS`, the check `:585`, to/from dict `:896`/`:979`, copy `:1979`, the view block `:1078–1095`, the periodicity blocks `:2140`, `:2255`) | **replaced** by `Structure.engine_offset` — absent means the rule; present only when the person assigned the origin (contract § 6.0, **D1**) |
| the corner derivers | `resolve_cell_origin` `:679`, `expected_cell_corner` `:780`, `_derived_corner_under_explicit_cell` `:840` | **removed**. `resolve_cell` and `effective_vacuum` stay: they size a box nobody typed |
| `cell.py` | composes box + corner, and `ResolvedCell` carries `origin_is_user_owned`, `corner_was_derived`, `contains_at_world_origin` and two fractional projections | **gains** `engine_offset`, `EngineFrame`, `to_engine`, `engine_frame` (contract § 6.0); `ResolvedCell` carries `engine_offset` / `box_corner`, and the three origin fields and the second projection **go**; `EngineFrame` states whether its offset was assigned |
| `periodicity_gate.py` | the `cell_origin` op (`:370–:624`), `BLOCK_KEYS`, the manual-origin regime and its notices | the op becomes `box_corner` (assign on a typed cell, or clear back to the rule), `BLOCK_KEYS` with it; the manual-origin regime becomes the assigned offset; the gate has three box states — derived, typed, typed with an assigned origin — not four |
| `modify.py` | the electrode builder states the flush corner (`:872–:902`); `calibrate_to_cell` (`:1137–:1170`) | the builder states **no** origin; `calibrate_to_cell` **retired** (**D3**) |
| a frame set | — | **one** offset, from frame 0, for every frame (contract § 6.0) |

### 5q.2 File access

| file | today | becomes |
|---|---|---|
| `.molstruct.json` sidecar — `sidecars/molstruct.py` (writer), `parse/sidecars/molstruct.py` (reader) | schema v9, carries `cell_origin` | **v10**: `cell_origin` is replaced by `engine_offset`, always written — `null` means the rule, `[0, 0, 0]` an engine's output, `−P` an origin the person assigned. The writer writes v10; the reader accepts v9 and ignores its `cell_origin`, as `model/structure.md` § 2.2 already rules for a removed key (**D2**) — a v9 corner may be the electrode builder's flush one, the placement that failed |
| every deck molbuilder writes (SIESTA `.fdf`, PySCF `.py`) | an `atom-metadata` block only when there are labels; no placement record | + a **`molbuilder engine-offset` block** for every deck (cell, offset applied, whether it was assigned, axis kinds). One writer beside `script_emit.emit_atom_metadata`, one reader beside `_extract_atom_metadata_dict` |
| engine output (`.XV`, `.out`, `.STRUCT_OUT`, `.ANI`, `.MD`, the PySCF logs) | read as bare coordinates plus a lattice | read into `engine_frame()`, offset 0 stated |
| the molwatch log (`trajectory_log/format.py` ← `jobset/prep.py`; the audit's § 1.18) | step 0 written in the design frame | step 0 written from `to_engine()` — closes that finding |
| exports — the Results export, and `cli.py:1163–1174` (`.XV` → pair) | set `cell_origin = None`, which then derives | engine coordinates + cell, v10, stating offset `0` (contract § 6.0): every run's export reloads as a typed cell with a stated origin of 0 (`watch.py`, `cli.py`) |
| transport artifacts | form A composes the `.XV` with `cell_origin: None` (`compose.py:623`) and derives | the `.XV` through `engine_frame()`, each rung through `to_engine()`. `atom-permutation.json` and `slot-provenance.json` are unaffected (indices, provenance) |

### 5q.3 Protocol agreement — the wire

* **One periodicity block**, built by one server function, `Structure.to_wire`:
  `{cell, engine_offset, resolved_cell, box_corner, axis_kind, vacuum,
  resolved_vacuum}` — `engine_offset` the offset the structure STATES (raw,
  echoed back by the client), `box_corner` where the box is drawn
  (`−engine_offset` of the coordinates it rides with). No `coordinates` tag:
  D4 made every viewer draw the coordinates of the structure it shows, so the
  structure says which they are. Every door that sends a structure spreads the
  block, and none composes its own. *(In the swap — `6c705058`, the commit
  that retired `cell_origin` for the stated `engine_offset` — 2026-09-25.)*
* **The doors**: `/api/structure/periodicity` (the Cell page), the structure
  load/save payload (`_shared._stated_periodicity` reads `engine_offset` back),
  and `/api/watch/load` / `/api/watch/data` — the Results door, which reads the
  run deck's ENGINE-OFFSET record for the axis kinds (D5; the `.source` pair for
  a run made before it) and states `engine_offset = 0` for the engine's frames.
  The transport citation viewer is still open.
* **MolView** draws the box at `box_corner`, verbatim (`render-engine.js`,
  `model-jobs.js`'s effective cell, `ui.js`'s Origin row, `model.js`'s
  `getUnitCellOrigin` — the ASSIGNED origin or `null`). The browser never
  computes a corner. *(In the swap.)*
* **The Cell page** (`modify/periodicity.js`, `templates/modify.html`): the
  origin group assigns the box's origin — three numbers, or *Use selection* at
  the atom's own unrounded position — sent as the block's `box_corner`, and
  *Derive it* became *Automatic* (**D1**). Assigned values are shown in full,
  so an Apply never re-sends a rounded origin. *(In the swap.)*
* `/api/modify/calibrate` (`web/blueprints/modify.py:656–675`) is **retired**
  (**D3**), with the `calibrate` op in `lib/molview/model-jobs.js` and
  `modify.calibrate_to_cell`.

### 5q.4 Validation

**The checks** (contract § 6.0): *the atoms fit* (fractional span `< 1`) at the
edit; *transport clearance along c* in the transport kind validator
(`validation/__init__.py`, beside the `TBT.k` checks).

**The tests** — through the road, and fewer than today:

| | what it drives | what it asserts |
|---|---|---|
| **T1** | per engine, `jobset init → prep`: a SIESTA relaxation, a SIESTA single point, the SIESTA and PySCF vibration ladders, PySCF, the five transport rungs | the deck's coordinates are design + `engine_offset`, every atom is inside with the rule's margins, and the deck's record equals the computed offset; on a typed cell with an assigned origin the deck is design − origin, its record says assigned, and an origin that leaves an atom outside is refused naming it; the vibration `freq` deck writes the relax output unchanged (a stated `0`); an atom beyond a periodic face is a warning in the report |
| **T2** | the Results door on a finished run | `box_corner = 0` and the coordinates are verbatim from the output, and the axis kinds are the deck record's (D5). The run directory is BUILT under `tmp_path` with a flush deck and output — never read from `projects/` (`testing.md` § 2a); a run with no record takes its kinds from the `.source` pair |
| **T3** | export from Results → reload | coordinates + offset is what the engine had: every run's export reloads as a typed cell with a stated offset of `0` (contract § 6.0), the box where the engine had it |
| **T4** | `prep device` on the fixture | every atom inside the cell along c; a face gap below the lead's `d/2` is warned, not refused (risk **R1**) |
| **T5** | the rule itself, API-level on constructed fixtures shaped like the junction (`tests/test_cell.py`) — the header says so | a Cartesian implementation fails it (a skewed cell's x-extent exceeds \|a\|); a widest-gap cut fails T1's slab, whose widest gap along c is inside it, as the junction's Au–S contacts (2.399 Å) were wider than its seam (2.355 Å). The junction's `[6.3684, 3.746, 18.5325]` is recomputed in the contract, not pinned |

Each is mutation-tested: break the rule, watch it fail. T1 also pins a rule
the audit found unpinned — the explicit-cell centring branch, whose deletion
left 441 tests green (A1.17).

**Retired**: the tests that pin the obsolete designs — the derived corner per
axis kind, the user-owned origin, the `cell_origin` op, the containment regimes,
the `resolved_cell_origin` wire shapes, and the nine tests asserting the
literal corner `[7.5, 7.5, 7.5]` on one fixture (the audit's § 5a). Counted at planning: 24 files and 210
references (`test_periodicity_gate.py` alone has 82, then
`test_structure_authority_roundtrip.py` 22, `test_web.py` 18,
`test_structure_periodicity.py` 16, `test_molview_model.py` 13). Unifying must
lower the count.

**"Nothing translates by hand"** is checked by a code-text review of § 5q.5's
inventory at the end of phase 4, not by a lint test.

### 5q.5 The inventory — every code site, and its phase

| site | role today | phase |
|---|---|---|
| `structure.py` | stores the corner; derives it three ways | P1 |
| `cell.py` | composes box + corner (`:186`, `:208`, `:234`) | P1 |
| `periodicity_gate.py` | the Cell-page door, the origin op and regime | P1 |
| `sidecars/molstruct.py`, `parse/sidecars/molstruct.py` | schema v9 | P1 |
| `modify.py` | builder's flush corner; `calibrate_to_cell` (retired, **D3**) | P1 |
| `cell.py` `_contains`, the three `_CONTAIN_EPS` / `_EPS` | containment re-implemented beside `Structure.cell_contains_atoms` (A1.13's containment half) | P1 |
| `validation/siesta.py:802–803` | a containment check that ignores the corner — `inv(cell)` on raw positions; the owner says inside, the check says outside (the audit's § 1.6) | P1 |
| `periodicity_gate.py:385,422,592`, `_shared.py:156` | user-facing error text citing a `§ 3c` that no longer exists | P1 / P3 |
| `siesta/input.py` | hand translation (`:871`); validators get a re-derived struct (`:990`) | P2 |
| `siesta/input.py` `_wrap_into_cell` (`:295`, `:948–956`) | the `wrap_into_cell` knob fractional-wraps atoms into a user-supplied cell — the re-wrapping contract § 6.0 forbids. Missed by the first inventory; the audit's § 8 named it an origin-rule site | P2 — **retired 2026-09-25**: the knob, its catalogue row, its validator check, the wrap branch and the tests that pinned it. A template written before then still names the item with a value, and `prep` refuses it like any unknown name until the block is deleted — kept so *(user, 2026-09-25: "retiring this unused item is correct and should be refused")*; a read-and-ignore rule was tried and reverted |
| `transport/transiesta.py` | a second hand translation (`:401–404`) | P2 |
| `pyscf/input.py` | coordinates verbatim into `gto.M` (`:137`, `:547`) | P2 |
| `script_emit.py` | the metadata block writer/reader; + the record's | P2 |
| `trajectory_log/format.py` ← `jobset/prep.py` | molwatch step 0 in the design frame | P2 |
| `transport/compose.py` | form A composes with no corner, then derives | P3 |
| `cli.py` | the `.XV` export (`:1163–1174`) | P3 |
| `web/blueprints/watch.py` | the Results periodicity block | P3 |
| `web/blueprints/_shared.py` | the payload's periodicity block | P3 |
| `web/blueprints/build.py` | the periodicity door's origin op | P3 |
| `web/blueprints/modify.py` | `/api/modify/calibrate` — retired (**D3**) | P3 |
| `lib/molview/render-engine.js`, `model-jobs.js`, `ui.js`, `model.js`, `demo.js` | draw / choose / show / commit an origin; `model-jobs.js`'s `calibrate` op (retired, **D3**) | P3 |
| `modify/periodicity.js`, `templates/modify.html` | the origin group — kept, re-founded on the assigned offset (**D1**) | P3 |
| `validation/__init__.py` | no transport clearance rule | P4 |
| `transport/sort.py`, `config/siesta.py` | comments naming the old field | P4 |
| *found by the structure review, 2026-09-25 (at `effb6ca3`):* | | |
| `jobset/prep.py::_vibration_stage_geometry` | hands `freq` the relax run's output coordinates, which the rule then re-centres — a relaxed geometry moved against the mesh (`engines/vibration.md` § 5.2a; −0.0168 Å on the H₂ e2e). A P2 regression | P1 — the output states `0` |
| `siesta/input.py` validation subject → `validation/sidecar.py` | the relaxation record's `geometry_sha256` is compared with the SHIFTED subject, so on the road a relaxed pair's record stops vouching and its checks are skipped. A P2 regression | P1 |
| `script_emit.render_deck` | a spec without `engine_frame` skips the gate and the record with a log note | P1 — refused |
| `trajectory_log/format.py` ← `jobset/prep.py` | the preview calls `to_engine(struct)` with no box, so step 0 differs from the deck whenever prep passes `cell=` | P1 — from the spec's frame |
| `cell.py` `cell.vacuum_ignored` | *"On an isolated axis it still sets where the structure sits"* — false since P2, and in every SIESTA report for a typed cell with a stated vacuum | P1 |
| `pyscf/input.py`, `pyscf/vibration_deck.py` | the pair writer bakes the design structure's sidecar beside the engine coordinates it saves | **done** (`dba66a9a`: `_sidecar_for` states 0, both decks) |
| `structure.py` `to_extxyz` / `to_ase` / `to_pyscf`, `cli.py --pyscf-atom-block` | design coordinates beside a lattice that implies the box at the origin | P4 |
| `transport/deck.py`, `validation/__init__.py` | no validation subject: the transport validators judge design coordinates in `resolve_cell()`, which for an isolated lead is not the deck's box | P4 |
| `transport/wizard.py:431` | the lead's z hand-shifted by its lowest layer — harmless (re-centred), and a translation outside the one door | P4 review |
| multi-frame pairs | "one offset from frame 0" is implemented nowhere, and no emitter takes frames | P4, with W32 |

**The documents to sweep** in P4 (restatements; the owner is
`structure-periodicity.md`, whose superseded clauses are deleted then):
`web/molview.md` (11), `model/structure.md` (11), `web/web-api.md` (6),
`model/structure-molstruct.md` (5), `engines/transport.md` (3),
`engines/vibration.md` (2), `architecture.md` (2), `science/normal-modes.md`,
`README.md`, `model/overview.md`, `backend-architecture.md` (1 each); and, found by the review, `model/parse.md` (the premise *"no deck shifts a structure that states one"*), `model/structure.md` § 2.2 (the removal rule D2 relied on), `execution/job-contracts.md` § 3.1 (the block grammar lacks ENGINE-OFFSET), `engines/vibration.md` § 5.2a, `engines/siesta.md` (centring described for the derived box only). Dated
plans and handovers are history and are not rewritten; `plans/plan.md`'s live
rows are.

### 5q.6 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0** | the name, contract § 6.0, this plan | **Done** 2026-09-25 (`a2901f4d`), and the contract revised as the decisions landed: the stated offset and the gate (`b13810f8`), centring for robustness (`790694f4`) |
| **P1** | data structure + file access: §§ 5q.1–5q.2's model and sidecar rows | T5 passes; the codec reads v9 and writes v10. **Committed**: the rule, `EngineFrame`, `to_engine`, `engine_frame`, `require_placed` (`dcfb371b`); the stated offset (`Structure.engine_offset`, in memory), the gate's containment, and the review's two regressions fixed — the vibration `freq` deck and the SIESTA relaxation record (`1df5cc24`); moving atoms only moves atoms, D6 (`4adc6832`). **Done** with the swap (`6c705058`): `cell_origin` retired to `RETIRED_METADATA_KEYS` (D2) and its derivers deleted; `cell.resolve` places the box at `−engine_offset`, with `cell.beyond_periodic_face` and a stated-offset `cell.atoms_outside`; the sidecar at v10 with `engine_offset`, readable {7, 8, 9, 10}; the gate's op `box_corner`; the electrode builder states no corner; both `.XV` readers state 0 |
| **P2** | every emitter through `to_engine`, each deck carrying its record | T1 passes for every engine. **Done** 2026-09-25 (`4a3172e1`): SIESTA, all five transport rungs, PySCF (both atom writers) and the molwatch preview; `render_deck` writes ENGINE-OFFSET and runs the gate; every prep renders from the compose record. The review of 2026-09-25 found three regressions: old templates carrying `wrap_into_cell` are refused (kept so, by decision — `d17c3ad4`); the `freq` re-centring and the relaxation record are fixed (`1df5cc24`). T1's assigned-origin clause landed with the swap (`6c705058`), for SIESTA and PySCF — and found the outside-origin refusal reaching `prep` as a traceback, fixed there (it is a `ValidationError` now). **T1 complete** (`dba66a9a`, after the review): every transport rung through prep, over a cited deck with its record and one from before the rule (D7), and the PySCF vibration deck |
| **P3** | readers and the wire (§ 5q.3), MolView and the Cell page | T2 and T3 pass, and on the dev server the browser draws what the deck says. **Committed** (`6c705058`): the wire's `engine_offset` + `box_corner` (`cell_origin` / `resolved_cell_origin` gone); MolView draws at `box_corner`; the Cell page's origin group sends `box_corner`, *Automatic* replacing *Derive it*; the Results door reads the run deck's ENGINE-OFFSET record for the axis kinds (D5) and states 0. The 28 test files naming the retired design were read in full against the contract (three agents) and revised or retired by that review: 6568 → 6552 test functions, every new or strengthened test mutation-checked (17 mutants, all red). T2 passes (`tests/watch/test_api_load.py`, its run built under `tmp_path`). **T3 passes** (2026-09-27, `tests/test_results_export_e2e.py`, one case per route a run's output takes out of the Results tab -- a SIESTA trajectory, whose cell is its output's; a PySCF trajectory, whose cell is its deck's record; SIESTA's own `<label>.xyz` through the structure preview -- each run made on the road, saved through Export → Data → Save to project and read back through the codec: the engine's cell, a stated offset of 0, the engine's coordinates. Each case was broken on purpose. M1's review found the third route stated no frame at all, and the export offering " .log_frame6"; both fixed in the same milestone). The dev-server browser check is done: 2026-09-25, on a cleared workspace store and a new project (`claude-w33`), the junction built from SMILES, saved as a v10 pair at 6 decimals, relaxed and cited for transport through the tabs, every deck carrying its record |
| **P4** | the checks, the test retirement, the document sweep | T4 passes; the review of § 5q.5 finds no hand translation. Containment is refused at the hand-off already (P1). The swap fixed what it made false in the documents — the contract's field table, op list and op table, backend rows and status lines, `structure.md` § 5, `transport.md`'s box section — and the 1.6e-5 Å miss misreported as "16 fm" in six files. **Open:** the `d/2` face-gap warning for transport rungs; the document sweep proper — deleting the superseded §§ 6, 6.1 clause 4 and 6.1a's corner column (§ 7 was rewritten to the one-Apply page on 2026-09-25, and clause 5 to the record); the test retirements the review named, **done 2026-09-25** (five confirmed by `tools/verify_subsumption.py`, one refocused where the tool showed it was the only pin, two with the API they tested); the § 5q.5 inventory rows marked P3/P4 that the swap does not reach (the design-frame exports, the transport validation subject — narrowed to a lead that states no cell — frame sets) |
| **P5** | **acceptance** — the ladder rebuilt from scratch in a new project (`claude-w33`, user 2026-09-25: "re-create the transport and spectrum tasks from scratch"): junction `structure/au333bdt`, relaxation `optimization/au333bdt-loose` (one CG step, 4926.6 s at 10 ranks, 67 SCF iterations, max \|F\| on the free BDT 1.90 → 0.79 eV/Å, every atom d/2 inside z), transport `transport/au333bdt-t` (bias 0; seed launched 2026-09-25) | the device reaches its SCF; the record is written; the Results tab shows each rung's engine frame. **In progress** |

**The commits run add → move every caller → delete**, each one green: the
old API goes in the same work item, never left as a shim, but not before its
readers have moved.

The ladder's rungs 1–3 concluded on flush decks. Re-prepped under the rule their
decks differ, so strict composition will not reuse them — that is correct, and
is why P5 re-runs them.

**The fixture**, all of it mine (`projects/claude-vib-ui`).
`structure/au333x6_bdt.xyz` + `.molstruct.json`: 120 atoms, two Au(111) 3×3×6
slabs around a BDT with its thiol hydrogens removed, built by the documented
CLI pipeline — `molbuilder smiles "SC1=CC=C(S)C=C1"` → `modify --delete 8,11`
→ `modify --orient-axis 0,5 --center midpoint` → `modify --electrode
Au:111:3x3x6@contact=2.4:registry=B:+z=5 --electrode
Au:111:3x3x6@contact=2.4:registry=A:-z=0`. Labels (0-based) `L-electrode`
66–92, `R-electrode` 39–65, `bridge` the rest; all 108 Au frozen; `c` =
37.065 Å (span + one layer spacing, set on the Cell page). Built on the
experimental `a` (4.078 Å; PBE's is 4.158) — slightly compressed gold,
consistent within the ladder. The relaxation `optimization/au333bdt-loose`:
one CG step at 0.5 eV/Å, 4949.6 s and 52 SCF iterations, max |F| on the free
BDT 3.31 → 1.15 eV/Å. The transport `transport/au333bdt-t`: the seed 4180.7 s
and 44 iterations; the leads 159.4 and 161.6 s, 15 iterations each, E_F
−1.203935 and −1.203934 eV; the device refused before its SCF (§ 5q.0, fact 1).

### 5q.7 Risks

* **R1 — the device's gaps against the lead's** *(reframed twice, 2026-09-25)*.
  What TranSIESTA requires is containment — every atom inside the cell (its
  own words, contract § 6.0, *Why centring*); `d/2` at each face is what its
  recipe recommends, and centring gives it. So the 0.21 mÅ by which the
  device's half-gap (1.1775) exceeds the lead's (1.17725) — `c` typed 37.065
  against span + spacing 37.0645 — is irrelevant to the requirement: either
  way every atom is more than a full ångström inside the cell. A lead and
  device placed by different rigid shifts is not what TranSIESTA checks (the
  leads ran flush in their own cells). T4 asserts containment and the `d/2`
  warning; only the P5 device run shows TranSIESTA accepting it.
* **R5 — the record is load-bearing before anything reads it.** `same_calculation`
  masks only volatile fields, so any change to the ENGINE-OFFSET block's text
  (a key added, the format bumped) makes every concluded rung "not the same
  calculation". Freeze the record's format before P5's ≈ 70-minute seed re-run.
* **R2 — existing runs.** A re-prep of any existing calculation renders shifted
  coordinates, so its old concluded attempts no longer count. Intended for
  transport; for a finished relaxation it means its runs do not match a re-prep.
* **R3 — egg-box.** Moving atoms against SIESTA's real-space mesh shifts total
  energies at the meV level, so an energy from before this change and one from
  after are not bit-comparable for the same structure.
* **R4 — frame sets** need one offset from frame 0, or the electrode-identity
  gate of `engines/transport.md` § 2a.9 breaks between frames.

### 5q.8 Decisions

* **D1 — the origin: the rule, unless the person assigns it** *(user,
  2026-09-25, revising the first answer: "we should add one that can allow
  user to explicitly assign the origin of the cell box as the user desire (for
  further modificaiton convenience etc). this could be handled by just write
  the -origin to the frame_offset such that the next treatment of the file will
  force the 0,0,0 to be the user specified position")*. The Cell page's origin
  group stays, on a typed cell only, and stores the offset; its rules are the
  contract's (§ 6.0, *A stated offset*). The first answer —
  *"always automatically calculated based on the cell unit and all the atoms"*
  — is still what happens wherever the person assigns nothing.
* **D2 — existing sidecars: the old corner is retired** *(user, 2026-09-25:
  "your option (a) is fine, and when files are saved, make sure no old retired
  key is written again")*. Corrected first by the structure review: three
  gates REFUSE an unknown sidecar key (`model/structure.md` § 2.2's *"ignores
  unknown keys"* contradicts its own section and the code), and every sidecar
  carries `cell_origin`. So `cell_origin` joins `RETIRED_METADATA_KEYS` — read
  and ignored, and an old structure opens with the rule's placement — and no
  writer ever emits it again. An origin a person typed by hand is assigned
  again on the Cell page. Chosen over migrating it as a stated offset because
  v9 cannot tell a typed corner from the electrode builder's flush one, and
  the flush one is the placement TranSIESTA refused.
* **D3 — calibrate: retired** *(user, 2026-09-25: "we can retire the
  calibrate button")*. The `calibrate` op, `/api/modify/calibrate`,
  `modify.calibrate_to_cell` and their tests go in P3. What calibrate was
  for — a structure saved with its box where the person wants it — is D1's
  assigned origin. **Done 2026-09-25**: the op, the route, the function, their
  tests and the clauses that described them.
* **D4 — every viewer draws the structure's own coordinates** *(user,
  2026-09-25: "the 3d viewer should just follow what coordinate is in the
  structure, and draw the cell box with the offset in mind (start from
  -offset). i don't believe the 3d viewer should do the job of translation.
  the siesta receives the correct result because the script generator/engine
  will do the correct thing to gate and translate the coordinate to siesta.
  while the 3d viewer sees the result it would see what siesta see and print
  so there is no inconsistency issues")*. A calculation page shows the design
  coordinates with the box at `−engine_offset`; the Results tab shows the
  engine's, with the box at `(0,0,0)`. No viewer translates.
* **D5 — the Results tab's axis kinds are the structure's** *(user,
  2026-09-25: "we should show the axis_info as in structure. siesta is always
  periodic, true, but the isolate axis get our additional gate of vacuum
  surrounding them and that shows in the cell box too")*: read from the
  deck's record, which carries them as the structure had them when the deck
  was written.
* **D6 — moving atoms only moves atoms** *(user, 2026-09-25: "leave the cell
  alone, moving atoms only moves atoms")*. A translation, rotation or
  orientation — of some atoms or all — changes coordinates only: the cell's
  vectors and a stated offset stay where they were. Under *Automatic* there is
  no stored corner to keep: the box is the rule's, centred on wherever the
  atoms now are. Retires § 6
  clause 5 (a whole-structure transform moved the box with the atoms) and the
  tests that pinned it. **Done 2026-09-25** (`Structure.affine`).

**D7–D16 — the review's ten questions** *(user, 2026-09-25: "go with your
recommendations on all ten"; and, on the workspace store: "you can clean up
the persistence and timeline record so you have a clean baseline to test e2e.
new project, new file, no residue state to deal with")*. Asked after the
four-agent review of the swap; each is the recommendation as put:

* **D7 — a relaxation from before the rule stays citable.** The transport
  citation states `0` for the `.XV` only when the cited deck carries its
  `engine-offset` record; otherwise the rule centres the junction, a rigid
  shift (contract § 6.0). **Done 2026-09-25** (`transport/compose.py`).
* **D8 — the dipole estimate is taken about the centre of mass**, so a charged
  system's number does not depend on where it sits in the box.
* **D9 — an append that centres the incoming structure drops its stated
  offset**: the centring reframed its coordinates, so the box centres on the
  joined atoms, and the append says so.
* **D10 — the unused placement API goes.** `engine_frame`,
  `EngineFrame.box_corner`, `ResolvedCell.corner` and `.offset_stated` have no
  reader; each engine-output door states `engine_offset=0` itself, and the
  contract's table says so.
* **D11 — the hand-off refusal has its own id**, `deck.atoms_outside`, so
  `cell.atoms_outside` carries one severity everywhere.
* **D12 — the Cell page names the atoms outside**, per axis, as the deck
  refusal does.
* **D13 — no box without a corner.** MolView draws no box when the server sent
  no `box_corner`. The store of drafts from before W33 was backed up and
  cleared rather than migrated (the second ruling above), so nothing restored
  can lack one.
* **D14 — a retired `cell_origin` is said, not dropped silently**: the load
  door names the corner it did not apply.
* **D15 — the load door's text branch loses its side blocks**
  (`atom_metadata`, `periodicity`, `info`: no caller sends them), and the
  Results door builds its structure directly.
* **D16 — the Cell-page gestures say what they read**: the ruler's picks, in
  order (user, 2026-08-31) — *Use picked atoms* / *Use picked atom*, and
  *Automatic*'s title says it empties the boxes for Apply.

**D8–D16 done 2026-09-25**, each with its test broken on purpose to watch it
fail; the review's retirements went with them, each confirmed first by
`tools/verify_subsumption.py` (§ 5q.6, P4).

---

## 5r. The structure-API cleanup — its order, and what not to fix *(A1; from the unification audit, 2026-09-25)*

*The findings are § 2's **A1** rows. The evidence, measurement by measurement, is
the audit itself, archived as the record:
[`archive/2026-09-22-unification-audit.md`](?doc=archive/2026-09-22-unification-audit.md).
This section keeps the two things the rows cannot: the order, and the
non-findings.*

### 5r.1 The order, and why

1. **The map first** — the stale comments and misleading docstrings (A1.15),
   in one pass, and **D5 before X1 ②–⑤**: a bad map is what made X1 ① read
   as a sanctioned live path. Nothing else is safe to touch until the
   documents describe the code.
2. **The four live defects that need no ruling** — A1.11 (`X`), A1.12 (the
   H-ratio on raw labels), A1.5's `.pdb` travelling name (fix `files()`'s
   `fmt`, and A1.13's root falls with it), A1.4's silent `JSONDecodeError`.
3. **V1.10 — the generator renders both halves.** The only structural change
   in the set; it also closes the half-written pair (a NaN in `info` raising
   at the sidecar write after the `.xyz` is on disk). Do A1.3's doc fix in
   the same commit — both are in `structure.md` § 2.4's four-clause block.
4. **A1.1, A1.4** — the rest of the data-loss and uncaught-exception set;
   independent, small, user-visible.
5. **A1.5** — one home for the structure-path rule, then rename-structure vs
   rename-file, the no-delete rule, and the hex check (X4 ⑤). One missing
   door and four things that grew where it should be.
6. **A1.7** — the registry seam; beside A1.1, all three are *a check that
   does not happen, and the absence is invisible*.
7. **A1.8** — the twelve `axis_kind` fallback deletions first (pure removal,
   and it makes the rest safe to read), then the partial-charge and k-hint
   doors. A1.14's instances join here. The transport no-cell arms are ruled
   and are X4 ③'s.
8. **A1.9, A1.10** — the second and third conditions: `schema_version` is two
   one-liners; the builder defects need the placeholder carried apart from
   the data, a shape decision before a fix.
9. **The origin-rule sites** — containment without the origin, and
   `wrap_into_cell` (retired 2026-09-25). **These are W33's** (§ 5q) and go with the engine
   offset, not here.
10. **A1.2** — after the CLI-stdout ruling (A1.15).
11. **A1.18** — the line-number pass, mechanical; then its behavioural list.
12. **A1.13** — each at its owner, never at the instance, starting with the
    `replace()` guard (A1.6), which is what makes the rest safe to touch.
13. **A1.17** — the three unpinned rules first (the `.pdb` one before all: a
    regression with its numbers already written down), then the six blind
    tests (worse than absent — they read as coverage), then the duplicates,
    re-running each cluster's mutant to confirm the keeper still goes red.
14. **A1.20** — residue last, and only what has a clean step-0 verdict.

**A1.21** (audit #2) comes after steps 1–4. **Standing on its own:** the 15
hand-built `structure_hash` fixtures (A1.17) agree with the writer only
because the gate is loose — convert each as its file is touched for another
reason.

### 5r.2 Do NOT "fix" these — confirmed non-findings

Recorded so nobody fixes them by analogy.

- **`frozen_atoms`'s shape is benign.** A data descriptor sends every read and
  write through `regions[FROZEN_LABEL]`, so the two cannot disagree in any
  order; `replace()` handles it by not re-passing it. **Not the defect `pbc`
  was** — the init-field spelling exists so `Structure(…, frozen_atoms=[…])`
  reaches the one place that spells the reserved label.
- **`resolve_cell` really is the one resolver.** Nothing else in the tree
  computes an effective cell.
- **The companion-lookup merge stays withdrawn.** What forbids it is a test
  pinning a deliberately more permissive guard, not `parse.md` § 5.3; a shared
  helper with the guard as a parameter would keep both. A third copy of the
  shape (`sibling_md_nc`) has neither guard.
- **The second multi-frame XYZ reader is forced**, not duplicated: it
  tolerates a torn final frame, which every live geomeTRIC run has and ase
  refuses. Unifying it would break live-run viewing.
- **Registry overlap: none** — 14 parsers against a 29-file synthetic run
  directory, no file claimed twice.
- **Bare atom-index arithmetic in `parse/`: none that carries data.** But
  `transport/transiesta.py` has four real ones (A1.14), in the block whose own
  docstring says an off-by-one *"computes transmission through a region that
  is not the molecule, and converges while doing it."*
- **The CLI's seam verdict** was raised and dropped on the user's ruling
  (2026-09-22: *"people working with CLI would know what they're doing… leave
  that out"*). X3 is the browser half only.

---


## 5s. The electronic state — charge and spin as one answer *(W34, 2026-09-25)*

*The contract is [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md)
§§ 2a–2b; this section is the order of work. The row is **W34** in § 2.*

### 5s.0 Why — measured on 2026-09-25, every one read in the code or the engine's source

* **Every layer read the raw fields and interpreted them itself.** The analyzer
  counted electrons at charge 0 and ignored periodicity (`chemistry.py`
  `analyze_structure`), so formate at −1 and a bulk gold lead were both told to
  go open-shell; the parity check used the run's charge on PySCF and, on SIESTA,
  only when the charge was typed (`validation/siesta.py`); for a metal-free
  structure the two reported one fact and could disagree.
* **Auto-detect overwrote.** It wrote `net_charge = 0` over a blank charge (the
  phosphate rule switched off), `spin_total = 0.0` beside `non-polarized`, and
  RKS/UKS over HF (`structure-optimization/viewer.js` `_applyAutoDetectToForms`,
  `lib/spectra/core.js` `applySuggested`, the two adapters); the chip never read
  the charge (`lib/detection-chip.js`).
* **Nothing travelled.** A transport calculation's spin started at the class
  default whatever the cited relaxation ran (`transport/citation_defaults.py`
  carries six pairs and the k-grid), under a caption naming the cited run; a
  charged cited run was not detected; a vibration built from a relaxed structure
  started neutral and closed-shell; a ladder stage could change either, with the
  `.DM`/`.chk` carried across unconditionally (`resolve.py`, `warm-files.toml`).
* **The charged-species science ignored the cell.** The Makov–Payne notice and
  script fired for slabs and crystals too, and SIESTA applies the same
  correction itself for a molecule in a cubic cell (`madelung.f`, printed as
  `siesta: Emadel`), which the script never read — a double count.
* **Nothing read back what ran**: no SIESTA charge or moment, no PySCF ⟨S²⟩, the
  stability outcome printed but not recorded, and a spin-polarized TBtrans run's
  `TBT_UP`/`TBT_DN` files invisible to the transport record's glob.
* **PySCF re-ruled silently**: `dft.RKS`/`scf.RHF` with `mol.spin != 0` become
  ROKS/ROHF inside PySCF.
* **The documents were wrong in four places**: *"a polarized run starts from zero
  net spin"* (SIESTA starts at maximum aligned moments), *"the parity rule runs in
  `_validate_siesta`"* (only for a typed charge), `cfg.charge` for PySCF (the
  field is `net_charge` since 2026-08-19), *"PySCF UKS at spin 0 is a free
  moment"* (it is pinned).
* **And one of those errors is printed to users.** The `config.spin_total`
  warning (`validation/siesta.py`, `_check_siesta_spin_treatment_needs_spin_total`)
  says *"the SCF then starts from zero net spin on every atom"* — SIESTA starts
  every atom at its maximum moment, aligned (`m_new_dm.F90`) — and offers
  `spin_total` as a starting point, when with `Spin.Fix` it constrains the total
  moment for the whole run.

### 5s.1 The design

The contract, § 2a: four items, one resolver, ten rules (ES1–ES10), a capability
table and the engines' semantics, each fact read from the engine's source. § 2b:
the species.

### 5s.2 Decisions *(user, 2026-09-25: "go with your recommendations on all seven")*

1. **The framework**, contract first, then code in phases (§ 5s.3). It absorbs
   the two questions held open earlier the same day — the analyzer judges the
   resolved charge (was D17) and does not read parity in a repeating cell (was
   D18).
2. **Charge and spin belong to the calculation** — a stage override of any of the
   four items is refused, as transport already refuses its shared items.
3. **Auto-detect fills only fields still at their default** — never `0` over a
   blank charge, never a pin beside a restricted treatment, and the HF/DFT
   choice kept.
4. **Transport's spin defaults from the cited run**, and a cited run carrying a
   net charge is refused.
5. **A vibration built from a relaxed structure inherits its state**, and a
   change is warned.
6. **Restricted-open is an explicit choice**, with the capability table: not
   offered on SIESTA (no such formalism), offered for a PySCF optimization or
   single point (ROHF/ROKS gradients exist), refused by name for a PySCF
   vibration (no analytic ROHF/ROKS Hessian in PySCF; the deck uses the
   analytic Hessian). `restricted` with unpaired electrons stays refused and
   names both ways out.
7. **Existing templates are migrated, not tolerated** — a one-time `jobset`
   command rewrites the old items (`spin`, the R/U inside `method`, SIESTA's
   `spin_treatment` spellings and `spin_total`); nothing reads the old names
   afterwards, and a template that still carries one is refused naming the
   command.

### 5s.3 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0** | this section, the contract (§§ 2a–2b), and a pointer at every restatement the contract supersedes (`engines/template.md` § 6.3's spin paragraph, `validation.md` §§ 2, 2.1, 9, `engines/siesta.md` §§ 4–5, `engines/pyscf.md`, `engines/vibration.md`, `engines/transport.md`, `engines/stages.md`, `science/overview.md`, `model/chemistry.md`) | the user has read the contract |
| **P1** | the items and the resolver: the catalogue rows, both configs' fields, `chemistry.electronic_state()`, every deck writer composing from the state (PySCF's class explicit, ROKS/ROHF included), the migration command, the old items deleted everywhere | every deck kind — SIESTA optimization, vibration and the five transport rungs; PySCF optimization and vibration — is written from the state, pinned through `jobset prep` per kind; the migration command turns an old template into one prep accepts |
| **P2** | the checks: parity for a finite system at the resolved charge (SIESTA's auto charge included); the recommendation at the resolved charge and periodicity; one family (ES9); the capability refusals (ES4, ES5); the charged-species checks keyed on the axis kinds; the correction script reads `siesta: Emadel`; the wording defects (*"closed-shell doublet"*, *"small Au cluster (27 atoms) … needs n ≥ 4"*, the empty `()`); the `config.spin_total` warning's premise (§ 5s.0) and the wrapper's IMAX hint (§ 5s.4) | formate at −1 and the gold lead prep without `config.spin`; a radical still gets it; a charged slab gets no Makov–Payne script; every new test mutation-checked |
| **P3** | the forms: Auto-detect fills defaults only; the chip describes the form's charge (`/api/structure/analyze` takes one); the Task setup stage columns exclude the four items; the transport caption names each value's source | on the dev server, Auto-detect leaves a blank charge blank and a typed −1 changes the chip |
| **P4** | the hand-over: `parse/fdf.py` reads `NetCharge`/`Spin`/`Spin.Fix`/`Spin.Total`; transport defaults its spin from the citation and refuses a charged one; the relaxation record carries the state, the vibration defaults from it, and the record check compares it | a polarized relaxation cited for transport yields polarized rungs; a charged citation is refused by name; a vibration of a charged relaxation starts charged |
| **P5** | the read-back: SIESTA's `.out` (net charge, fixed or converged moment) into the run record; PySCF's class, ⟨S²⟩ and stability recorded; the transport record reads both TBtrans channels; the Results tab shows asked against used, and a difference is a finding | a spin-polarized run's moment and a UKS run's ⟨S²⟩ appear on the Results tab; a two-channel transmission is read |

### 5s.4 Found on the way — named here, fixed only where a phase says so

* A fresh PySCF bundle's first `prep` says *"this calculation is already under way
  here: warm files at the root: <name>.source.xyz"* — the structure copy `init`
  writes is read as a warm file.
* The wrapper's failure hint (`runwrap.py`) still says *"SpinPolarized with
  Spin.Total unset or 0 on a d/f-shell metal also triggers IMAX=0"* — a mechanism
  retracted on 2026-09-17 (`science/overview.md`), in the retired v4 keyword.
  P2 corrects it with the other wording.
* The transport tab's shared panel is not persisted: an adopted citation or a
  restored session resets it to the defaults (UI inventory, not yet reproduced).
* The PySCF vibration record keeps the raw `net_charge` — `None` for an
  auto-detected charge — so `spectra.json` cannot say what charge ran. P5.
* `engines/transport.md` § 3.1's *"the six numbers"* names seven items (the code
  carries six pairs and the k-grid).

## 5t. The run record — scope and order of work *(W35, 2026-09-26)*

*The contract is [`model/parse.md`](?doc=model/parse.md) § 5d and
[`engines/transport.md`](?doc=engines/transport.md) § 2a.12; this is the order
of work. The row is **W35** in § 2.*

### 5t.0 Why — measured, by two inventories and a device that diverged unseen

*As measured on 2026-09-25/26, before P1 — P1's row says what changed.*

* **Three regexes for one line, all blind to TranSIESTA.** The wrapper's
  timing tee (`runwrap.py`), the monitor (`monitor.py` `_SCF_LINE`) and the
  parser (`parse/engines/siesta.py`) each match `scf:` and none matches
  `ts-scf:`: the device's timing log held 7 iterations against 1000 and gave
  *"4049.94 s/iter"* (the true rate was 27.5 s); the monitor logged *"no SCF
  progress"* for 7.6 hours; the parser gave the periodic initialization's
  energy (−437,029 eV) for a NEGF loop at −205,444 eV and 584 electrons short,
  and a live device reads *converged* from the initialization's marker.
* **Parameters are recorded as asked, never as used.** SIESTA's own
  `fdf.<timestamp>.log` has no reader; the wrapper's absent list prints
  catalogue defaults under *"the engine default applies"*
  (`negf_eq_pole_ev (catalogue default 0.0)` where SIESTA used the continued
  fraction's own 0.2507 Ry);
  PySCF's three-column read-back is printed only by the optimization script
  and read by nothing; the pseudopotentials are recorded nowhere.
* **The monitor never writes its closing lines**: the wrapper kills it and it
  handles no signal — 0 of 9 monitor logs in `claude-w33` carry
  `[UTIL-SUMMARY]`, so no run records its CPU and GPU means, nor the limit
  `[UTIL-BASIS]` states.
* **The Results tab shows no parameters**: the deck, the validation report
  and `run.json` are reachable only as raw text.
* **The transport record**: written only by `summarize run` and refused
  until the transmission has output; rung attempts picked by file time while
  `.gathered-from` goes unread; `energies_relative_to_ef` hard-coded; the
  spin channels' files missed; a TBtrans run labelled *SIESTA binary* and
  its `.out` claimed by no parser.
* **The contour**: no `contour.eq`, so TranSIESTA's 42-pole continued
  fraction — dQ −29 at step 1 and −584 at the cap; 123 poles on the same deck
  conserved the charge to 0.024 by step 4.
* **The retry**: the wrapper warm-retried the diverged device from its own
  density for three more hours.

### 5t.1 The design

`model/parse.md` § 5d: one record per attempt, four parts (computation,
setup, deck, verdict), composed on read;
three columns for every parameter; one grammar per engine family; the verdict
and its symptoms. `engines/transport.md` § 2a.12: the transport report from
the rung records, and the stated contour.

### 5t.2 Decisions *(user, 2026-09-26: "get the framework and code unified and finalized")*

Taken as a yes to the three recommendations made the same day, with the
user's own additions:

1. **The run record as scoped**, for every kind — not only transport.
2. **The contour is stated**: a circle and a tail whose lower bound sits
   below the seed's lowest eigenvalue — the manual's rule — rather than a
   pole count guessed from one measurement.
3. **On divergence the monitor warns, and the wrapper does not warm-retry**;
   stopping stays the person's call.
4. **The parameters are reported as optimization's are — and better**: with
   what the engine used, and with the deck as it ran (user).

And the same day, after P1 and its review:

5. **P2 is built with all eight of the review's corrections** (§ 5t.5; user:
   *"go with all eight"*).
6. **Framework, never a patch** (user: *"we need a systematic and framework
   level design and fix not a patching or hacking … such that all the other
   users can benefit"*). So the record is a declared table of contributors,
   one reader per file (`model/parse.md` § 5d.1b); the Run panel renders any
   record; the SCF plots draw whatever phases and criteria a run states.
7. **The transport result is shown, not listed** (user: *"plots that show
   the convergence of the calculation and … the DOS … in a more graphical
   way"*): each rung's convergence, and T(E), the DOS and the eigenchannels
   as plots. **TBtrans writes everything by default** — device DOS
   (`TBT.DOS.Gf`), spectral DOS from each electrode (`TBT.DOS.A`), electrode
   bulk DOS and transmission (`TBT.DOS.Elecs`), eigenchannels (`TBT.T.Eig`) —
   none of which it writes for a two-electrode junction unless asked
   (`Util/TS/TBtrans/m_tbt_options.F90`). P3.
8. **One monitor, every engine, reporting what is going on** (user: *"the
   monitor is a script that will be exposed to everyone. It is run as a
   batch job"*; *"do not reinvent the wheels"*): the existing
   `mb_monitor.py`, launched by the shared part of every wrapper — today its
   launch sits in the SIESTA branch and its files ship only beside `.fdf`
   decks (`runwrap.py`) — reading each run's state through a stdlib reader
   chosen by the output file's role and shipped beside it, and reporting that
   state to the webhook: phase, iteration, energy, each residual against its
   criterion, the charge in a NEGF loop, the forces in a relaxation, and the
   warnings. P4.
10. **The monitor reads through the framework, for every engine — pulled
   forward from P4 as a clean-up** (user, 2026-09-26: *"we have a job
   directory parser that handles this … a whole framework on how to parse the
   directory and judge its status and then read the results"*; *"there's a file
   that can give a copy over. We did this for everything else"*; *"clean up
   all the shit you put together … do this holistically"*). P1 fed the monitor
   the grammar as command-line flags beside a reader of its own; now the
   framework's readers travel as `runwrap.MONITOR_COMPANIONS`
   (`execution/run-reports.md` § 2.3) — `runfiles`, the grammars, the timing
   instrument's rows, the ending scan, `run_status` — the monitor launches
   for every engine and names its files through `runfiles`, and one reader
   per line: the SCF row's values, the forces' `Max` line, the `redata:`
   targets and the molwatch log's lines leave the parsers for the tables the
   parsers and the monitor both read. The record's verdict reads the same
   scan `run_status` makes.

   **Done 2026-09-26, and further than the paragraph above** (user: *"the
   whole output parser should be already written"*; *"the type of the
   calculation can be used as an input"*): **each parser is split into its
   reading pass** — `siesta_reader.SiestaReader`, `molwatch_reader.
   MolwatchReader`, stdlib, shipped — and the registered parser only builds
   arrays from it; the monitor feeds that same pass the output as it grows,
   so both side readers (`live_state`, `criteria`) are gone. The step is the
   one SIESTA states (`Begin <CG|Broyden|FIRE> opt. move = N`, `Begin FC step
   = N`; a single point states none), so the output says what a step is and
   no calculation needs telling. **The wrapper asks, it does not grep**: its
   propor hint and warm retries ask `_run_ending.py` (`_mb_ending`) over the
   output and SIESTA's stderr, whose `die` flushes stdout on node 0 alone;
   the cause is the FIRST fatal line (it was the last, which `die`'s own
   `Stopping Program` cascade always is). **What a report carries is one
   declaration** (`report_fields.py`), each field saying which engines and
   calculations can state it: the Task-setup card is offered only those
   (`GET /api/notify/report-fields`), a description naming another is
   refused at save, and the four hand-kept copies are gone
   (`engines/stages.md` § 6.9). **In direct mode the job is its process
   tree** over the cores it was launched on, and a GPU is sampled only for a
   run that uses one (`run-reports.md` § 2.1a). Checked on real runs through
   the road — SIESTA relax and FC, PySCF water (`tests/test_siesta_vibration_
   e2e.py`, `tests/test_vibration_e2e.py` now read the monitor's closing
   lines) — which found two defects the unit tests could not: the first CPU
   sample taken over two microseconds (262144 %), and a progress log the
   PySCF deck rewrites in place read on from the seed's old offset, so the
   force had no tolerance beside it. Both fixed and mutation-checked.

   **Reviewed the same day by two agents, every claim re-verified against the
   code, and corrected.** The door's cause is what SIESTA says stopped the
   run — `SCF_NOT_CONV … (required).` makes the SCF's failure fatal
   (`Src/siesta_forces.F90`), and the warm retry asks exactly that, so a
   tolerated non-convergence followed by another crash is not retried; a
   run with no output still has its stderr heard. The monitor's cores are the
   launcher's (serial, OpenMP-only, pure MPI, hybrid), a PySCF GPU run is
   sampled, a deck name the catalogue cannot read back is said in the log
   instead of killing the monitor at start. Each parser states its own
   residuals; a PySCF step's SCF iteration and ΔE, |g|, ddm reach the report;
   `n_iters` unstated is absent, not 0; every finished step is an SCF-converged message, a
   PySCF relaxation's first included; the seconds per iteration exclude a
   relaxation's step boundaries (0.84 → 0.58 s on a real H2 run); the
   process-tree reading takes no rate from a read a reaped rank crossed.
   `max_force` is offered for every run (force-constant runs and transport
   rungs state one). The shipped modules are nobody's deck: a `.py` naming no
   `JOB` is no label, so a prepped PySCF attempt has its record again, and the
   SIESTA sniffer no longer claims `siesta_reader.py` as an output. The
   Task-setup card keeps a description's report list when it could not ask
   which fields apply, and takes only the latest answer.
9. **Then the whole road in the browser** (user: *"an end-to-end test
   through the browser … all based on the web UI design and the correct
   contract … and inspect the graphical presentation of the results and the
   graphical placement of the elements in the settings of the jobs … using
   the correct CSS framework"*): P6.

### 5t.3 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0** | the contract and this plan | the user has read them |
| **P1** | **the grammar and the readers**: the SIESTA-family line table; the parser's NEGF phase (`ts-scf:`, `ts-q:`, `ts-Vha:`), the start-up echo, the charge distribution, the electrode checks, `Emadel`, start and end of run; the `fdf`-log reader; the TBtrans reader, spin channels included; the timing tee and the monitor rendered from the table; the monitor's closing summary on SIGTERM | the 2026-09-25/26 device outputs parse to 7 periodic + 1000 NEGF cycles carrying dQ; the timing log counts NEGF rows; the monitor reports progress through a NEGF loop — each pinned on a trimmed real output, mutation-checked. **Done** 2026-09-26, then **reviewed the same day by five agents and corrected**: `parse/engines/siesta_grammar.py` is the table — the SCF row of both phases built from one spelling and case-blind in every reader (the user's 2026-05-28 rule), the ending markers (moved from `_run_ending`, whose four literals the parser had retyped), `SCF cycle continued`, the TranSIESTA lines (spin-polarized columns, contour segments), the build header and the launch lines with one reader each (`read_launch_line`: serial mode is one rank, and `bench`'s private rank regex is gone), the geometry-step line; the parser's rules read the table's patterns; the cheap ending scan is phase-aware, guarded on both devices; `siesta_fdflog.py` keeps every reading of a key read twice (the pole energy: 0.1102 then 0.2507 Ry, the second in effect) on fdf's own label rule; `tbtrans.py` per spin pass; the tee and the monitor rendered from the table — the monitor reporting E_KS, not Eharris; the timing instrument per phase, one level, the headline the NEGF loop's; the monitor's stop: a lock-free flag, SIGTERM for the job's end and SIGUSR1 for a warm retry — which sent a false *it ended* until the review — no sample counted after the stop, and the wrapper waiting for its closing lines (a race the batch caught). Pinned in `tests/parse/test_siesta_negf_phase.py`, `test_engine_used_parameters.py`, `test_tbtrans_out.py`, `tests/test_monitor.py` (the shipped copy, molbuilder unimportable, both signals) and `tests/test_run_ending_one_table.py`, every one mutation-checked. **Not measured, so not pinned:** the spin-polarized TranSIESTA rows and TBtrans passes (no polarized transport run exists), `SCF cycle continued`, and the per-phase timing — its rows are the device's own, its epochs constructed until P3's first device run writes a two-phase log |
| **P2** | **the record**: `RunDirResult.record` through `parse_dir`; `/api/results/dir` serves it; a Run panel for every kind — computation, setup with asked ≠ used first, the deck, the verdict — **with the review's eight corrections, § 5t.5, decided 2026-09-26**; the record composed from a declared table of contributors; one parameters fence for both engines | on the dev server, an optimization, a vibration and each transport rung show their Run panel; the SCF plots draw each phase against its own criterion |
| **P3** | **the transport report** from the rung records, composed on read: `.gathered-from` provenance, the E_F reference checked, both spin channels, the device's NEGF facts; **shown, not listed** — each rung's convergence plotted by the SCF-progress component the trajectory viewer uses, and the results plotted: T(E) per bias point and channel, the I–V curve, the device DOS, each electrode's spectral and bulk DOS, the eigenchannels — TBtrans writing all of them by default (decision 7), read by one reader of its output kinds (`parse/engines/tbtrans.py`) | the `claude-w33` ladder, read in the browser, its plots drawn from a real TBtrans run |
| **P4** | **the contour, the divergence, and the monitor's warnings**: the device deck states its contour from the seed's eigenvalues; the settings gate refuses one that cannot cover the spectrum; no warm retry of a diverged run; the symptoms, which the monitor then warns on. *(One monitor for every engine, reading through the framework's shipped readers and reporting the run's state, is decision 10.)* | a device deck carries its contour; a diverging output yields the symptom verdict, and the monitor warns on it |
| **P5** | **the document sweep** — § 5t.4 and the inventories' stale statements | the review finds none |
| **P6** | **the whole road in the browser** (decision 9): an optimization, a vibration and a transport calculation, each from the UI through the hand-over, Task setup, the printed `prep` / `launch` verbs and the Results tab — every step through the designed doors, no script; the settings' layout and the results' plots inspected against `web/ui-contract.md` and the CSS framework | each road finishes in the browser, and the inspection's findings are fixed at their owners |

### 5t.4 Found on the way — named here, fixed where a phase says so

* The Watch tab was retired on 2026-05-19 and documents still name it —
  `execution/run-reports.md` § 2, `execution/job-system.md` and
  `execution/job-contracts.md` (not `web/trajectory.md`, which names only the
  live `/api/watch/*` routes, as this item said). P5.
* ~~A PySCF run's manifest promises the monitor, `util.csv` and SCF-timing
  files, which the wrapper never writes for PySCF (`runfiles.py`).~~ **Fixed**
  in P1: the three rows are SIESTA's.
* The SIESTA SCF-residual plot draws `DM.Tolerance` over the dHmax trace.
  SIESTA's dHmax criterion is in the `.out` for both phases —
  `redata: Hamiltonian tolerance for SCF` and `ts: SCF Hamiltonian tolerance`
  (this item said "only in the `fdf` log") — beside which of the criteria are
  required. P2.
* ~~`web/trajectory.md` says a SIESTA `.out` carries no time of day; it
  carries `>> Start of run` and `>> End of run`.~~ **Fixed** in P1.
* `engines/transport.md` §§ 6.1 and `jobset/prep.py` describe the gather's
  gate as *byte-for-byte*; it is `same_calculation`. P5.
* `execution/job-contracts.md`, `runwrap.py` and `parse/contract.py` say a
  TranSIESTA deck carries no PROVENANCE block; it does. P5.
* The SIESTA vibration record's `runtime_info` is empty, so its Host/CPU/GPU
  rows read *—*. P2.
* `transport/record.py` reads tbtrans's reported voltage and drops it, keeps
  its own copy of the current line (`_CURRENT_RE`) and globs `.TBT.AVTRANS_*`
  alone, so a polarized point would read as pending. P3: it moves onto
  `parse/engines/tbtrans.py`.
* **Found by the 2026-09-26 review, before W35** — the solver lines
  (`parse/engines/_diag.py`): SIESTA 5.4.2 prints `diag: Algorithm`, never the
  `redata: Diagonalization algorithm` / `Diag.ELPA.GPU` lines `REQUESTED`
  matches (its `diag:` and `* ProcessorY, Blocksize:` patterns do match real
  lines), so `runtime_info["siesta_diag"]` is empty for every real run and
  the trajectory viewer's solver badge is fed only by
  `tests/test_siesta_runtime_info_build.py`'s invented header — whose other
  lines (`* Running on 4 MPI processes`, `ELPA support`, `End of run:` without
  `>>`) no SIESTA prints either; `tests/test_cli_runtime_info.py` uses the
  same stub. P2 folds the solver lines into the table and retires both.
* `parse/instruments/monitor.py` reads `[UTIL-SUMMARY]` only; the limit and
  the kernel's peak on `[UTIL-BASIS]` have no reader (`model/parse.md`
  § 5c.1). P2.
* **The Makov-Payne post-process script** (`siesta/makov_payne.py`, written
  beside a charged deck) keeps its own `.out` regexes for `E_KS` and the
  cell, and defaults to `<label>.out` — a name no wrapper writes — before
  globbing `*-run*.out` for the newest. A second reader of the SIESTA family
  and a guessed name; it should read the run the way the monitor does. P5.
* **Decided (user, 2026-09-26)** — *"the monitor … is different from the
  result tab … you can make up your correct decision about that side"*: the
  monitor's closing line says whether a relaxation's geometry relaxed, as the
  output states it (`RunEnding.relaxed`). The Results tab's verdict is P2's.
* **Decided (user, 2026-09-26)** — *"can't we just put it in one single
  Python file?"*: the monitor and the modules it reads through travel as ONE
  file, `mb_monitor.pyz` (`runwrap.MONITOR_BUNDLE`), a Python zip application
  holding each module's own file unchanged; the wrapper runs it, and asks
  `mb_monitor.pyz ending …` how a run ended. Fourteen files had stood beside
  every deck.
* **Decided (user, 2026-09-26)** — *"the status can be detected … part of the
  parser for the job directory"*: the directory door answers a prepped,
  never-launched attempt `pending` and a launched, silent one `queued`, from
  its `run.json` (`run_status`'s `launch`) — the rule the jobset layer kept
  above the door, which answered the same attempt "running".
* **Decided (user, 2026-09-26)** — *"Starting and finishing message is okay.
  No stalling"*, and the SCF-convergence messages are by design: a run tells
  its channels when it starts, on each converged SCF and every N hours as
  ticked, and when it ends; the monitor judges no stall — no `[STALL]` line,
  no stall message, no `--stall-heartbeat`, no rate hidden as "stalled"
  (`run-reports.md` § 2). The stall ping came in on 2026-06-27 (`01ba00d5`)
  and went to every channel from 2026-08-26 (`958d46d2`).
* **Decided (user, 2026-09-26)** — *"It shows what it is. Not finished, error,
  or finish"*: `run_status` has no `stale` state and no age rule — a file with
  no ending is `running`, not finished, however long it is quiet
  (`running-a-job.md` § 4.2). The monitor, which sees the PID go, says a run
  with no ending and no exit recorded `failed`, stopped before its end.
  `stale` came in on 2026-06-19 (`d620b1da`).
* **Built ahead of P2, found by the 2026-09-26 review**: the run record
  (`parse/dirs/record.py`, `setup.py`), `JobDirParser` serving `status` and
  `record`, `/api/results/dir` carrying it; the Run panel
  (`lib/results/run-panel.js`, `web/results.md` § 3a, 2026-09-27); a folder
  read alone -- a SIESTA run named by hand, an output copied without its deck
  -- has its record too (`model/parse.md` § 5d.1, 2026-09-27). **Decided
  (user, 2026-09-27)**: a fact shown twice is not a fault; one fact from two
  sources is -- *"the key is to have information source unified rather than
  worrying about repeats"*; the viewers keep their lines. The trajectory
  viewer's seconds per iteration and its badge's *ended* time are read from
  their one source -- the SCF-timing log through the record's reader, and the
  output's own end (2026-09-27; its three estimates are gone). **P2 still
  owes**: the setup rows' `echo` column (§ 5d.3);
  comparing the launch rows asked with those run
  (`bench.compare_asked_to_ran`); the SCF plots by phase; the parameters
  fence narrowed to the calculation's items.
* **The 2026-09-26 review's eight questions — decided by the user
  (2026-09-26/27)**:
  1. A force-stopped run reads `failed · stopped before its end`: `run_status`
     reads the monitor's closing record after the output's ending and the
     marker (`ffe0e8c4`).
  2. Flat stages record their launch too: `<basename>.run.json` beside the
     stage's deck, read by the ladder, the folder and the run record
     (`project-layout.md` § 1.6.3, 2026-09-27).
  3. A folder read alone that holds only a product: obsolete — confirmed in
     the browser, a folder holding only a spectrum lists it and opens it in
     the spectra viewer (2026-09-27).
  4. One SIESTA ending reader, the reading pass; `ending_of(path)` the one
     door (`ffe0e8c4`).
  5. `run_status` reads SIESTA's stderr as the wrapper does: the session
     log whose first section is the run (`wrapper_log`, which now travels),
     and a stop's detail quotes the line that stopped it (2026-09-27).
  6. The memory peak shown is the job's own: the kernel's counter when the
     job has its own cgroup, the largest sample otherwise — `utilisation()`
     decides, the record and the bench summary carry it (2026-09-27).
     *Unmeasured:* on Sol's cgroup v1 the counter and the samples are read
     from the monitor's own cgroup and the limit from the job's; whether
     ranks launched into another step are inside it is for a Sol run to say.
  7. Tolerances in the residuals' units: PySCF's SCF criteria stated in eV
     in `runtime_info.scf_criteria` beside SIESTA's; the viewer draws every
     residual's line from them (2026-09-27).
  8. No `notify` block, no notification (`ffe0e8c4`).
* **Found reading the Results tab's code (2026-09-27), open**:
  * ~~loading one run parses its output three times~~ — the directory's
    metadata was composed twice per load, each composing the relaxation
    record by a full parse, and a stopped run's poll re-read the file for
    its stop reason; now the load's own parse serves all three (2026-09-27);
  * ~~the run record is composed on every folder scan and no page shows
    it~~ — the Run panel shows it (2026-09-27);
  * ~~every visit scanned its folder twice~~ — `pageshow` re-scanned on a
    fresh load too, emptying the menu the first scan had filled; the picker
    test that read it failed under load (`2d3b67e0`, 2026-09-27);
  * ~~the vibration deck states no SCF criteria~~ — both PySCF decks read
    them back off their solver through one emitter
    (`pyscf/input.emit_scf_criteria_readback`, 2026-09-27);
  * ~~two tests read old frozen outputs~~ — replaced by a SIESTA run the road
    makes and stops (`tests/test_siesta_stopped_run_e2e.py`, 2026-09-27); the
    device's `fdf` log still answers what only a device run shows (a key read
    as `CG` and `cg`).
* The runaway symptom (§ 5d.6) cannot be "|ts-Vha| exceeds 1 eV": the
  converging device swung −1.39 → +1.95 eV in its first two NEGF iterations.
  P4 sets the rule on the two measured runs.
* **W34's contract wrote `Spin.Fix` beside `Spin non-polarized`** for every
  closed-shell molecule; SIESTA stops on `Spin.Fix` at any spin but
  collinear-polarized (`read_options.F90`). Corrected in the contract
  (`science/chemistry-correctness.md` § 2a.3–2a.4, ES6) with its
  restatements (`science/overview.md`, `engines/siesta.md` — whose "below
  ~1 meV" image energy is ≈ 0.37 eV — `science/validation.md`,
  `execution/running-a-job.md`); W34's P1 builds it so.

### 5t.5 P2's design, corrected by the 2026-09-26 review — **decided: all eight** *(user, 2026-09-26)*

The review read the P2 drafts against every door the record would use and
the files real attempts leave. Its corrections:

1. **Cheap reads only.** The record never builds a trajectory on a folder
   scan — a device `.out` is MBs, and the viewer parses it on mount anyway:
   the `.out` head and tail through the table, the `fdf` log, the wrapper
   log, the instruments, the deck, `.concluded`. Per-phase convergence comes
   from the phase-aware ending scan. **`evolution` leaves the record**: its
   one reader, the plots, reads the trajectory.
2. **One door.** `/api/results/dir` asks `parse_dir` — the container and
   read-alone rules move into `JobDirParser` first, and the unread `files`
   and `active` go — and the benchmark's `summarize.parse_point` composes
   from the same computation reader.
3. **Which run.** A record describes the attempt's latest `-runN` and lists
   the earlier ones with how each ended; the `fdf` log pairs with the `.out`
   by its stamp (exact on 122 of 122 real outputs), the wrapper log by the
   first `run index` line it states.
4. **The columns from the run itself.** *Default* and *asked* as the run
   recorded them — PySCF's fence; SIESTA's wrapper fence extended to list
   every item's default — not today's catalogue. *Used* never picks one of
   several readings, and shows the engine's own echo where one exists
   (`Number of poles = 42`). Rows by engine, calculation and stage; items
   sharing a block as one row.
5. **Pseudopotentials** by file, uuid (the `.out` names both), sha256 and
   header — not "matches the library today".
6. **Symptoms are P4's**; P2 carries asked ≠ used only.
7. ~~**One home per fact on the page**: the trajectory viewer's runtime line
   and the spectrum viewer's Host/CPU/GPU rows go, in favour of the Run
   panel~~ — **superseded (user, 2026-09-27): one SOURCE per fact; a fact
   shown twice is not a fault**, and the viewers keep their lines, the
   redundancy a run copied out of its calculation keeps. The panel is carried
   by the picker's selection event and cleared when the folder changes
   (built, 2026-09-27).
8. **The SCF plots by phase**, each drawn against its own required
   criterion — dHmax against the H tolerance, dDmax against the DM tolerance,
   the NEGF dQ against the charge tolerance.

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
| **Module** | module | its role · its layer (L1/L2/L3) · its doc | **generated** from `architecture.md` § 3's index, the one list of each module's layer (the scan that classified every module, `test_layering.py`, was retired 2026-09-27) |
| **Data structure** | persisted file / schema | its shape · its version · its one reader and one writer | **written** — the doors are a design fact, not derivable |
| **Reference** | constant · citation | its value · its source · **why this value** | **written** — see 7.3 |

### 7.3 The reference index, which is the one with real content in it

Constants have one home already (`molbuilder/constants.py`), and the dialects
a quantity is written in have theirs (`molbuilder/units.py`); the rule for both
is `architecture.md` § 3.  There is no lint -- `082ba979` retired it in favour
of that written rule. What has no home is the **reasoning**, and it is
genuinely scientific. The worked example, found while measuring this:
`trajectory_log/emitter.py` retypes `HARTREE_BOHR_TO_EV_ANG = 51.42208619`
rather than deriving it from CODATA-2018 (which gives 51.422067476, ~0.4 ppm
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

*(2026-09-26: the source scans in `test_one_naming_authority.py` and
`test_config_dir_has_one_home.py` were deleted on the user's rule — a test runs
the code; what each file keeps does. 2026-09-27: `test_layering.py`'s three
scans likewise -- the layer rule is review's, `code-audit.md` § 1c (e) -- and
its one test that runs the code moved to `test_monitor_bundle_runs_alone.py`.)*

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

## 9. ATOM as a pseudopotential validator — PENDING, not started

*(User, 2026-09-19. Recorded now so the scope is honest; the immediate work
is § 9.1, which is done, and § 9.2 is future.)*

### 9.1 What the configuration-time check must guarantee — DONE

The user's rule for this level, verbatim: *"make sure that the file are
explicitly provided, the name matches the atom name, and that's pretty much
it at this level."* Two things, both at configuration time, both now enforced
by the checks the Build preflight and `prep` already share:

| | where |
|---|---|
| **explicitly provided** — `psml_lib` names a library, OR the calculation folder already holds them; never neither | `validation.siesta._check_siesta_pseudo_coverage`. With a calculation folder in hand this is answerable, so it is an **ERROR**; before one exists (Build tab) it stays a WARN, because whether the folder will supply them is not knowable yet |
| **the name matches the element** — the file for element `E` is `E.psml`, which is the name SIESTA opens | `pseudos.check_coverage` → `misnamed`, in `ERROR_STATUSES`. The map is keyed on what a file DECLARES, so a correct gold pseudopotential saved as `gold.psml` used to answer `ok` for `Au` while SIESTA could not start |

Both are refusals at the moment the calculation is configured, which is the
only moment they are cheap. Nothing below this line is required for that.

### 9.2 ATOM as an optional extra layer — PLANNED, not started

*(User, 2026-09-19: treat it the way 3DNA is treated — ask whether they have
it, look for it under the molbuilder root, and enable the extra validation
only when it is there.)*

`~/Downloads/atom-4.2.7-100` — ATOM 4.2.7, the SIESTA project's own
pseudopotential program (Froyen / Troullier / Martins, maintained by Alberto
García). Not in the repo.

**It is the 3DNA case exactly, including the licence.** `COPYRIGHT` says
**"REDISTRIBUTION OF THIS CODE IS PROHIBITED"**, so molbuilder must not
bundle, mirror or fetch it — the same standing we already give 3DNA
(`builders/backends/_threedna.py`, "Licensing"). That makes the discovery
design a copy rather than an invention:

| 3DNA today | ATOM |
|---|---|
| in-tree `<repo_root>/x3dna*/`, version-agnostic glob | `<repo_root>/atom*/` |
| completeness filter: `bin/fiber` executable **and** `config/` present | `atm` built, **and** `Tutorial/Utils/` present (the `pt.sh` driver lives there, not beside the binary) |
| `$X3DNA`, then `fiber` on PATH | `$ATOM_PROGRAM` — which `pt.sh` already honours — then `atm` on PATH |
| `x3dna*/` in `.gitignore` | `atom*/` likewise, **added now**: the ignore has to exist *before* the folder does, or a redistribution-prohibited package lands in `git status` |
| `BackendUnavailable` names where to download | same, pointing at the SIESTA pseudopotential page |

**How it gets there is the user's to do, once** (user, 2026-09-19), and it
is the 3DNA sequence unchanged: go to the site, accept the licence, download
the package, unpack it under the molbuilder root. molbuilder does no part of
that. From then on it is present, detection finds it, and the extra layer is
simply available — there is no enable switch to forget, and no state to keep
beyond the folder being there.

**What the extra layer measures.** `Tutorial/Utils/` holds the drivers:
`ae.sh` (all-electron), `pg.sh` (generation), `pt.sh` (**the pseudopotential
test**). The workflow the manual prescribes is `ae` over a series of atomic
configurations, then `pt` over the same series with the pseudopotential, and
compare — eigenvalues, and the inter-configuration energy changes the tutorial
greps as `&d`. That is **transferability**, measured rather than trusted. The
AE-vs-PS **logarithmic derivatives** (`logder.f`, plotted per channel) are the
standard **ghost-state** diagnostic. Those are precisely the two items
[`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) § 3 declares
out of scope today, and the ATOM tutorial's own warning is the argument for
them: *"You should thoroughly test a pseudopotential before using it."*

**It also broadens which formats molbuilder can accept** (user). ATOM's native
pseudopotential formats are `.vps` / `.psf` — `pt.sh` takes
`<ptname.inp> <psname.vps>` — and `Util/` converts among VPS spellings
(`cdf2vps`, `vps2cdf`, `vpsa2bin`, `vpsb2asc`). So a library need not be PSML
to be usable or checkable.

> **BROADER INPUT, SAME STRICTNESS** — stated here because it is the thing a
> second format would quietly erode. § 9.1's two rules are about the FILE, not
> about PSML: whichever formats are accepted, the source is still stated
> explicitly, and the file for element `E` is still named `E.<ext>`. Accepting
> `.vps` therefore adds one question and it must be ANSWERED, not guessed: if
> a folder holds both `Au.psml` and `Au.vps`, that is a refusal naming both,
> never a preference order. Nothing about a wider door makes it right to pick
> for the user.

**The open question that keeps this from being a task.** ATOM has
`write_psml.f90` and **no PSML reader**, and `pt.sh` reads `.vps`. So it
cannot be pointed at a downloaded PseudoDojo PSML as-is. Either a PSML→VPS
path is needed, or the layer is scoped to the formats ATOM already reads —
which is also the format-broadening above, and may be the same piece of work.
That choice has not been made, and it decides whether this validates the
library we have or the ones we could start accepting.

**Also open:** whether `atm` is built as part of an env
(`feedback_no_env_deployment_changes` — nothing is installed without asking),
and whether the check runs per prep or on demand like `molbuilder pseudo
check`.

## 10. Consolidated status — the 2026-09-19/20 session

One arc, started by an end-to-end workflow test and ended by the page
refactor that test's findings argued for, then a review round that found six
defects in the refactor itself. **All pushed** (`e06d345c..139b11b1`); both
gates green — `none2e` 9286/0 and `e2e` 164/0.

### 10.1 What is DONE

**A directory says what it is** — `calcdir.json`, § 1.4a's mechanism.
Written by the creator, read by one module, and the tuned walk it replaces
(`_CALC_SEARCH_DEPTH = 4`) deleted. Every consumer that used to guess now
asks: the parse door, the Results route, the checkpoint panel.

| | |
|---|---|
| `dc29f3df` `5334b76d` `76282e71` | the record, its contract, and the walk's deletion |
| `6ddc551a` `e808ce68` | the Results door asks what a directory IS; the page says so |
| `421d79f7` | the checkpoint panel asks instead of counting path segments |
| `75b71bb6` | an unmarked directory is not asked to invent a run state |
| `84553f7a` | a container has no run state but may have a PRODUCT |

**The render consumes the API.** The picker and the presenters take the
server's per-file answer instead of re-deriving it from filenames.

| | |
|---|---|
| `46fdfffa` `b5a7b858` | `resultCategory` takes `meta`; the contract says so; it reaches the heading and the mount |
| `06229ad8` | which files are one run is READ BACK (`runfiles.parse`), not cut out of a name by regex |

**One fact, one home.** `c286af2d` (a trial's warm files under the trial's
own label), `0ab07a14` (the browser prep writes the ledger it advertises),
`6916b4fc` (the progress channel lives in the run).

**The pseudopotential is explicit at configuration time** (user ruling).
`fcfde4c7` — the file SIESTA opens must exist and be named for its element;
`57c709d7` — the catalogue says *one source, one set, no conflicts*.
`49a3aa90` was reverted by `cd2a0b25`: two of its three guards had no
observed failure behind them, and the user's design argument against the
third was right.

**Task setup is a function of the folder** — `web/task-setup.md` § 2.1,
written long before it was kept. Four groups, and the measure of progress
was `_resetPerFolderState()` shrinking to nothing:

| | | reset list |
|---|---|---|
| `d2daa765` | the folder door, and the answer NAMES the folder | 8 |
| `97a25580` | `loadFolder` takes one answer, not four files | 8 |
| `4217f80e` | attempts and tokens come from it | 6 |
| `2f993971` | the hand-over branch paints every card the description one does | 6 |
| `bfb38c6e` | the buffers are one object, replaced whole | **0** |

`f6d9011d` — the hand-over writes its files and stops; the tab-to-tab jump
is deleted rather than fixed (user: *"skip the fancy tab to tab jump to
avoid implicit coupling"*).

### 10.2 The rule this session actually established

**An answer names its subject, and the reader checks.** `calcdir.json`'s
`of` on disk, `run_dir` / `dir` on the wire, `said.dir !== _dir` in the
page. It is the half that clearing cannot cover: there are two ways a
surface shows the wrong folder — state that LINGERS and an answer that
LANDS LATE — and Task setup had no `AbortController` among its twelve
calls, so the second was unguarded entirely.

### 10.2a The second day — the Results panel owns its folder *(2026-09-20)*

The workflow test's W1/W2/W3 are closed and the note is archived
([`archive/2026-09-19-workflow-test-findings.md`](?doc=archive/2026-09-19-workflow-test-findings.md)).
What followed was a user ruling on the tab's interaction model and a review
of the work that implemented it.

**The panel owns a directory.** Browsing the sidebar no longer moves it;
**Reload from current project dir** is the one gesture that re-points it; tab
re-entry re-reads the folder already bound; the dropdown alone decides what is
shown. `results.md` § 2.1 is the contract.

This had been decided twice before in opposite directions — retired 2026-06-09
(#301, single clicks hijacked the inspector mid-read), restored 2026-08-04
(a live BDT-Au111 job rendered as a finished run from another folder, *"every
number plausible and every number wrong"*). **The third answer works because
the header now names the bound folder and says when the sidebar has left it.**
Read the 2026-08-04 note again and the fault is its last clause — *with
nothing on screen saying so*. Following was one way to keep the panel honest;
naming the folder is the other, and it survives you scrolling around. **If
that readout goes away the subscription has to come back**, and the CSS, the
template, the contract and a test all say so.

**Three viewers were unreachable, all the same shape:** a reader that exists,
a presenter that claims the format, and no row in the parse registry —
`.transport.json` (2026-09-17), a sweep's `job-set.json`, and `.pdb`. Each was
found by asking the door what it answers for a REAL directory, never by
reading the presenter. `.pdb` then had to go through `StructureCodec`: a
`.pdb` here is a person's structure, so reading it with `from_pdb` dropped the
regions and frozen atoms its `.molstruct.json` carries.

**Six defects in the same session's own work**, found by read-only review and
each re-measured before acting: the divergence header composing a folder with
a file from a different one (the 2026-08-04 defect wearing its own
mitigation); a dropdown pick writing the panel's folder back to the sidebar
and silencing the warning; Reload taking its FILE from the sidebar; Reload no
longer re-reading for viewers that do not poll; a warning colour token that
does not exist; and a `.pdb` sniff window that refused ten real RCSB entries.

### 10.2b What this arc cost, and the two rules that came out of it

**Doctrine is not the contract.** Three failing tests were treated as
authorities and the code bent to satisfy them; checking each against the
document it cites changed the answer every time, and one cited a plan section
deleted three days earlier. Code pins to the design; a lint, ledger or
exemption that conflicts with it is the thing to fix.

**History does not belong in the file.** A comment describing what the code
used to do rots by construction, and a rotted comment is worse than none
because someone believes it — which is what `results.md` § 2.3 did on
2026-09-19, telling a reader to build a route that had shipped the same day.
One false claim was found in three separate homes. The fix is deletion; the
commit message is where history goes, being immutable and already dated.

### 10.3 What is OPEN

* **§ 9.2 — ATOM** as an optional pseudopotential validator, on the 3DNA
  model. Planned, not started; the open question is that ATOM writes PSML
  and cannot read one.
* **The `generator_mismatch` severity.** `projects/pseudopotential` mixes
  two ONCVPSP releases, and all ten files from the 4.0.1 batch are flagged
  by the physics checks (four block as `semilocal_only`, six warn as
  `partial_projectors`) while all sixty-two from 3.3.0 are clean. The
  version is a perfect predictor here, which is the evidence for making
  C4 an ERROR — but only after the library is unified, or every calculation
  touching those ten elements refuses.
* **`projects/BDT/`'s results.** Every SIESTA run under it used the
  defective v0.5 sulfur, in a thiol-gold junction where S is the binding
  atom. `Au-BDT-Au/`, `BDT-Au/`, `gasrun2` and `gasrun4` used the good one.
* ~~**The other tabs** — the Results picker never checked that the answer was
  about the folder it asked for.~~ **CLOSED 2026-09-20.** The picker now drops
  a reply whose scan is no longer the bound one (`if (boundDir !== dir) return`),
  which is the rule the rest of the session established and the one surface
  still missing it — on the tab the rule was learned on. Compared against OUR
  OWN request rather than `body.run_dir`, because the server resolves symlinks
  and a string compare there would refuse every correct answer under a
  symlinked root. Closed with it: `_metaFor` handed presenters three of the
  five fields `presenters.md` § 2 documents, so a `match()` reading
  `meta.label` or `meta.stage` got `undefined` — it returns the whole record
  now. § 2.1's rule still applies to any other page that follows the sidebar.
* **`detect()` cannot explain a refusal** *(2026-09-20)*. It fans a boolean
  `can_parse` over every registered parser, so a damaged file is
  indistinguishable from an absent one: a sweep plan truncated by a killed
  write lists as *"no result files yet"*. The parsers now say why when asked
  directly; carrying a reason through the fan-out is a registry change.
* **`task-setup` acts on `onChange`** (`viewer.js:3665-3668` subscribes both
  `onChange` and `onCommit` to the same `loadFolder`). The Results tab stopped
  doing this on 2026-09-20; this tab still does.

## 11. The test-audit consolidation — one list, re-derived 2026-09-20

**This section replaces two records.** `archive/2026-09-08-test-audit-findings.md` and
`archive/2026-09-08-test-design-findings.md` were the 2026-09-08/09 audit's two halves.
Both are archived; everything still live is here, and nothing below is
inherited on trust — each row was re-measured against the tree, and the rows
where the audits were **wrong** are marked, because that is the part a reader
would otherwise re-derive.

**How to read the ranking.** By what a wrong answer costs the SCIENCE, not by
effort. The first three produce a run that converges on the wrong thing.

### 11.1 A wrong molecule reaches the engines — DECISION NEEDED

| | |
|---|---|
| **P1** | **DECIDED AND DONE 2026-09-20.** `build_peptide` returned a peptide ALDEHYDE and said nothing. Measured: `build_peptide("ARNDC")` → `C20 H35 N9 O8 S`, 38 heavy; the C-terminal carbonyl carbon's neighbours are **H (1.032 Å)**, O (1.229 Å), C (1.52 Å) — `-C(=O)H`, not `-C(=O)OH`. A free acid is `C20 H35 N9 O9 S`, 39 heavy. One oxygen short: the OXT, which the structure kit never writes and which occurred **zero times in the repository**. Silent because the H count is 35 either way — the aldehyde H stands in for the hydroxyl H — so every atom-count check is blind by construction. **THE RULING: molbuilder does not add the oxygen.** Adding hydrogens is a deliberate exception and stays — the geometry is predictable and two independent kits cross-check each other — but placing an OXT is a structural guess the tool has no business making for the person; *"we just need the student to understand what's the limit"*. So `build_peptide` now warns, naming the compound it produced and what that costs a calculation. It **detects** rather than assumes (`"OXT" in atom_names`), so it stops the day the kit writes one. The warning rides the channel the web page, CLI and console already display. `test_the_c_terminus_limit_is_stated` pins it and is the only check in that file that can see the defect at all |

### 11.2 Silent identity corruption — the top of the old list

| | |
|---|---|
| **P2** | **CLOSED 2026-09-20 — re-measured, and it was never a gap.** The complaint was that `test_delete_preserves_metadata_in_lockstep` asserts four `len(col) == n_atoms` and nothing about contents. True, and harmless: **the risk it implies is covered by siblings.** (a) *Per-atom columns ride the same slice* — `elements`, `positions`, `atom_names`, `residue_ids`, `residue_names`, `chain_ids` are each `[struct.X[i] for i in keep]`, so a label travels with its atom by construction (user, 2026-09-20: *"internally structure label goes with per atom"*). (b) `elements` and both survivor coordinates ARE content-asserted by `test_delete_drops_listed_indices` (`test_modify.py:212`); an `elements` mutant measurably dies there. (c) The one part that is genuine arithmetic rather than a slice — remapping `regions` and `frozen_atoms`, where indices shift and entries pointing at deleted atoms must be dropped — is asserted in `tests/test_reserved_label_one_store.py`: `delete_atoms(s, [0])`, then `regions["keep"] == [2]` and `frozen_atoms == [0, 2]`. `_reindex_transport_metadata`'s docstring records why that test exists: *"a remap that forgot it silently constrained the wrong atoms."* **This row is the FOURTH instance of the error § 11.6 records** — a suite-wide absence asserted from one file's contents. It was ranked first and was not a defect at all |
| **P3** | **CLOSED 2026-09-20 — the second half was a test of `mv`.** The finding asked the file-move tests to read `regions` and `frozen_atoms` back out of a moved sidecar. User: *"the so-called moving file problem is just hallucinating — why would you not trust the file system"*. Right: a move does not alter bytes, so asserting they survived is a test of the operating system, not of molbuilder. **What CAN break is the pairing**, because a sidecar's name is derived from the structure's stem (`files.py:246`: renaming to `bridge.xyz` *"orphans `water.molstruct.json`"*) — and that IS asserted: the rename test checks the old pair is gone and `picker_root / "bridge.molstruct.json"` exists, i.e. the sidecar followed to its derived name. The remaining half — `_seed_paired` hand-writing a v7-stamped payload carrying the top-level `frozen_atoms` key that v7 removed, which `molstruct.load_text` genuinely refuses — is real but inert: the tests using it only move bytes, and the read-back that would have exposed it is the thing that should not be added. Fixture hygiene at most. **FIFTH instance of § 11.6's error** |

### 11.3 A gate that does not run, and formulas tested where they cannot fail

| | |
|---|---|
| **P4** | **CLOSED 2026-09-20 — the premise was false, and it was mine.** The row said `test_pyscf_smoke.py` is *"the only test that executes a generated script"* and that *"a rendered deck that no longer runs ships silently"*. **Neither is true.** Measured across the suite: 133 test files touch the PySCF generator, and **four execute a real generated deck in the PySCF environment, all through the production door** — `test_spectra_from_a_real_run_e2e` (`prepare_deck` + `conda run -n <env> python <deck>`, asserting `returncode == 0`), `test_trajectory_from_a_real_run_e2e` (the OPTIMIZATION path, same pattern, line 124), `test_vibration_e2e` (whole bundles), and `test_molwatch_preview` (via `run_in_env`). SIESTA is covered too (`test_transiesta_siesta_smoke_l4`, `test_siesta_keyword_smoke`, and a keywords-exist-in-the-binary check). **What is actually left is one dead file**: `test_pyscf_smoke.py` collects zero tests anywhere — no pyscf in `molbuilder`, no pytest in `molbuilder-pySCF` — and the work it was written for is done four times over by neighbours that use `conda run` correctly. Retire it or point it at the same door; either way it is housekeeping, not a gap. *(User: "all the script generators have their PySCF component as well — I really don't know where you get those claims." The claim came from the inherited audit's § 1, which said only that THIS FILE never runs, and which I generalised into a suite-wide absence. Sixth instance of § 11.6's error.)* |
| **P5** | **WITHDRAWN 2026-09-20 — trivial, and the row was the pattern not the problem.** The claim: `bulk_z_period`'s period is asserted only on uniform layers, where a wrong formula coincides, while the one non-uniform fixture discards it (`_zper, d, _n = ...`). The mechanism is true and it does not matter. **The period is `z_span + d` — one line, no branch, no edge case** — so once `d` is right the period is right by arithmetic, and `test_bulk_z_period_uses_the_median_not_the_mean` already pins the only decision in the function: median, not mean, so one relaxed surface layer cannot drag the spacing. Asserting the period would pin a typo. *(I then proposed a "better" test on the physical invariant — after tiling, the seam equals one interlayer spacing. That is worse: seam = `(min + span + d) − max` ≡ `d` identically, for any input. A tautology of the implementation, offered as physics.)* **The production chain is correct and was never in doubt**: the lead's third lattice vector is `span + d`, so the next image's first layer lands one spacing above the top — setting it to the bare span would overlap the surfaces, which is what the `+ d` prevents (user raised exactly this; measured seam +2.50 Å on the compose fixture). Nothing owed. Filed because a value was computed and not asserted, without asking whether the value could be wrong |
| **P6** | **A-DNA and B-DNA cannot be told apart if they are SWAPPED.** `test_backends.py:307` asserts `diff.max() > 0.1` Å. *The audit's stated failure mode is measurably false* — two independent B builds differ by **0.000000** (fiber is deterministic) and would fail, and A vs B moves even the least-moved atom by 3.168 Å. But both directions of a swap give ≈7.9 and pass, and a swap is exactly the plumbing bug the test exists to catch. **Fix, derived and measured:** assert rise per base pair from the C1′ z-centroids — A gives 2.548 Å/step, B gives 3.375, against Arnott canonical 2.56 / 3.38. External anchor, kills the swap, subsumes the current assertion |

### 11.4 Real, cheaper, no scientific cost

| | |
|---|---|
| **P7** | `test_not_is_complement` has a **live mutant**: `Not` complementing within the operand's own span survives all 56 tests in the file. *The audit blamed the relational assertion form; measured, that is not the cause* — its premised mutant IS caught, by `test_minus:340`. The cause is the fixture: `ByElement(Au)` reaches the last atom, so `max(operand)+1 == n` and the two readings coincide. **Fix: one line** — use an operand that stops short of the last atom |
| **P8** | `test_structure_save_endpoint.py`'s docstring promises the written pair "reads back through the load door with metadata preserved". Seven tests, **none imports `StructureCodec`**; `_browser_blob()`'s `frozen_atoms` and `cell_origin` are never read back. That is the regression the file exists for (task #75, *"every save produced a pair the app could no longer open"*) |
| **P9** | **DONE 2026-09-20.** A backend the person did not install answered **HTTP 500** — reproduced with 3DNA stubbed absent: `{"backend":"threedna"}` and `{"input":"ds,..."}` both 500, via the route's generic `except Exception`. `BackendUnavailable` existed as a dedicated type and **nothing in the repo caught it**. Now the contract's own ADVISORY bucket — `web-api.md` § 1 reserves 5xx for *"an I/O error, an engine that fell over, a bug"*, and a tool you chose not to install is none of those — so **200 with `ok: false`**, plus `reason: "backend_unavailable"` and `backend`, machine-readable so a page can disable what the box cannot do. `nucleic.py`'s duplex gate raised a bare `ValueError` for the same condition and now raises `BackendUnavailable` too: one condition, one type, one handler. The message is unchanged — `design.md` requires it to name the preconditions tried, the download URL, the licence terms and the fallbacks, and it does. Three tests, mutation-checked (removing the handler fails both shapes with *"reserves 5xx for a fault on our side"*); 106 pass across backends, status contract and rate limit. **Still open, the other half:** `/api/backends` exists and no page reads it, so Modify still offers all four |
| **P10** | The TBT window's `from`/`to` **numbers** are asserted nowhere. *The audit filed this against a dead line* — it quotes `TS.TBT.NumE`/`Emin`, keywords removed 2026-09-15 because the installed tbtrans 5.4.2 cannot read them. **And do NOT attempt its proposed fix:** whether `%block TBT.Contour`'s bounds are absolute or E_F-relative is a recorded OPEN QUESTION in four places (`transport/stages.py:94`, `config/transport.py:340`, `record.py:70`), so asserting a sign relation with E_F would assert a rule no document states. Bracketing is already structurally forced by the `range:` on the two fields. **Fix:** add `from`/`to` against `cfg` to `test_transport_au_bdt_au_validation.py`, which already parses the block properly |
| **P11** | `spectra/test_selection.py:191` promises two cases and tests one; the two take **different branches** at `selection.py:119`, and no fixture in the tree can produce the second (`_prior_with_es_on_mode` always populates it). One extra assert |
| **P12** | The no-background-poll property is guarded by **nothing**. *Both audits are wrong about why*: `test_checkpoint_js_has_no_polling_timer` does not exist — deleted 2026-09-10 in `082ba979`, one day AFTER the doc's "re-verified 2026-09-09: still in the tree". The property holds (`checkpoint.js`'s only `setInterval` is prose at line 33) and `test_checkpoint_sensor_js.py:21` still advertises it as guarantee 3 over an empty region. **Fix:** stub `setInterval` in the existing `_DOM` harness and assert no timer after `onDirectoryChange`. The blocker that deferred this is gone — node is installed and 538 JS tests now pass where 67 of 717 used to |
| **P13** | `_threedna.py:126-134` decides availability from `isfile + X_OK + isdir` and never runs the tool. A design decision about how eagerly to probe at import |
| **P14** | Three small ones: `test_annotations_fdf.py` is **order-dependent inside its own file** (`test_render_fdf_unchanged_without_annotations` passes vacuously when run alone — the leak into other files is inert, *the audit overstated the blast radius*); `TestOpsPreservePeriodicity`'s class docstring still says the ops must preserve "k-grid", which moved to `SiestaConfig`, contradicting `_assert_lattice_preserved`'s own docstring five lines away; and `test_the_composed_sweep_survives_json` cannot test what it names, because both producers are stubbed — *though the audit's "every assertion cannot fail" is too strong: the 200 and the `ok` stamp are real* |

### 11.5 Closed by re-measurement — do not re-derive these

* **`test_add_atom_zero_offset_is_advisory_not_blocked`** — NOT REAL. `tests/validation/test_geometry.py` pins all three distance bands directly, including water at 0.957 Å producing no issue, which kills the mutant the finding posits.
* **`test_isolated_axis_keeps_pbc_false_despite_cell`** — NOT REAL. Wrong layer (the test is about the codec deriving `pbc` from `axis_kind`); the physics IS checked, by `validation/geometry.py:153` `cell.image_distance`; and the fixture's nearest image is 9.26 Å, above molbuilder's own 6.0 Å criterion, so there is nothing to catch.
* **`TestOpsPreservePeriodicity`'s rotation claim** — NOT REAL. A rotation returning `cell=None` IS caught, by `TestRigidTransformMovesTheBox` in the same file (4 failures). The audit names that class and files the gap anyway.
* **`detect_layers`' `abs=1e-3`** and **`bulk_z_period`'s default approx** — the filed complaints are unfounded (both are effectively exact). What is real is P5, and separately that `LAYER_TOL_ANG = 0.5` — the decision of what counts as one layer — has no test at all.
* **The junction `2.355`** — the stated failure mode is false: the Au planes are placed by absolute `start_z` at -6.855/-4.500/+4.500/+6.855 and cannot overlap. Only the retyped constant is real, and it is molbuilder's own `fcc_lattice.json` value, so deriving it is legitimate.

### 11.4a Decisions recorded, to implement together

Taken during the one-at-a-time review so the work can be batched rather
than paid for in separate test cycles.

* **The Modify tab asks what this machine can do** *(decided 2026-09-20,
  not yet built)*. `/api/backends` exists, reports which builders resolve,
  and **no page reads it** — its own handler comment says so. Meanwhile
  `modify.html:145-197` offers all four backends and the duplex input
  unconditionally, with static prose *"(Requires 3DNA…)"*. The server half
  is now in place: a missing backend answers `200 / ok:false` with
  `reason: "backend_unavailable"` and the backend name, so the page has
  something to branch on. **The decision:** fetch `/api/backends` on Modify
  mount, disable the options that do not resolve, and hide the two-strand
  hint when no duplex-capable backend is present. Not "grey it out and hope"
  — the refusal path stays as the backstop, because a backend can vanish
  between page load and submit.

### 11.5a Rulings that close items — do not reopen without asking

* **Hydrogens are added on purpose, and that stays.** `chemistry.add_hydrogens`
  saturating a dangling valence is a deliberate design decision, not drift: the
  geometry is predictable and molbuilder runs two independent kits that
  cross-check each other's result. It is the one place the tool completes the
  structure kit's work. Adding the C-terminal **oxygen** is explicitly NOT the
  same call — harder to predict and harder to justify — which is why P1 warns
  instead of fixing. *(I had filed the silent capping as a defect of the same
  class as the peptide; it is not. Recorded so the next reader does not refile
  it.)*
* **3DNA is not probed by running it — `#80` is CLOSED, won't do.** Availability
  stays a file check. The person installs 3DNA themselves under its licence and
  is responsible for it; going a mile to verify someone else's installation is
  not this tool's job. **What must improve instead is the failure**: when it does
  not work, say exactly what failed.
* **What replaces both:** the error surface, below.

### 11.6 The pattern both audits kept repeating

`test-design-findings.md` § 4's preamble names its own recurring error —
*"a suite-wide absence asserted from one file's contents"* — and records seven
withdrawals for it. **Two more of exactly that were still in its open list**
(the two NOT REALs above), and the fresh-eyes pass found two further rows whose
stated failure mode was measurably false. Of the still-open science section,
roughly 4 in 7 rows were wrong in some load-bearing way while the underlying
concern was often real. The appendix's own warning — re-derive before acting —
held up completely, and is why every row above carries its measurement.
