# The plan — one file

**Role:** plan — **the only one.** Every open item from the nine plan
documents that preceded it lives here; those nine are archived under
`docs/archive/2026-09-01-*.md` as records of what was decided and built.
**Domain:** all
**Started:** 2026-09-01, by consolidation

> *(user, 2026-09-01: "We don't need ten plan files scattered. We want one
> plan folder or one plan file and stay with that file.")*

**Status, 2026-10-09 — consolidated.** *(user, 2026-10-09: "review your
overall design, make sure all are integrated in plan with context and details,
consolidate the plan, archive obsolete or done items, consolidate design
document, archive and make sure current document contracts are updated
correctly based on validated facts")*. Ten read-only validators read every
section against the code at `61db1660`, each claim re-read at its cited place
before anything moved. What was done, superseded or measured untrue is in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md),
verbatim, each section under its verdict. What stands here is open, with what
is left, where it is in the code and which contract owns it. The duplicate
lists that had grown beside § 2 — *Unscheduled*, the leftovers of § 0b and
§ 0c, W56's and W57's open lines, the items "carried" by archived sections —
are folded into § 2, so R3's one table is one table again.

**Cleaned 2026-10-13** *(user: "clean up plan/todo")*: the rows done since
moved to [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)
(Q17, Q18, eleven of § 2's, Q19's phase-1 record); § 5aa, the groups, built
from 2.7.0, moved whole to
[`archive/2026-10-13-5aa-groups.md`](?doc=archive/2026-10-13-5aa-groups.md),
nothing of it open (two small code items in Q19's Python list); Q13 says what
is done since; the handover is current. **Consolidated 2026-10-14** *(user:
"consolidate the plan, review the plan and document with agent holistically
and check where missing gap, error, inconsistency, design problem")*: the rows
done in 2.9.5–2.9.6 moved to
[`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md); V1.9
kept for what is left of it; the day's decisions written as rows (V1.39,
V1.40, T-CO2) and step 1's list brought to them. **Ordered 2026-10-13** *(user:
"consolidate the plan for a logical priority for the remaining work")*: the
queue and the milestone table, drifted apart and behind the work, replaced by
one list in priority order (§ 0a, *THE ORDER*). There is no other to-do list:
this file is it.

---

## ▶ HANDOVER — where we are, and where to start *(2026-10-14)*

*(user, 2026-10-09: "consolidate the plan and make sure we have all the
discussions persistent and ready so that when i come back here we can
continue without losing any context/details")* Read this first; each line
points at the row or section that holds the detail.

**In progress — 2.9.6** *(2026-10-13/14, committed, not yet pushed: step 1
of the order, its push at the step's end)*: T1, F-MDNC, F19b; the pin of V1.9
(M2m, sidecar v13); A1.11, A1.5 / A1.13's `.pdb` name, A1.4; the held-set rule
of `engines/vibration.md` § 2.3a (holding more for the spectrum is
information; the clarification written for transport too, `transport.md`
§ 4); C and O pseudopotentials checked in for the tests. Rows archived in
[`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md);
the changelog's 2.9.6 row says each.

**Decided with the user on 2026-10-14, not built** — each a row of § 2, in
step 1's order: **V1.40**, a continuing PySCF stage's starting geometry checked
at launch through the codec (the job keeps its plain read; the geometry's
sidecar carried with it); **T-CO2**, the relaxation-to-vibration road on CO2;
**V1.39**, a vibration's result stating its structure whole — schema 8, the
hash retired, the sweep comparing structures within 10⁻⁶ Å, a frame set citing
the result alone, and `<label>.spectra.nc` beside the JSON (NetCDF-4 in
hierarchical groups; `netcdf4` added to the PySCF environment, approved) —
designed in `engines/vibration.md` § 6.10.

**Before that — 2.9.0 to 2.9.5** *(2026-10-13, each pushed)*: the k-point
wording in one vocabulary (2.9.0); the 2026-10-11 walk's leftovers (2.9.1);
the first phase of Q19's three passes — small UI, CSS, Python — each
reviewed by fresh agents on the full code and contract text, the none2e
batch run, the one confirm unified on the one dialog component (2.9.2);
the server's numerical threads capped at one (2.9.3, in effect on port
8888); Task setup as a checklist — every card folds, its head saying its
answer, the rail one card at a time, Save in a bar pinned to the window, and
the task's Prep after the group tabs (2.9.4, the user's request; `web/
task-setup.md` §§ 7a, 9a.1, 11); a direct launch exiting 1 with the status's
words when its run failed, every SIESTA deck stating its Hamiltonian test
(`SCF.H.Tolerance`), and the SCF plot drawing every residual the engine
converges on, each declared once an engine and served with the run (2.9.5;
`web/trajectory.md` § 3.0). Q19's row holds the detail and what its reviews
left open.

**Before that**: Q17, transport over a set of frames (2026-10-10, § 5z,
archived); Q18, the CSS sweep (2026-10-10); § 5aa, the groups — a stage's
group in `task.json`, every group prepared at once, a group's inputs gathered
at its first launch (2.7.0, archived 2026-10-13).

**molbuilder 2.0.0** *(2026-10-11, tag `v2.0.0`; the user: "we are closer
to release")*: the version has one home (`molbuilder/__init__.py`), every deck
and run script records it, and every version has its line in the changelog
(`process/conventions.md` § 2, *Versioning*; `process/changelog.md`). **The
numbering, disciplined 2026-10-13** (the user: "middle number means new
functionality, major number means major framework change"; "we will discipline
from now on"): the middle number for new functionality only, the last for
everything else, once per push -- 2.1.0 to 2.9.0 moved the middle number for
any change to a file, too loosely, and stand as given.

**Next, in order** (§ 0a, *THE ORDER*, consolidated 2026-10-13): **1** wrong
or lost results; **2** finish what the framework half-built; **3** tidy the
code and the pages (Q19's later phases among it); **4** prove the whole road
in the browser; **5** the documents; **6** new science, each designed with
the user first.

**The AuBDTAu workflow is the user's, on Sol**, once the framework is
established *(the user, 2026-10-10: "no, you should not run it on this
workstation"; "we focus on workflow and validation of design of framework,
not yet on science. i'll test real system on sol")*. Found for it on
2026-10-10: `AuBDTAu/spectrum/siesta-vib`'s own result says its reference
geometry is not a stationary point (0.37 eV/Å on the free atoms against its
0.01 eV/Å tolerance; MeshCutoff 300 Ry against the relaxation's 350 Ry); its
`job-set.json` and `optimization/min_force_relax`'s predate the 2026-10-06
job keys (`jobset migrate`); and the walk's AuBDTAu relaxation
(`claude-transport-walk/optimization/au-dta-relax2/01_coarse/run-1`) ran with
one k-point on a slab periodic in x and y — check a relaxation's mesh before
citing it.

**Decisions that wait on the user** (§ 0a, *THE ORDER*, lists them). **Only Sol
can show**, when the user is there: the junction on Sol (E13), the Sol memory
measurement, a group's header under a real scheduler, the AuBDTAu workflow.

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

### THE ORDER — the remaining work, by priority *(2026-10-13)*

> *(user, 2026-10-13: "consolidate the plan for a logical priority for the
> remaining work")* — proposed by Claude from a read of every open row and
> section; **the user's to change**. It replaces the queue (the Q rows) and
> the milestone table (the M rows), which had drifted apart and behind the
> work — transport's surface (Q5, M5) was finished on 2026-10-11 and still
> listed, M2j was closed and still listed, M8's CSS was the same work as
> Q19's. Both, as they stood, are in
> [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md).

**Why this order.** The user's goal, 2026-10-10: *"we focus on workflow and
validation of design of framework, not yet on science"*. So: first what can
give a wrong answer without saying so; then the framework pieces left half
built; then the tidy-up, while nothing new is being built on the code; then
the whole road proved in the browser on the result; then the documents,
written from what is then true; new science last, each designed with the user
first. Inside a step the listed order holds; every row is re-derived against
the code when it is taken (§ 2's rule), and a step closes as a milestone does
(below).

| # | step | what it is, plainly | its items — the detail is in § 2 or the section named |
|---|---|---|---|
| **1** | **Wrong or lost results** | places where molbuilder can show, read or keep the wrong data and not say so | *Done 2.9.5–2.9.6, archived 2026-10-14:* T1, F-MDNC, F19b, W40, the pin of V1.9 (M2m), A1.11, A1.4, the `.pdb` half of A1.5 / A1.13, and the held-set rule of `engines/vibration.md` § 2.3a. **Open, in order:** § 5r steps 3–4 (V1.10 a structure pair half written, A1.1 a left-handed cell passed in silence); X3 (the Cell page commits a box with no seam verdict); PDB-T (a PDB title past column 80); F14 (the force over every atom, the held ones included, and a slab's pressure never shown, TranSIESTA's *FORCES WRONG* not captured; `max_force` offered to a transport rung); K17 (`ParallelOverK`, the pseudopotential name fallback, the geomeTRIC log — § 5w.1); PL-R's defect (`summarize` reports the latest of a trial's runs without saying there were two); RMP (§ 5ab: the remapped state stated at every output — V1.41's force-constant prep among its doors); V1.40 (a continuing PySCF stage's geometry unchecked); V1.43 (a cited pair's electrodes checked against what its relaxation held — the user's word owed); T-CO2 (the relaxation-to-vibration road on CO2); **then 2.9.6 is pushed**, and V1.39 (a vibration's result states its structure, in a JSON and a NetCDF) is **2.10.0**, new functionality |
| **2** | **Finish what the framework half-built** | pieces whose design is agreed and partly built, so two ways of doing one thing live side by side | M3 the run record (W35, § 5t.3: P2's remainder, P4 the contour, divergence and the monitor's warnings); M6 P5 (W34, § 5s.3: the charge and spin a run actually used, read back and set against what was asked); M2k the GPU claim and match, and W36 ⑦ (an inherited thread count guarded); § 5w's remainder (M11: K13 one atomic writer, K14 each engine's parallel model as data, K16 the wrapper fence, K18 a PySCF run's level of theory, K9 each engine default declared once with its source); Q11 (the named-queue check in one door); Q8 (one `proxy.trust`; a benchmark's grid stated by its description); M4 (W33 P4, § 5q.6: the design-frame exports, the transport validation subject); W44 (a stage's required files, checked — or a ruling that prep's gather replaces it); W45 (checkpointing's owed half); W48 (facts the catalogue and the config classes both state); N10's two calculation-root checks; X4 ② ③; W41; W57-M, W57-K, W57-O; A1.14; V1.9's remainder (the broad hash, the join check, derived structures — its ruling the user's); V1.11 |
| **3** | **Tidy the code and the pages** | no change to what molbuilder does; the same thing done one way, in one place | Q19's later phases — small UI P3–P4; the CSS phases, M8's W1–W6, W13 and W15 (sealing MolView) folded in as one CSS programme; Python phases 2–5 with S1 (§ 0a's Q19 row lists them); § 5r's remainder (A1.6, A1.8–A1.10, A1.15, A1.17–A1.20), then A1.21 (the audit of the rest of the tree); § 8 rows 10 and 11; F9 (a group prep's notes once, not per member); A1.5's remainder (one home for the structure-path rule, renaming a structure vs a file) and A1.13's; V1.14; V1.42 (the relaxation record's held set stated once); the tests — PYF-T, L2, T-ENV, R11, W57-T, DOC-TOC (retire); Q2f (absolute imports — agreed 2026-10-03, done as its own focused session with agent reviews, as the user asked) |
| **4** | **Prove the whole road** | each kind of calculation driven start to finish through the browser and the printed commands, on the code as it then is | M3 P6 = W50 step 3 (an optimization, a vibration and a transport calculation: the parameter tab, the hand-over, Task setup, `prep` / `launch`, the Results tab — every step through the designed doors); M2n's leftovers (the sweep for tests that read text; W57-G, rules held by an assertion on a real run; W56-H, a PySCF relaxation with a held atom; K11 judged); F11's walk (the user's go); the full batch after it |
| **5** | **The documents** | the contracts brought to what is true, and made findable | § 7's four indexes (steps 2–4); § 8 row 13; M4's document sweep (§ 5q.5's list); § 5t P5 (the Makov-Payne script); § 5w.2's text sweep (after K9); E14 (two references checked into `references.bib`); TPL, SMALL and CARRY, each re-derived when taken |
| **6** | **New science** | new things a person can do — each brought to the user as a design, with its science (`scientific-review`: the literature, the numbers, the cost), before any code | M10, from a vibration mode to the current (W42; V1.24, V1.26 — its decision owed, V1.27, V1.34) with V1.25, the mode-displaced frame set (its amplitude the user's); the vibration's other open rows (V1.12, V1.15, V1.17, V1.33, V1.37, V1.38); § 5e, a person's own engine text (W47); S13, transport's convergence sweep; K22, per-atom starting spins (the user's design word); § 9.2, ATOM as an optional pseudopotential check |

**Decisions that wait on the user** *(2026-10-14)*: V1.43 — check a cited pair's
electrodes against its relaxation record's held set (recommended: yes); V1.9's
remainder — whether a broad-hash difference blocks (two rulings disagree);
F11 — the walk's go (~45 min of this machine). The questions this list
carried before were answered on 2026-10-13 from the contracts and the user's
own rulings (W40, F-MDNC, PL-R in § 2; EL-W and E7 in
[`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)).
Step 6's designs — V1.25's amplitude, V1.26, K22 — are brought with their
science when that step starts. **Only Sol can show** (the user's runs, when
the user is there; none on this workstation, *the user, 2026-10-10*): E13 (the
junction on Sol), the Sol memory measurement (M2n), a group's header under a
real scheduler (Q13), the AuBDTAu workflow; and, before a vibration runs there
on 2.10.0, its job environments repaired for `netcdf4` (V1.39) — the user's.

**How a step closes** (each is a milestone):
1. its items are built — the contract first wherever an item changes a rule,
   then the code, then a test that drives the road (`jobset
   init/prep/launch/summarize`, or the page on a live server) and is broken on
   purpose once to watch it fail;
2. the targeted tests pass (the unit's own and what it touched), and the
   package is `pyflakes`-clean (`conventions.md` § 1);
3. **an agent reads the FULL code text** of the milestone's diff and of the
   contract sections it touches — not a grep — and reports each finding with
   its evidence;
4. every finding is re-read against the code before it is acted on, then fixed
   or answered;
5. this section records the commits, the review and its outcome — and only
   then does the next step start.

A full test batch (`tools/testrun.py`) runs at the end of each step, with
nothing changing under it; the last non-e2e batch ran 2026-10-14 on V1.9 (7,212 passed, 2 skipped, 6 expected failures); the e2e batch last ran whole 2026-10-13 (the non-e2e suite then 7,218
tests, and the e2e batch, 302, all passing).

**The Q19 row** — the three passes, its phase 1 done and its later phases
listed — stays here as step 3's detail:

| # | item | state | next |
|---|---|---|---|
| **Q19** | **Three passes — CSS, small UI, Python** *(proposed 2026-10-13 from three read-only surveys; the user: "go with your recommendations … think at framework level or larger scope where the snippet sits, need a systematic point of view for fixing"; for Python: "make sure final outcome to be checked against design contract, and reviewed in their full code/text and validated correctly")* | **phase 1 done 2026-10-13**, each part reviewed by fresh agents on the full code and contract text: 2.9.2 (small UI P1–P2 — one page fence, the shape shown fixed, the chip's advice gone, the five generators on one chain; CSS P1 — one dialog component, no browser dialog left; Python P1 — the package pyflakes-clean), 2.9.3 (the server's one numerical thread), 2.9.4 (Task setup as a checklist and the save bar, the user's request). Its record: [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md) | **open — step 3, in this order** (the user approved the passes' recommendations, 2026-10-13) — **small UI**: P3 prep's answer served as lines (the wording to keep is the user's call), P4 one element builder, one path helper, one download helper, and keyboard access (the sidebar rows, glyph buttons named); Structure optimization's *discard unsaved changes?* on Load (a design call: the viewer is read-only, `runtime.md` puts the question on a dirty canvas). **CSS**, decisions 1–8 as recommended: one button vocabulary, a flat `.card` variant for Results' restatements, `.num-table` reused, tokens for durations, z-index layers, weights and line heights, the sealed modules' palettes (`vibrationview`, `spectrumchart`, `docs-render`), the stylesheet order, `.lock-reason`'s raw values, the junction card's control under its prose; MolView's export dialog, a third modal outside `lib/dialog.js`; three animations that ignore reduced motion; the layout test skipping Transport, This machine and JupyterNB. **Python** phases 2–5: the files a calculation holds read once (`JobSet.of`, the template text, `Shape.named(task.shape)`), `done` returning its `RunStatus`, one instruments reader, the long functions (`render_run_wrapper`, the engines' `spec_for`), `RunRef` for `written_at` and the continuation's fields, S1's wrapper hook per engine, prep's planning — one path for a single run and a swept stage (named when the groups closed; measured before acting), and three small dead guards with the cold sweep's docstring (`next_step`'s `s is None`, the `bench=` that is always false, `_with_notify`'s `None` tests — round B of 2026-10-12 did not re-read them); phase 5 the layer index's missing rows and four upward imports |

---

## 2. OPEN — the one list

**Every open item, in one table** *(consolidated 2026-09-10 at the user's instruction: "consolidate plan and to do, archive finished things and untrue things, clean it up so we have one list"; re-consolidated 2026-10-09)*. A design section (§ 5e–§ 9) keeps the *why* and the order of its own work and sends its rows here; the order (§ 0a) says when.

**A row is evidence of when it was written** — re-derive it against the code before acting (the rule, `archive/2026-10-08-plan-consolidation.md` § 5a). Each row below was re-read at its cited place on 2026-10-09; *Re-checked* says what was found. The long histories of the rows that had grown to tens of lines (W36, W38, W41, W30, W32, X4, W52, W54) are in [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md), verbatim; the row here is what is left.

**Rows that were done, withdrawn or measured untrue are not here** — [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md), [`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md) [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md) [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md) and [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md), each with what it turned out to be. Archived ids still cited elsewhere resolve there: A1.12, A1.16, A1.2, A1.3, A1.7, B2, B3, B4, D2, D4, DOC-TEX, E1, E10, E11, N5, N9, R5, S-T2, S18, S3, TD10 text, TR1–TR3, TR4–TR6, TR7–TR8, TR9–TR10, TD6, TD12, R2-15, SCF-RUNG, CHIP, V1.31, W10, W20, W27, W37, W39, W46, W51, W52, W53, W54, X1.

| # | area | item | from | state |
|---|---|---|---|---|
| **E13** | engine / science | **The first real junction walk — BDT–Au on Sol. DROPPED BY THE CONSOLIDATION.** Named in the roadmap and again in `transport-design` § 7's *"order of proof"* as the run that follows P6. plan.md has the machine-blocked infrastructure (E5 — archived 2026-09-10 — and E7) and the browser and deck walks (W12 — archived 2026-09-10 — and E11) but no row for the transport composite's first real science run | roadmap · `transport-design` § 7 | **the Sol half only** — after § 5u step 4; the workstation half ran 2026-09-29 (the Au–BDT–Au ladder to T(E) on `claude-au-bdt-au`, TD8's acceptance ladder) and is archived *(2026-10-08)* · *Re-checked 2026-10-09:* open — the Sol half only; the workstation half closed by TD8 and Q15's run E. |
| **E14** | science / references | **Reed 2006 and Stokbro 2003 into `science/references.bib`.** `engines/transport.md` § 8 cites both (and Solomon 2008 beside them); none is in the `.bib`, and the rule is that an entry is added when a passage cites it. The 2026-10-08 archive reason — *no live document cites them* — was untrue | roadmap § 7.3 | **reopened 2026-10-09** — each entry checked against the paper before it goes in (§ 7.3) |
| **V1** | engine / science / web | **The vibration calculation — one path on two engines.** The contract is [`engines/vibration.md`](?doc=engines/vibration.md) (§ 10 says what stands); the science is [`science/normal-modes.md`](?doc=science/normal-modes.md); the design and the first audit are archived (`archive/2026-09-24-*`). Built 2026-09-21 → 24: one harmonic path with the rank rule and its gate against PySCF, one mass convention, stationarity on the free atoms, the runs write the pair, the Hessian over the free atoms with the two measured corrections, the SIESTA arm (deck on a sorted copy, `atom-permutation.json` with its key, the `.FC` reader, `summarize run`, the warm-file section, the kind's start state), the equilibrium block optional, the wrapper's banner. **Open, the sub-rows:** | `engines/vibration.md` § 10 | open · *Re-checked 2026-10-09:* a container: V1.11, V1.15 and V1.38 narrowed, V1.25 unblocked, V1.31 archived. |
| ↳ | **V1.9** | **The structure identity rule — what is left** (the rule settled in discussion 2026-09-22; its first half, the pin, built 2026-10-14 — 2.9.6, M2m: the geometry hash over the atom lines as written and the cell and axis kinds, never the comment line, in every pair's sidecar, compared by the one read door, prep refusing a changed structure and the browser's load door saying so; the row as it stood is in [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md)). **Open:** a *broad hash* over the labels, regions and identity columns, for provenance; the check when two runs are joined (the geometry hash only); and **derived structures** — a reordered copy minting its own identity and recording its parent's with the permutation (`atom-permutation.json` records the permutation and no parent today). Two rulings disagree and are the user's to settle when this is taken: V1.9 says a broad-hash difference is reported, never blocking; W39's customized parameters, as agreed 2026-09-27, enter the hash and prep refuses when it moves ([`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md), M2m, W39). `spectra.json`'s own `structure_hash` is V1.39's (decided 2026-10-14: retired, the result states its structure) | `model/structure-molstruct.md` § 3 · `engines/vibration.md` § 6.10 | open — the pin done 2.9.6 |
| ↳ | **V1.10** | **The pair writer renders both halves** (ruled 2026-09-23, the unification audit § 1.1a): `pair()` returns text, so a sidecar that cannot be written (a NaN in `info`) stops the write before the `.xyz` is on disk. **The deck's half is done (2026-10-05):** the PySCF deck writes its pairs with the codec itself, imported from `mb_pyscf.pyz`, and its own serialiser — the third — is deleted | unification audit § 1.1a | open (the codec's half) · *Re-checked 2026-10-09:* open — `pair()` returns the sidecar as a dict (`workingcopy_structure.py:298-308`) and `_write_pair` writes the `.xyz` first (`:420-446`); `structure.md` § 2.4 marks both-or-neither not built. |
| ↳ | **V1.11** | **Untested and open**: the reduced Hessian with a GPU mean field; a composed permutation for a structure sorted for two reasons (ruled one record, no caller yet); `transport/compose.py` writing its record through `write_permutation` and stamping its key (U10; reading through `read_permutation` done 2026-09-28) | `vibration.md` § 4.4, § 5.2 | → **§ 5u step 3** *(2026-09-29: compose writes the permutation by hand and calls the sort directly, so no key is stamped; `sort_by` / `write_permutation` exist for it)* — **the compose half built 2026-10-02 (M5 step 3)**: it sorts with `sort_by(…, "transport")` and writes through `write_permutation`, the key stamped (`test_transport_prep.py` reads it back) · *Re-checked 2026-10-09:* the compose half done (`transport/compose.py:757` `sort_by`, `:831` `write_permutation`); the test the row names (`test_transport_prep.py`) went in `93977f5d`, so no test reads the transport permutation key back. Untested: the reduced Hessian with a GPU mean field; one record for a structure sorted for two reasons. |
| ↳ | **V1.12** | **Not done**: a mode-by-mode intensity cross-check against an external code (Gaussian / ORCA / Turbomole); absolute intensities carry the caveat until then | `vibration.md` § 9 | open · *Re-checked 2026-10-09:* open (`vibration.md` § 10). |
| ↳ | **V1.14** | **From the UI walk of 2026-09-23, not this kind's**: the Molbuilder tab's save prompt doubles a typed suffix (`x.xyz.xyz`); the `#`-labelled provenance regions (`O#`) warned as unconsumed on every SMILES-built molecule — needs a ruling; the vacuum notice advising 8 Å on a gas-phase PySCF run | unification audit § 1.18 U3, U4, U7 | **the suffix done 2026-10-13** — the door adds the extension to a name typed without it and takes one typed with it, any case (`molview-doors._askProjectPath`, both the structure and the picture; `projects.md` § 3; `test_molview_files_door_js.py`, mutation red); *open:* the vacuum advice · *Re-checked 2026-10-09:* the doubled suffix stood — `lib/projects/molview-doors.js:114-117` appends `.xyz` to a typed `x.xyz`, the save route writes the path as given; the 8 Å vacuum advice is engine-blind (`cell.py:398-405`); the `#`-label warning is settled by *labels are the user's* — nothing to build. |
| ↳ | **V1.17** | **Four held systems through the whole road**, today rank rows only: acetylene with both carbons held (the collinear trap end to end), NH₃ with its three hydrogens held (nothing removed), an empty held list reproducing the free path *exactly* (what makes "the free case is the held case with nothing held" a fact), the water dimer with one molecule held (the over-removal guard) | `vibration.md` § 9 | open · *Re-checked 2026-10-09:* open (`vibration.md` § 10). |
| ↳ | **V1.24** | **Mode matching across runs** (needs a design): the overlap of eigenvectors in the shared free subspace, mass-weighted — the active-region convergence test (Models A/B/C) and the PySCF-against-SIESTA comparison of one molecule are the same calculation | `science/normal-modes.md` § 4b.6 F, § 4b.3 | open · *Re-checked 2026-10-09:* open — M10. |
| ↳ | **V1.25** | **The mode-displaced frame set** *(decided 2026-09-24, user)*: a generator that writes ONE multi-frame pair — the base as frame 0, then `R_A(Q) = R_A⁰ + Q·L_canonical` with `R_F` unmoved at the zero-point amplitude and its thermal growth — from a spectra file, for a named mode; the rule (run, mode, amplitudes) recorded in the pair's `info`. It is the interface to transport's frame axis (W32); a person's own script writes the same pair. Level one of `vibration.md` § 5.6. *W42 (draft 2026-09-28) proposes one set for many modes, at the Gauss–Hermite nodes of the thermal distribution (D6, D7)* | `engines/transport.md` § 2a.9 · `model/structure-molstruct.md` § 6.1 · `science/normal-modes.md` § 4b.6 G | **contract written 2026-09-24; re-ruled 2026-09-29 (TD10)**: the generator is a separate backend procedure, designed on its own — the equilibrium relaxed structure and a normal mode in, one multi-frame structure out — and **each frame's parameter set goes in the structure's `customized` section (W39), not `info`**; transport is its consumer, never its builder. Open, after W39 · *Re-checked 2026-10-09:* unblocked — W39 built in `c93da223`. The contract is `vibration.md` § 5.10 ③: the generator, its own module, writes one frame set for **one** mode (user, 2026-10-09) — `with_frames`, the mode on the structure's rows, each frame's node and `weight` through `set_customized(…, frame=i)`, frame 0's weight included, the set's summing to 1. **Its amplitude, to design** (the user, 2026-10-09: *"how much should the max displacement of a selected vibrational mode use? ... temperature involved? how about local EF enhanced (such as SERS in Raman etc) mode? what's the good starting point"*) — proposed 2026-10-09: the spread is the mode's own, `σ² = (ħ/2ω)(2n+1)` — thermal `n` from the experiment's temperature (zero-point above ~300 cm⁻¹ at 300 K, temperature-driven for contact modes), a stated effective `n` / `T_eff` for a pumped mode (SERS: incoherent, still Gaussian, wider), the max displacement then an output; a coherently driven mode is the one case where the max displacement `A` is the input — the arcsine distribution, Gauss–Chebyshev frames at `0, ±0.866 A`, weights `⅓` each; always a warning past a stated fraction of the nearest-neighbour distance. Its consumer, Q17, is built (a person's script writes the same set today, as the real-junction e2e's does). |
| ↳ | **V1.26** | **`Δρ_ν(r)` maps** (decided with M10's design, step 6): SIESTA's density grid at `±Q_ν`, differenced — the discussion's intermediate quantity before any oscillator strength on a metal | `science/normal-modes.md` § 4b.7 | open — M10; its decision is owed · *Re-checked 2026-10-09:* open — M10; its decision owed. |
| ↳ | **V1.27** | **The PySCF probe** (decided — W42's D8): five points per mode instead of two; the coupling per zero-point amplitude `g_ν = ∂ε/∂Q · √(ħ/2ω)` in meV written beside `ΔE/(2A)`; a molecule-projected frontier quantity for a cluster whose HOMO and LUMO are metal states. *Decided with W42 (D8, 2026-09-28): the probe runs on the thermal nodes* | `vibration.md` § 4.8; `science/normal-modes.md` § 4b.5 F–G | open — M10 · *Re-checked 2026-10-09:* decided (D8), not built — M10. |
| ↳ | **V1.33** | **The Task setup tab re-derives the vibration ladder in JavaScript** (`proposedFromHandover`, `task-setup/viewer.js:2775`, spells `relax`/`freq` and the box rule by hand; `_afterRunLines` is gone) while Python owns it (`pyscf/stages.py:120`, `vibration_stages`). A second copy of one rule; the hand-over or the folder answer should carry the proposed ladder computed server-side, and the page should ask the description which rung is the force-constant one. Found by the 2026-09-24 review | `vibration.md` § 2.2, § 5.2a | open *(re-cited 2026-10-08)* · *Re-checked 2026-10-09:* the JavaScript copy is now `task-setup/viewer.js:2807-2838`; Python's `pyscf/stages.py:120`. |
| ↳ | **V1.34** | **The response screen — which mode to take to transport** (decided with W42, 2026-09-28): the vibration kind's half of W42 — each mode's character at load, the charge each mode moves from the force-constant run, the frame set at thermal nodes (V1.25), the PySCF probe on them (V1.27) | `vibration.md` § 5.10 · `science/normal-modes.md` § 4c | open — M10 (W42 decided) · *Re-checked 2026-10-09:* open — M10. |
| ↳ | **V1.37** | **A vibration's result is written readable by its owner alone.** Found 2026-09-29 on the Au–BDT–Au spectrum: `aubdtauvib.spectra.json` is `0600` while every other file of the run is `0664`. `sidecars/spectra.dump_spectra_json` writes through its own `tempfile.mkstemp` + `os.replace` and never widens the mode (`sidecars/spectra.py:163-180`); `persist.write_bytes` -- the one writer -- does (`persist.py:68-88`), and says why (*"mkstemp creates 0600, which is not what a shared artifact should end"* up as). The finish runs it job-side from the bundled copy (`spectra_sidecar`), which may be why it keeps a writer of its own: whether `persist` can travel in the bundle decides the fix -- one writer, or the same widening in the bundled one | `persist.py` · `sidecars/spectra.py` | open · *Re-checked 2026-10-09:* the writer is `write_spectra_payload` (`sidecars/spectra.py:98`, its mkstemp `:159-180`); `persist.py:88-129` is the writer that widens the mode — K13's. |
| ↳ | **V1.38** | **A δ-convergence comparison that also varies the mesh.** `engines/vibration.md` § 9 still names it as owed; V1.23 closed as built with the δ axis only (the displacement sweep), so the mesh half — the one that separates the grid's error from δ's (§ 9's H₂ measurement) — has no row. Found by the 2026-09-29 coverage check | `vibration.md` § 9 · `science/normal-modes.md` § 4b.6 C | open · *Re-checked 2026-10-09:* narrower: the sweep takes a mesh stage already (`spectra/displacement_sweep.py:8,17,152`; `vibration.md` § 5.9); what is left is the measured δ+mesh comparison on H₂. |
| ↳ | **V1.39** | **A vibration's result states its structure whole, in a JSON and a NetCDF** (decided with the user 2026-10-14; revised the same day after two review agents; the design is `engines/vibration.md` § 6.10). Schema 8: `structure` is `Structure.to_dict()` of the equilibrium structure, the input's run records left behind; built on PySCF from the deck's `_MB_SIDECAR` through `molstruct.apply_to_structure`, on SIESTA from the deck's own ATOM-METADATA and ENGINE-OFFSET blocks put back in the input order (the held set checked against the FC range), the `vibration` block adding only the title and identity columns; `structure_hash`, `equilibrium.elements/positions_ang` and `frozen_atom_idxs` retired into it; `equilibrium.masses_amu` kept (on the `mass_amu` channel it would make the structure read as half a frame set); `relaxation.record` carrying the `relax` stage's record whole; `n_atoms_total`, `free_atom_idxs`, `n_atoms_in_hessian`, `hessian_scope` stated and checked; the reader serving one shape, then the consumers migrated (`lib/spectra/core.js`, `blueprints/spectra.py`, the sweep, the frame-set generator), then the writers; the SIESTA sweep comparing stated structures within 10⁻⁶ Å; a frame set citing the result alone, its `result_sha256` compared at the transport citation; `<label>.spectra.nc`, NetCDF-4 in three groups (`/structure` with `to_dict()` whole as JSON text, `/modes`, `/thermo/grid`), the scalars as attributes, through the variable record and loop moved out of `transport/datafile.py`, read back equal to the JSON. **What it changes beside**: `structure.py`, `molstruct.py` and `persist.py` join `mb_vibration.pyz`, `persist.py` `mb_pyscf.pyz`; both files written through `persist` (V1.37 and K13's spectra item close with it); `netcdf4` in the PySCF recipe and listed by both SIESTA recipes, every self-test importing it, the SIESTA `loads` check importing it before the force-constant run; I15 retires, I23, § 5.5 and `configuration.md`'s bundle rule gain `netCDF4`; `.spectra.nc` registered (`runfiles.WRITTEN`, § 6.9, `job-contracts.md` § 6.1). **Done when** both engines' e2e vibrations write both files, the `.nc` reads back equal, a v7 result still loads, and the viewer draws from the v8 shape | `engines/vibration.md` § 6.10 | decided, not built — **2.10.0**, after 2.9.6 is pushed |
| ↳ | **V1.40** | **A continuing PySCF stage's starting geometry is checked at launch** (decided with the user 2026-10-14). The script swaps in the previous run's `<label>_optimized.xyz` coordinates and takes everything else from its own deck (`pyscf/input.py`, `continues`), which is right only while the file lists the deck's atoms in its order. Before the job is sent, launch reads the previous run's pair through the codec — the pin, the atom count, the elements in order against the stage's deck — and refuses naming the way out; the job keeps its plain read. **The carry copies a structure file as a pair**, its sidecar named by `molstruct.sidecar_path_for` (the one pairing rule, `structure-molstruct.md` § 6), never a second row in `pyscf/warm-files.toml`; `engines/pyscf.md`'s *copied at `prep`* corrected to launch (`job-system.md`). Whether SIESTA's carried `.XV` needs the same check is asked when this is built. **Silent failure it closes**: a run's geometry file replaced by hand, its atoms reordered, held and labelled as other atoms. **Done when** a launch refuses a stage whose previous pair's pin or atom order is wrong (a `launch_protocol.toml` row) and the carried pair opens through the codec in the new run | `engines/pyscf.md` § 3 · `model/structure-molstruct.md` § 3, § 6 | decided, not built |
| ↳ | **T-CO2** | **The relaxation-to-vibration road on CO2** (agreed with the user 2026-10-14: *"why don't we switch to CO2 molecule so it can cover all the needs?"*). The shared relaxed-for-vibration run (`tests/support/real_runs.relaxed_for_vibration`, the fixture renamed from `real_h2_relaxed_for_vibration`) becomes CO2, relaxed with the carbon held and both oxygens free, and every test that reads it follows: the relaxation-record tests (`tests/test_siesta_relax_run_e2e.py`, the held-set rows of `engines/vibration.md` § 2.3a moving here from the real-junction test), the monitor (`test_monitor_watches_a_live_run_e2e.py`, its `_STEM`) and the inspector (`test_inspector_pageshow_refresh_e2e.py`). The SIESTA vibration e2e (`test_siesta_vibration_e2e.py`, its own four tests: the ladder, the ticked stage alone, a two-stage displacement sweep, the flat refusal) moves to CO2 too. CO2 is linear (five rigid motions) with infrared-active and -inactive modes, as the PySCF vibration e2e already has it. H2 stays for the flat and layered road and the run index; the hydrogen chain stays for transport. The pseudopotentials are checked in (`tests/fixtures/psml`, C and O beside H, one PseudoDojo release). **Cost**, estimated, one rank: the shared relaxation one to two minutes; the vibration e2e's relaxation and four force-constant runs of 12 displaced SCFs each (the carbon held). **Done when** the moved tests pass on CO2 and the held-set rows hold a superset with an atom still free | `tests/fixtures/psml/README.md` · `engines/vibration.md` § 2.3a | open — step 1, before V1.39 |
| ↳ | **V1.41** | **A force-constant stage reads `relax`'s coordinates in the engine's sorted order** — the mapping contract broken: `_vibration_stage_geometry` (`jobset/prep.py`) pairs them row by row with this prep's sorted copy, checking only the element sequence, right only while both preps held the same atoms. **Folded into RMP** (§ 5ab, rule 4): the stage takes `relax`'s restored result, or warns and restores through `relax`'s index | `model/overview.md` § 2.2 · § 5ab | open — step 1, with RMP |
| ↳ | **V1.42** | **The relaxation record states its held set twice** — `held_atom_idxs` beside `held_atom_keys` (`parse/contract.py`), the first read by tests alone, while `vibration.md` § 2.3a says the record names atoms never by an index. Cut it, and `model/parse.md` § 5b.1's row with it | `model/parse.md` § 5b.1 | open — step 3 |
| ↳ | **V1.43** | **A cited pair's electrodes are checked against its label, not against what its relaxation held.** `wizard._refuse_unless_frozen_bulk` reads `frozen_atoms`; a cited run's electrodes are also measured unmoved against where its relaxation started, a cited pair's are not (`transport/compose.py`, `_pair_parts` gives no starting geometry). So a pair whose relaxation left an electrode free and whose label now holds it passes, though the leads' promise — the bulk they stand for — is about the relaxation. Proposed: when a cited pair carries a relaxation record, its electrode atoms must be among the record's `held_atom_keys` — the structure checked against its own record. **The user's word owed** (recommended: yes) | `engines/transport.md` § 1.1, Step 0 | open — step 1 |
| **RMP** | structure / parse / web / execution | **The remapped state — the mapping contract stated at every output** (decided with the user 2026-10-14; its revision plan approved the same day; the design record is § 5ab). Today a reordered run's output — a SIESTA vibration's stages, every transport rung — is read in the engine's order, unmarked, everywhere but the vibration finish and the transport composer: the Results tab, a structure exported from it, `xv2xyz --from-run`, the force-constant prep (V1.41). The rule: stated remapped, never silent; never without its index (sidecar v14); a run's restored result written when it finishes; a building door restoring with a warning; the Molbuilder tab warning and offering restore | § 5ab · `model/overview.md` § 2.2 | approved — a validating review, then R0 (the contracts) |
| **W33** | structure / engines / web | **THE ENGINE OFFSET — one placement rule for every engine** *(user, 2026-09-25: "always adjust it before sending to siesta or other engines that the coordinates of all atoms are centered inside the cell"; "we don't have to have special logic to treat isolated, periodic, transport axis_info differently")*. Every engine receives the design coordinates + `engine_offset` with the cell at the origin; the offset is computed from the cell and every atom (the fractional span centred, never re-wrapped), recorded in every deck, and stated 0 for an engine's own output, so the Results tab draws what the engine had. Retires `cell_origin` (an origin the person assigns is kept, stored as the offset), the derive-on-null corner and its per-axis rules, calibrate, and the two hand translations. Found by the fake-junction ladder: a TranSIESTA device refused on a flush corner, a Results box the engine never had, the molwatch step-0 jump (the 2026-09-25 audit's finding X4 — not § 2's row X4) | `model/structure-periodicity.md` § 6.0 · § 5q | **P0–P3 done** (P3's T3 on 2026-09-27, `tests/test_results_export_e2e.py`; its dev-server check on 2026-09-25)**; P4 open** — P5 closed by TD8, the Au–BDT–Au ladder run to T(E) on 2026-09-29 (§ 5q.6 is the per-phase record). **D1–D16 settled** (§ 5q.8) · *Re-checked 2026-10-09:* P4's three items (§ 5q.6) — M4. |
| **W34** | science / engines / web | **THE ELECTRONIC STATE — charge and spin as one answer per calculation** *(user, 2026-09-25: "put spin and charge setup in to a unified framework so that these information can be produced consistently and systematically for different engines"; "investigate holistically ... from template to validation ... a design gap/framework level investigation rather than a patch"; "documentation should have an explicit discussion on species with these properties")*. Four engine-neutral template items — `net_charge`, `spin_treatment`, `unpaired_electrons`, `method` — resolved once with what the structure adds (the charge's source, the electron count, finite or repeating) and read by every deck writer, check, hand-over, form and read-back; restricted-open explicit; a species-by-species chapter. Found by the transport ladder's rung reports (a gold lead and a formate ion both told "switch to open-shell") | `science/chemistry-correctness.md` §§ 2a–2b · § 5s | **P1–P4 built** (M6, 2026-09-28/29; decisions 1–9, § 5s.2); **P5 open** · *Re-checked 2026-10-09:* P5 open — the TBtrans two-channel half built (`transport/record.py:157-165`, through `tbtrans.transmission_files`); no reader of SIESTA's charge and moment, none of PySCF's ⟨S²⟩. |
| **W35** | parse / execution / web | **THE RUN RECORD — what ran, with what, and how it went, for every run; one SIESTA-family reader; the transport report** *(user, 2026-09-26: "include in the report of transport results ... all indicators that can provide scientific information, progress tracking and symptom of non-convergence"; "parameters used for calculating transport should be similarly reported ... check what we did for the other tasks, and find the best way to honestly and fully record what is the computation setup and scientific setup, and how result evolves in the record, and output report"; "fdf parser need unified upgrade"; "get the framework and code unified and finalized")*. One record per attempt, composed on read — computation, setup (default · asked · engine used), the deck, every iteration of every phase from one grammar, the verdict with its symptoms — read by a Run panel for every kind and by a transport report built from its rungs; the NEGF contour stated rather than left to TranSIESTA's fallback. Found by a device that diverged for eleven hours with every reader saying otherwise | `model/parse.md` § 5d · `engines/transport.md` § 2a.12 · § 5t | **P0–P1 done (`a4c1d28e`); decision 10 done — the monitor, the wrapper's endings and the report fields through the framework; P2: the record and the Run panel built (2026-09-27), the rest below** · *Re-checked 2026-10-09:* M3 — P2's remainder, P4, P5, P6 (§ 5t.3). |
| **W36** | execution / packaging | **The run-independence review (2026-09-27), agreed item by item** *(user: "i need to go through all of them so we have agreement before i let you just go ahead to fix all")* — its full text, ① to ⑪ with each agreement, is in the 2026-10-09 archive. ①–⑥, ⑧, ⑨ and ⑩ are built. **⑦, open: the job inherits the whole shell** — `submit.py:1232` passes `{**os.environ}` and `:307` `--export ALL`, and the wrapper honours `OMP_NUM_THREADS` (`runwrap.py:2224-2236`); the agreed fix is *keep inheriting, guarded*: launch hands `-np` / `-omp`, which win (`submit._run_sh_args`), so an inherited count cannot resize a launched run — what is unbuilt is the by-hand path and the log naming an override. `running-a-job.md` § 2.0a states this; its § 3.2–3.3 precedence still puts `OMP_NUM_THREADS` second, against ⑦'s *"a bare OMP_NUM_THREADS no longer sets it"* — one table, written when ⑦ is built. **⑪, done 2026-10-09**: the `$MOLBUILDER_ROOT` text is gone from `job-contracts.md` § 2.5, `repo_root`'s docstring names its real callers | `execution/running-a-job.md` § 2.0a, § 3.3 | ⑦ open — M2k |
| **W38** | execution / front end | **The multi-door review (2026-09-27): one fact, two or more code paths that can answer it differently** *(user: "we are looking for framework level unification and consistency")* — F1–F9 and M1–M5, its full text in the 2026-10-09 archive. Built: F2 F3 (M2f), F4 F5 (M2i), F7, F9, M1–M5, F1 F6's one placement and one record (unit 11a), the bias chain's copy list from the one door (`submit.py:2014-2022`, the warm list's `along` row), F8 (M2j — the Results door reads `jobset_status`, `results.py:198-211`; `every_run`, `runstatus.py:240-275`). **Open: M2k's GPU claim / match half** | `execution/gpu.md` § 1 | the GPU half open — M2k; M2j closed at its review |
| **W41** | engines / results | **A SIESTA vibration's finish, the remainder** *(user, 2026-09-28: "siesta only gave some optimization ...")* — the finish built (M2b′, M2b″: the job derives its spectrum, `mb_vibration.pyz`); the measurement and the rulings are in the 2026-10-09 archive. **Left:** the force-constant `.out` titled *"SIESTA optimization"* — the title guessed from the suffix (`lib/inspectors/trajectory.js:108-115`), against `web/results.md` § 0.3; I7's sorted positions; the ⚠ beside an in-tolerance stationarity sentence; Makov–Payne run by hand; the finish parsing the whole output where FC step 0 suffices; PySCF leaving `ir_fd_step_ang` out of its record — written `null` under the explicitness rule. Re-running a failed finish is settled by rule (no recovery machinery; launch again) | `engines/vibration.md` § 5.5, `web/results.md` § 0.3 | open |
| **W42** | engines / science | **FROM A MODE TO THE CURRENT — which vibration to take to transport** *(user, 2026-09-28: "think about this and how the spectrum calculation can be enriched by the additional electronic state/PDOS calculation, and propose an improvement in the vibration calculation contract/document/design and see how we can move forward ... this is after all the other issues are done, so this is just a draft for next step, do not implement")*. The science, `science/normal-modes.md` § 4c (proposed): at a metal the molecule's orbitals are resonances and the junction has a Fermi level, so the current sees `T(E)` near `E_F`, not the HOMO–LUMO gap; a mode acts on a level `ε_r` or on the coupling `Γ` (the contact modes, whose exponential coupling gives a large positive curvature); a DC measurement, thermal or light-driven, sees the second derivative, `⟨ΔG⟩_ν/G = ½ T″σ²/T₀`, and the two scores — the level shift at `σ`, the conductance change — are kept side by side; three frames at the Gauss–Hermite nodes of the thermal distribution give the curvature and the second-order average at once; one transmission at the base screens every mode by the rigid-shift estimate `T″ ≈ g_ε² ∂²T₀/∂E²`; the free atoms' projected DOS and their own states (the extended molecule's MPSH) stand in for HOMO and LUMO. **The design, proposed:** `engines/vibration.md` § 5.10 — ① each mode's character (element shares, the free part's whole-body share, the bond stretches at `σ`), derived at load, no calculation and no new label; ② the charge each mode moves between the held anchor and the free part, from the force-constant run's own displaced SCFs; ③ V1.25's frame set for many modes at the Gauss–Hermite nodes — *made by V1.25's separate procedure, each frame's parameters in the structure's `customized` section (TD10, 2026-09-29)*; ④ an `electronic` kind — one SCF per frame, each job's finish writing `E_F`, the charges, the projected DOS and, with sisl, the levels; ⑤ `summarize` over a frame set, electronic or transport, writing `<label>.mode-response.json` — slope, curvature and thermal average per mode, levels followed by overlap, the rigid-shift estimate beside the explicit value; ⑥ the ranking table in that record's presenter; the PySCF probe on the same nodes (V1.27). Folds in V1.24–V1.27. **Decided with the draft** *(user, 2026-09-28: "the anchor is the fixed atoms. we don't specify the anchor, the atoms is labeled already with fixing or unfixed")*: D1 — the anchor is the held atoms and every projection is over the free atoms, per element, with no new label; D2 — then not needed, the result already carries the free and held atoms. **Decided 2026-09-28, the user's answers:** D3 — yes, the charge each mode moves, *"and result presentation should include the correct data and UI design to present it"*; D4 — yes, an `electronic` kind, *"and make it clear the difference between pySCF and siesta, from setup and calculation and presentation"*; D5 — sisl in every SIESTA env, *"so it is always available when we need them ... fix the install-env and other env related setup to get this consistently done - do not reinvent wheel as we have a full system of install implemented"* (built with the M2b‴ milestone); D7 — the thermal nodes, as recommended (the displaced structures the electronic and transport runs compute on — not the animation, which the user asked about); D8 — the PySCF probe on the same nodes, as recommended. **Taken as recommended unless the user objects:** D6, one displaced-structure set for many modes. **Registered with it:** E2 — a vibrational-only result records no pressure (built with M2b‴); E3 — a command that re-runs a failed finish on its attempt, the user's yes owed. **Order, once agreed:** P0 the decisions; P1 ① (no engine); P2 ② — measured first, whether SIESTA prints the populations at every FC step; P3 ③ with its displacement statement, checked by the frame set's citation promises (W32 ②); P4 ④ and ⑤ for the `electronic` kind — the frame axis for one kind first; P5 transport on the frame set (W32 ③–⑤, behind W32's gate) and ⑤'s transport half; P6 ⑥. Validation H₂, then the carbon-chain junction, then Au–BDT–Au. **Found while drafting, fixed with M2b‴:** the SIESTA deck's commented population hint spelled the retired `WriteMullikenPop` (`siesta/input.py`); it names 5.4.2's `Charge.Mulliken` and `Charge.Mulliken.Format` | `science/normal-modes.md` § 4c · `engines/vibration.md` § 5.10 · `engines/transport.md` § 2a.9 | **draft 2026-09-28 — decided D1–D5, D7, D8; D6 as recommended; after the current work order** · *Re-checked 2026-10-09:* decided, M10; V1.26's decision owed; the frame set it uses is V1.25's — **D6 (*one set for many modes*) reversed 2026-10-09: one mode a set, each its own transport calculation**. |
| **W44** | execution | **A stage's `required` files, and the check that its run directory holds them — designed in four contracts, built in none.** `execution/job-contracts.md` § 4.4 (`:141`; *"Neither half of this is built … the catalogue carries no `required` item"*), `engines/template.md` § 12.1 row 8, `engines/stages.md` (*"the check itself is unbuilt"*); `worked-example.md`'s gap 13 names the hazard — *a TranSIESTA ladder starting without its `.TSHS`* — and `run-identity.md` states it as if it were checked. Found by the 2026-09-29 coverage check | `job-contracts.md` § 4.4 · `template.md` § 12.1 | open — beside M2g/M2h (the restart-file list, the hand-over), or a ruling that prep's gather refusals replace it · *Re-checked 2026-10-09:* open — `job-contracts.md:141` says neither half is built; `template.md` § 12.1 row 8. |
| **W45** | execution / checkpoint | **Checkpointing's owed half.** The save is always (prep and Task setup's Save call `checkpoint.save_before`, `prep.py:1890`, `:2851-2856`; `checkpointing.md` § 12 A3 ✅). **Left:** a `checkpoint verify` verb (the CLI has init, save, list, tag, restore, config — `cli.py:2885-3132`; `verify_archive` is reached only from restore, `checkpoint.py:839,1695`); invariants S2, S3, S4, S6 and L8 without a test (`checkpointing.md` § 13.4 points here), A3's test and S5's run-layer half with no row before this one; `project-layout.md`'s invariant 16 and § 6.1 say the archive does not reach a bench trial's `.DM`, against S1 (*every regular file goes to git or the archive*), which the code follows — the layout's sentence is corrected to S1 when this is built; § 12's S2/S4/S6/L8 still read *"needs the layout / the description"*, which exist | `execution/checkpointing.md` § 12, § 13.4 | open |
| **W47** | engines / execution | **A person's own deck text: § 5e's engine additions and the USER-CUSTOM zone's carrier.** No `user_custom` item exists, so in the staged path the zone is emitted empty and a person's text does not survive (`template.md`, `job-contracts.md`, `worked-example.md`'s gap 12); § 5e (*a person's own engine text, as an input*) is a design with no row and says the zone's defects are *"listed where they belong"* — they were listed nowhere | § 5e · `template.md` § 12.1 row 1 | open — § 5e not started · *Re-checked 2026-10-09:* open, § 5e not started. `engines/template.md` § 9.2 disagrees with § 5e on four points — the carrier (a catalogue item `user_custom` against a separate input), the zone's role, stage overrides against a per-addition switch, the placement; § 5e is the user's ruling of 2026-09-05 and owns it, its open question 1 leaves the zone's fate to its design. `template.md` § 12.1 row 1. |
| **W48** | engines | **Facts stated twice — the catalogue and the config dataclasses.** `template.md` § 2.1a: 491 facts live in two places, and `kind`, `default` and `expands` are unguarded (a live divergence in `use_gpu`'s `expands`) — *`default` guarded since 2026-09-29 (`test_catalogue_agreement.py:245` `test_the_DEFAULT_agrees`, M5 step 2), zero disagreements measured that day*; `form-schema.md` reads `workflow_group` from the class — the same class of drift | `template.md` § 2.1a · `form-schema.md` | open — its deletion is argued in `template.md` · *Re-checked 2026-10-09:* open. |
| **W50** | front end / engines / execution | **EVERY PARAMETER OF THE STRUCTURE OPTIMIZATION, SPECTRUM AND TRANSPORT WORKFLOWS, CONFIRMED END TO END** *(user, 2026-09-29: the quote in M11; "this is not a poking test, a static review followed with e2e validation")*. Found the day it was asked: the PySCF vibration refused a periodic structure where the engine's own rule notes it (fixed, `762ca298`), and gold on def2-SVP received no core potential while the hint stayed silent on a false belief about PySCF (fixed, `5a8133cc`) — each a parameter whose chain nobody had read end to end since the Spectrum tab gained its second engine (2026-09-24). **The method**, as the structure optimization's full-text review (`archive/2026-08-14-template-execution-review.md`): (1) a static review per track — Structure optimization on SIESTA and on PySCF, Spectrum on PySCF and on SIESTA, Transport's five rungs — every file of the chain read in full (the tab's HTML and JS, the routes, the catalogue rows and `template.py`, config / resolve / prep / validation, the deck writer) and each parameter traced from its control to its line in the generated script and to where the engine's own source reads it, nothing changed and nothing run; (2) every finding re-read against the code and fixed at its owner, the contract first; (3) the road in the browser per track — the tab, the hand-over, Task setup, the printed `prep` / `launch`, a small run, the Results tab | `web/spectra.md` · `engines/vibration.md` · `engines/pyscf.md` · `engines/siesta.md` · `engines/transport.md` · `web/task-setup.md` | **started 2026-09-29** — step (1) done the same day: five reports, every defect verified; step (2) is § 5w — the findings grouped by the mechanism that produced them, each class one declaration in the template and one door (K1–K17), approved 2026-09-29 — K6 first · *Re-checked 2026-10-09:* a pointer — K6 done; the remainder is Q9, § 5w. |
| ↳ | **V1.15** | **Recorded, not in scope**: the transport connection (displace along a mode, then transport; the electron–vibration coupling from `FC.Save.dHS`) and Born-charge infrared on SIESTA — each a feature to design as one | `vibration.md` § 5.6 | not started · *Re-checked 2026-10-09:* narrower: displace-then-transport is V1.25 + Q17; left: the electron–vibration coupling from `FC.Save.dHS`, and Born-charge IR on SIESTA. |
| **W1** | front end | **The document tier (step C).** `html, body`, `header`, `button`, `footer`, `textarea` genuinely differ per page; the `*` reset is already deleted. Blocked on a browser pass over all pages | `css-system` § 4C | partly · *Re-checked 2026-10-09:* open — M8; re-derive at its start. |
| **W2** | front end | **One home per component (step D).** `.card`, `.status`, `header .tagline`. **One value to settle first:** `.card`'s padding is `var(--space-md) 18px 18px` and 18 is off the 4px grid the contract declares — moving it shifts every page by 2px | `css-system` § 4D | not started · *Re-checked 2026-10-09:* `.card` has one home now (`lib/page-shell.css:284-288`, tokens since `a09a8d60`); left: `.status` in two sheets (`spectra/style.css`, `page-shell.css`), `header .tagline` in four. |
| **W3** | front end | **Per-page token/namespace passes (step E)**, one page per commit: `spectra`, `structure-optimization`, `transport`, `results`, `documents` | `css-system` § 4E | partly · *Re-checked 2026-10-09:* open — M8; re-derive at its start. |
| **W4** | front end | **Guards 1 and 2 (step F)** — one home including elements; a page sheet contains only its own tier. Guards 3 and 4 landed. **Both remaining gaps are now provable, 2026-09-07:** guard 1 is absent — ~~*by an explicit skip*, `test_css_no_duplicate_selectors.py:150` reads `if "." not in norm: continue`~~ (that file is gone, 2026-10-08; the CSS guards on disk are `test_css_classes_are_defined.py`, `test_css_module_boundary.py`, `test_css_negated_var.py`); guard 2 has no test at all | `css-system` § 4F | partly · *Re-checked 2026-10-09:* open — M8; the CSS test files it names exist. |
| **W5** | front end | **The inspectors module's appearance still lives in `results/style.css`.** **Re-derived 2026-09-07 and the number was prose:** 70 was `grep -c inspector`, which counts the file's 200-line comment header and a hierarchy diagram. Comments stripped and classified by who EMITS each class: **22 module-owned rule blocks**, and **6 of those are dead** — `.inspector-section`, `-section-header`, `-section-body`, `-section-hint`, `.source-body-error`, `.structure-error` have **zero emitters anywhere** in the repo and are deletable outright, which this row never said. Three sheets are already repatriated. *"Renders unstyled elsewhere"* is **latent, not reachable**: `registry.js` is script-tagged by `results.html` only, so the css-system doc's premise (it also loads on /molbuilder and /spectra) is stale. Also: `inspectors/bench-summary.css` is missing from the boundary guard's `MODULE_SHEETS`, so that guard treats a module sheet as a page sheet | `css-system` § 7.0 | partly · *Re-checked 2026-10-09:* six dead classes still in `results/style.css`; `MODULE_SHEETS` (`tests/test_css_module_boundary.py:59-68`) lacks `bench-summary.css`, `fc-sweep.css` and `transport.css`. |
| **W6** | front end | **The editor module.** The loader half is confirmed and accurate: `lib/codemirror-load.js` is the one loader, two of three surfaces import it, and `lib/inspectors/markdown.js` still hand-rolls its own pair — *definitions* at `markdown.js:31` and `:38` (this row cited only the call site; 10 CodeMirror references in the file on 2026-10-08). **The sheet number was wrong twice over, re-derived 2026-09-07: 21 rule blocks / 60 declarations, not 30 and not 40.** The original 25+4+1 was never reproducible as a block count either — `projects-sidebar.css` has held 16 CodeMirror blocks at every commit back to 2026-08-28. The caps (1500-line selection, 1 MB view-only) are on `preview.js` alone, confirmed | `editor-module` | partly · *Re-checked 2026-10-09:* `markdown.js:45-50` loads CodeMirror itself. |
| **W13** | front end | **Raw px/rem literals — re-derived a THIRD time, 2026-09-07, and the definition finally holds still.** 160 / 740 reproduce exactly, but only because the regex reads raw file text *including comments*. Counting literals **in declarations**: **133** across the eight page sheets, **650** in `lib/`. Two things the row hides: `lib/tokens.css`'s 44 literals ARE the scale definitions — the token layer, not violations — and `lib/molview/molview.css` alone is **252**, 39% of the whole `lib/` figure. So "lib/ carries 740" is really "MolView carries 252, and the rest of lib carries ~400". 777 → 384 → 160/740 → 133/650 are four scopes, not four measurements | roadmap § 7.4c | partly · *Re-checked 2026-10-09:* open — M8; re-derive at its start. |
| **W15** | front end | **Sealing the MolView module's internals and finishing the ES-module conversion** — both **browser-verified** before they count. ~~Plus routing the CLI through the shared codec~~ — **DONE 2026-09-22**: `xv2xyz` (`cli.py:968`, a pair on 2026-10-08) was the last CLI converter writing a lone geometry, and the writers no longer take a path at all, so the class is closed rather than swept (`model/structure.md` §§ 2.3, 2.4). Still open here: exercising the last annotation-channel kind. Re-measured 2026-10-08 (first 2026-09-07): `lib/molview/` has **no `_seal.js`** where `spectrumchart/` and `vibrationview/` both do, and `results.html` loads 5 scripts as classic `<script defer>` against 10 on `type="module"` (seven against two in September) | `web/overview.md` § 4, § 6 | partly — the ES-module half's remaining list is `web/overview.md` § 6 (the `spectra` engine on it: `lib/spectra/core.js` loads as a classic script, `results.html:250`); the sealing half is re-measured at M8 — the shared 3Dmol embed it named (`lib/viewer/`, task #104) is gone *(2026-09-29)* · *Re-checked 2026-10-09:* no `lib/molview/_seal.js`; `spectra/core.js` loads as a classic script at `results.html:241`; 14 more plain classic `<script>` tags are uncounted. Its *task #102* and *#103* (the classic registry and `lib/inspectors` → `presenters`, the results module and the shared primitives as modules) are cited at `web/overview.md:183-189`, `spectrumchart.md:1163-1164`, `spectra.md:652` — none done. |
| **X3** | front end / science | **The Cell page commits a box and returns no seam verdict.** Verified 2026-09-22: `_seam_notices` has one caller, `/api/modify/slab` (`web/blueprints/modify.py:696`; `classify_seam` itself at `modify.py:741/781` and `wizard.py:230`, 2026-10-08); `POST /api/structure/periodicity` (`build.py:381`) runs `apply_edit` + `validate_periodicity` and neither calls it. So the canonical junction walk — build slab, see `collision`, set `c` on the Cell page — gets **no answer at the one moment the geometry is complete**, which is where `science/junction-cell.md` § 6.1 says the loop closes. Re-building to get a verdict would overwrite `c` with the extent again. **A report, never a gate** — the box is the author's to set. *(The CLI half of this was raised and DROPPED on the user's ruling, 2026-09-22: "people working with CLI would know what they're doing… leave that out." This row is the browser only.)* | found 2026-09-22 | open · *Re-checked 2026-10-09:* open — `_seam_notices` has one caller (`blueprints/modify.py:696`); the periodicity route (`build.py:381`) asks none. `junction-cell.md` § 6.1 marks it. |
| **X4** | transport / metadata | **The transport structure's metadata seam** — ①, ④ and ⑥ built (`transiesta.py:360-386` `species_order`, now I15; the catalogue's `shared`; `sidecar.py:13-16`); the measurement is in the 2026-10-09 archive. **② open**: `wizard.as_structure` builds the lead's `Structure` field by field and drops `annotations`, `info`, the identity columns and now `customized` (`transport/wizard.py:106-146`) — the lead is carried from the device's through its doors (`frame_at`, `replace`, the region subset), never rebuilt. **③ open**: `transiesta._compute_cell_from_extents` is live (`transiesta.py:171,333`; `wizard.py:392`; `test_transport_cell.py:158-173`) against `transport.md` § 2a.9, *a transport calculation derives no cell* — read which road reaches it; if none, it is deleted with its test, if one does, that road is refused by I14. **⑤** rides with V1.9 (M2m) | `engines/transport.md` § 2a.9, § 6.2; `model/structure.md` § 2.2a | ② ③ open; ⑤ M2m |
| **A1** | structure / validation / tests | **The structure-API audit's open findings — moved here 2026-09-25, when the audit was archived.** The audit (seven reviews of the structure API, 2026-09-22/23) held its findings and its cleanup order outside this list; they are the rows below and § 5r, and the audit itself is the evidence record, measurement by measurement: [`archive/2026-09-22-unification-audit.md`](?doc=archive/2026-09-22-unification-audit.md). **Every row is as measured on 2026-09-23 — re-derive before acting (§ 5a).** The origin-rule sites it found are W33's, not these | the unification audit, archived | open · *Re-checked 2026-10-09:* open — § 5r keeps the order; A1.2, A1.3, A1.7 and A1.16 archived. |
| ↳ | **A1.1** | **`molbuilder validate` with no `--engine` runs no cell check.** An explicit left-handed cell → `n_errors 0`, exit 0; with `--engine siesta` → `cell.left_handed`, exit 2. `cli.py:376` branches to `validate_geometry` | audit § 1.2 | open · *Re-checked 2026-10-09:* `cli.py:386-389`; `validation/geometry.py:101-111` returns silently on a left-handed cell. |
| ↳ | **A1.5** | **The pair's doors re-derive one rule.** Eleven spellings of *"is this a structure path"* (`workingcopy_structure.py`, `files.py:257`, `selection.py:99` — dead, `siesta/input.py:1872,1902`, `build.py:240,722,813`, `cli.py:93,773`, `structure.py:94`) and a fourth in the browser (`task-setup/viewer.js:1144` derives `<stem>.molstruct.json`). Rename strands labels: `water.xyz → notes.txt` leaves `notes.molstruct.json`, and the reverse adopts a foreign sidecar unchecked. ~~A `.pdb` source travels under three names — describe records `c.source.pdb`, the codec writes `c.source.pdb.xyz` (`files()` never passes `fmt`), prep looks for `c.source.xyz` → *"the structure this calculation describes is not here"*~~ — *done 2026-10-14 (2.9.6)*: every source is `<label>.source.xyz` (`source_files`), and `files()` takes the format its target names, as `write` does (`StructureCodec.format_named`, the one rule; the export door states its XYZ) | audit §§ 1.8, 1.8a, 1.18 | open (the rest) · *Re-checked 2026-10-09:* the browser derives the sidecar name itself (`task-setup/viewer.js:1053`); a `.pdb` target is not replaceable (`workingcopy_structure.py:117,330-333`); `files('mol.pdb')` writes `mol.pdb.xyz` (A1.13). |
| ↳ | **A1.6** | **`Structure.replace()` still hand-enumerates.** Nine fields in a literal, five more from `_carry_nonatom()`. Completeness is pinned by a test iterating `dataclasses.fields()` — but its fixture has `annotations={}`, so a `replace()` that drops annotations passes it. Deriving the carried set from `dataclasses.fields()` makes it complete by construction; `frozen_atoms` stays excluded (a property over `regions`, no storage). Read `replace()`, `_carry_nonatom()` and `__post_init__` end to end first — it is the most load-bearing method in the model (`structure.py:1091`, `:1778`; 2026-10-08) | audit § 1.16e; the 2026-09-22 handover | open · *Re-checked 2026-10-09:* `replace()` lists the fields by hand (`structure.py:1272-1336`). |
| ↳ | **A1.8** | **Rules re-derived at the call site.** `estimate_partial_charges` (`chemistry.py:1392`) is label-blind (water `O1,H2,H3` → **0.0 D**; `_DEFAULT_EN = 2.20` is hydrogen's value); **twelve dead `axis_kind` fallbacks** (13 on 2026-10-08) (11 × `("isolated",)*3`, 1 × `()`, all unreachable, plus `validation/siesta.py:734–737` resolving the opposite way — and `transiesta.py`'s was `("periodic",)*3`) — pure deletion, first; the k-sampling hint measures the gap in two frames (hexagonal cell: the hint says ~5.5 Å, the perpendicular gap is 4.16, `_min_image_distance` 6.5) | audit § 1.12a–c | open · *Re-checked 2026-10-09:* twelve dead `axis_kind` fallbacks (e.g. `validation/geometry.py:116`, `validation/__init__.py:115`); `chemistry.py:1392`. |
| ↳ | **A1.9** | **`load()` restamps `schema_version`.** A sidecar written at 7 reads back as 9 through `molstruct.load` — so no reader can say what version a file was. W33 moved the schema to v10 and inherited it: `parse/sidecars/molstruct.py` stamps v10 on every read | audit § 1.13 | open · *Re-checked 2026-10-09:* `_normalised_dict` stamps the current version (`parse/sidecars/molstruct.py:110,236-249`) — v11 now. |
| ↳ | **A1.10** | **Placeholders stored as facts, and two builder defects.** The backbone check keys on `rid − 1`, so a 5P duplex is refused (*"residue 4 O3' → residue 5 P 16.74 Å"*) with the blame on `$X3DNA`; rdkit-added hydrogens carry `(1, MOL, A)` and are persisted as real identity (7 of 12 atoms); `smiles.py:155,187` and `_common.py:35,40` spell index names `C1, O3, H4…` as stated identity, so every SMILES-built molecule ships a sidecar; `_amber.py:80` warns *"requested B-form … not enforced"* on every build. The placeholder must be carried apart from the data — a shape decision before a fix | audit §§ 1.15, 1.18 | open · *Re-checked 2026-10-09:* open. |
| ↳ | **A1.13** | **One rule, several enumerations — and two strictnesses.** ~~Three `_CONTAIN_EPS`/`_EPS` (one unread); `cell._contains` re-implements `Structure.cell_contains_atoms`~~ — gone with W33 (`6c705058`): containment is one distance tolerance, the hand-off's; `affine`/`concat` hand-list the columns; `vacuum=-5` is accepted by the model and refused by the gate; `annotations` accepts a `str` index `regions` refuses; `set_channel` installs then validates, so a refused channel stays and breaks `copy()`; ~~`files('mol.pdb')` writes an XYZ inside~~ (done 2.9.6, A1.5); `apply_to_structure` on a partial payload resets the cell. Fix each at its owner, never at the instance | audit §§ 3, 4 | open · *Re-checked 2026-10-09:* open — ~~`files('mol.pdb')` writes `mol.pdb.xyz`~~ (done 2.9.6). |
| ↳ | **A1.14** | **The same shape, latent** (unreachable or harmless today). A second PDB reader (`builders/backends/_common.py:57–93`: `Mg → M`, `Cl → C` on a blank element column); a second deserialiser (`selection.py:131–173`; `_shared.py:124–165, 1349–1406`); eight metadata-dropping rebuilds (`add_hydrogens`, `protonate_phosphate_oxygens`, `_drop_overlapping_hydrogens`, `relieve_clashes`, `_strip_5prime_phosphate`, `select_chain`, `_patch_residue`, `_fix_methylene_hydrogens`); `describe.write_description(struct=None)`; `_reset_to_derived`'s own 1e-6 threshold; `_validate_transport_kind` reading the raw cell; `transiesta.py:601,705` bare `+ 1` into the deck; `pyscf/input.py:1575` serialising with `json.dump`; `chemistry.py`'s `_adjacency` on raw `"H"` and a second periodic table; `emit_atom_metadata` dropping a kind-less channel silently; `build.py:410/1296`; the backend set spelled in seven places | audit § 1.18 | open — its transport half is the electrode block's hand-written `+1` (`transiesta.emit_electrode_declarations`, `:432`); § 5r's order *(2026-09-29: not done by § 5u step 1, which had claimed it)* · *Re-checked 2026-10-09:* the transport `+1` sits at `transiesta.py:500,580`. |
| ↳ | **A1.15** | **Rules nobody wrote down** — the stale comments are fixed (`structure.py:5-16` and `workingcopy_structure.py:26-32` read right on 2026-10-08; `structure.py:892` and `validation/sidecar.py:3–7` not re-read). To state: what the CLI's single stdout stream carries; `cell: null` in a load body; a partial `apply_to_structure` payload; a `.XV`'s companion sidecar; whether *Delete file* pairs the sidecar; `Frame.lattice` against `Structure.cell` (which frame a run artifact is in is W33's) | audit § 1.18 | open · *Re-checked 2026-10-09:* the single-stdout-stream item is answered by the 2026-10-09 ruling (a lone file is read as atoms and says so, `cli.py:421-424`); `Frame` still exists (`frame.py:64`). |
| ↳ | **A1.17** | **The tests: the count must come down, and three rules are unpinned.** Unpinned: `write(struct, "x.pdb")` producing a readable pair (restoring the pre-2026-09-07 bug leaves 460 passed); geometry-before-sidecar, both-or-neither (reversed → 441 passed); the explicit-cell centring branch (deleted → 441 passed — W33 makes it the rule and pins it). Blind or shape-asserting: ~~`test_periodicity_gate.py:1126` (inverted — fails on cosmetics, passes on deletion), `test_cell.py:202` (`"a " in message`)~~, `test_cell.py:588` (a signature), ~~`TestDocMatchesTheDoor` (blind to a shrinking `OPS`)~~ — the struck three fixed or retired with W33 (`6c705058`), which also pins the explicit-cell centring branch as the rule (T5, T1). ~~~24 duplicates in twelve clusters — 16–17 tests carry *"the default isolated vacuum gap is 3 Å"*~~ (gone, 2026-10-08). ~~`structure_hash` is still hand-built in test fixtures (18 test files name it on 2026-10-08; 15 fixtures in September)~~ — converted 2026-10-14 with V1.9: every pair a test writes goes through the codec, and the files still naming it write an older version on purpose or are the vibration result's own hash (V1.39), three matching a different error than they name | audit § 5a | open · *Re-checked 2026-10-09:* 19 test files name `structure_hash`; the front-end half of the atom-number invariant (`model/overview.md`; `structure-annotations.md` § 6 now names the real homes, `_atom.js` `toDisplay`) is this row. |
| ↳ | **A1.18** | **The contracts' line numbers are 15 % right.** 6 of 39 `file.py:NNN` references resolve (re-derived exhaustively, 2026-09-22); seven symbol names have never existed; five retired concepts are written as current. `model/parse.md:355` already states the rule — a line number is a pin; the fix is one mechanical pass, then the behavioural list | audit § 2 | open · *Re-checked 2026-10-09:* open. |
| ↳ | **A1.19** | **The test harness can report a green suite that is not green.** In `tools/`, open and not held: ~~`run lf` with nothing to rerun shouts NOT GREEN (exit 5 on a green last run — `exit 4 \| 0/0 ran` appears 302 times in history, so the guard gets trained away)~~ — **closed 2026-09-30** with the spread runs' review: `run lf` runs only when pytest's last-failed list names a test file here, and otherwise says *nothing to rerun* and exits 0 (`testing.md` § 6.1a); ~~`testrun.py failed` emits node-ids with a `[teardown]` suffix pytest refuses; the head line stops summing (`2/2 ran \| pass 1 FAIL 3`)~~ — fixed (`testrun.py:319`, `:234`; 2026-10-08). Same class, open: `progress_plugin.py:68`'s `except OSError` silently disables the writer and `cmd_status` returns 0 for no-data; the env canary's *DISARMED* goes through `warnings.warn` with no hook, so a run whose canary proved nothing reads like one that proved everything | audit § 0c | open · *Re-checked 2026-10-09:* open — `tools/progress_plugin.py`'s `except OSError: pass`; `conftest.py:401`. |
| ↳ | **A1.20** | **Residue, with the step-0 read done** (`process/code-audit.md` § 1d): ~~five `_enumerate_files` buckets with no reader, built on the Watch polling path with four directory scans~~ (gone, 2026-10-08); three legacy shim classes and the five test-only names that depend on them (one decision); ~~`sha256_of_file` (`sidecars/molstruct.py:220`), whose docstring calls it the `structure_hash` pin (X4 ⑤)~~ (gone 2026-10-14 with V1.9's pin); `selection_rules` (`script_emit.py:391-448`), a format field with no producer and no consumer; `sidecars.molstruct.load_text` (×2 on 2026-10-08 — re-read before acting); ~~the `*-electrode` convention two modules advertised and `sort.PARTITION_LABELS` could not compose~~ (gone 2026-10-02, § 0b item 3). Last in § 5r's order | audit § 5 | open · *Re-checked 2026-10-09:* open — `sidecars/molstruct.py:230,706`. |
| ↳ | **A1.21** | **Audit #2 — the rest of the tree, planned and not started.** ~70,000 of ~110,000 lines were outside the audit: `web/static/lib/` (33k — a language boundary), `jobset/` (12k — a process boundary), the rest of `web/blueprints/` (~9k), `runwrap.py` (5k), `runtime_config`/`template`/`monitor`/`checkpoint`/`task` (~9k), `config/` + `validation/` (~8k). Kept separate because the failure shape differs (one fact per side of a boundary, not one door per operation) and so does the evidence (a JS finding needs no Python fixture; a wrapper one needs a submitted job) | audit § 7c | planned · *Re-checked 2026-10-09:* open, with A14: the 44 hand-built run-file names counted 2026-09-07 predate `runfiles.RunNames` — re-measure by review. |
| **S1** | architecture seams | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — three engine branches in `runwrap.py` and 22 lines carrying an engine-name literal (2026-10-08; four branches and 128 literals when last counted, 2026-09-06) · *Re-checked 2026-10-13:* conditions on an engine at `runwrap.py:592`, `:772`/`:816`, `:1914`, `:1962`, `:2116`, `:3171`, `:3183`, `:3432`, `:3920`; 22 lines carry an engine-name literal. Closed by the Python pass's phase 4 as proposed (Q19: one wrapper hook per engine on `jobset/engines.engine_seam`), when the user takes it. |
| **S13** | architecture seams | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built, re-verified 2026-09-20.** Nothing in the tree names it at all now — even the `transport/wizard.py` comment that used to is gone, so the only record that it is owed is `engines/transport.md` § 8 and this row · **→ after § 5u step 4, unscheduled** (§ 5u.5: a sweep needs a ladder that has produced a curve) · *Re-checked 2026-10-09:* open; `transport.md` § 8 now cites this row. |
| **N10** | parse / front end | **A CALCULATION ROOT IS NOT A RUN DIRECTORY, and the Results tab has only one notion.** Measured on the real transport ladder: `jobset_status` answers 5 stages all `pending, prepped, not launched`; `run_status` -- which `/api/results/dir` calls unconditionally -- answers `running, no result file yet`, so the tab tells a person a calculation nobody launched is running. The tab offers that root its own INPUT structure as the result, and says nothing about the five stages. `transport.md` § 2a.12 has required the ladder's state, the curve with its treatment named, and the provenance chain since before the surface was built. **The predicate already exists** -- `checkpoint._is_bundle_root` -- and a second copy in `parse/dirs` would be instance 14 of § 8 | § 5c.3, `transport.md` § 2a.12 | **RE-MEASURED 2026-09-23 — the headline defect is CLOSED; (e) folded into § 5u step 9 (2026-10-08); the two-predicate question open.** § 5c.3's two premises are both stale: `/api/results/dir` no longer calls `run_status` unconditionally (`results.py:185-229` asks `calcdirs` first — `record` / `read` / `root_of` — and builds the ladder there; a container gets `status: None`), and the public owner that landed is a NEW module, `calcdirs`, **not** the `checkpoint._is_bundle_root` step (a) proposed — which is still private at `checkpoint.py:379`. So (a) is superseded, (b) is done in substance but via `place == CONTAINER` rather than a `jobset_status` ladder (`jobset_status` has zero hits in `results.py`), (f) is DONE (`inspectors/transport.js:180-205` renders the provenance chain), and **(c) the payload and (d) the ladder view are DONE 2026-09-24** (`ladder` on `/api/results/dir`, the empty-state card's table, `results.md` § 2.4); **(e) the inspector's parser → § 5u step 9** — `transport.js` no longer calls `JSON.parse` (2026-10-08); what remains of the reader is step 9's. **AND A NEW, SMALLER QUESTION:** there are now TWO predicates for *is this a calculation root* on DIFFERENT evidence — `_is_bundle_root` tests for `task.json`/`job-set.json` existing, `container_or_run` reads `task.json`'s `shape`. The § 8 duplication § 5c.3 warned about is real in a milder form, found by a browser walk that 2,800 passing tests missed · *Re-checked 2026-10-09:* `container_or_run` is gone (`runs.place_of`, `runs.py:56`); two root checks remain on different evidence — `checkpoint._is_bundle_root` (`:379-382`, task.json or job-set.json) and `calcdirs.root_of` (`:109-130`, task.json). No failure on molbuilder's road (job-set.json always sits beside task.json): the unification only. |
| **F9** | execution | **A group prep prints its pseudopotential copy lines and the `geometry.h_ratio` note (a warning until 2026-10-11) once per rung** (three times each) — found on the road walk 2026-10-08, not reached by B8 | § 5x.7 F9 | open · *Re-checked 2026-10-09:* `prep_group` (`prep.py:2970`) does not de-duplicate the members' notes; its h_ratio half is A1.12. |
| **F11** | execution | **The wrapper's kill line is unseen on a killed point**, and two more mechanisms have not been seen on the road: a **cold sweep** (`launch task --stage device --cold` on a finished sweep) and the **take-over hop** (a point taken over from a run that took it over). One walk closes all three — ~45 minutes of this machine (a device point ≈ 15 min), launched through Task setup and left to run; **waits for the user's go** | § 5x.7 | open — the user's · *Re-checked 2026-10-09:* the cold sweep is seen on Q15's run E (`tests/test_transport_on_a_real_junction_e2e.py:238-240, 396-409`); the kill line and the take-over hop are not. |
| **F14** | parse | **The unconstrained max force and a periodic slab's pressure are never shown** (the monitor and wrapper print the constrained max); TranSIESTA's *FORCES WRONG* is not captured | § 5x.7 F14 | open · *Re-checked 2026-10-09:* no `FORCES WRONG` capture anywhere; `report_fields.py:58-61` offers `max_force` to a TranSIESTA device (§ 5t.4). |
| **T-ENV** | tests | **Installer tests answer the environment manager in-process**: `tests/test_envs_one_answer_about_an_env.py` (H6, `dispatch_into_env` replaced to return a typed PySCF verify answer) and the stubs `tests/test_envs_install.py` writes -- a command's answer swapped in-process, which tier 1 forbids (`testing.md` § 0); judged under § 5y's rule in their own pass | § 5y | open · *Re-checked 2026-10-09:* open (`tests/test_envs_one_answer_about_an_env.py:182`). |
| **PYF-T** | code hygiene | **The tests, `tools/` and `scripts/` are not pyflakes-clean**: 56 findings — 52 in tests, 3 in `tools/`, 1 in `scripts/` (47 unused imports, 7 unused locals, 2 f-strings without a placeholder), measured 2026-10-13 with `python -m pyflakes $(git ls-files '*.py' ':!:molbuilder/*' ':!:tests/data/*')` — the package is clean (`conventions.md` § 1) | Q19's Python P1 | open — each read before it is removed: an unused import in a test can be a fixture pulled in by name |
| **L2** | tests | **`tests/field/test_ask_the_target.py` fails instead of skipping when invoked outside its batch** (`. does not read as a machine record`: the record path is unset) — a field test gates itself on its backend (`testing.md` § 3) | B8 round 2 | open · *Re-checked 2026-10-09:* open — `tests/field/test_ask_the_target.py:49-51` asserts instead of skipping. |
| **W56-2** | transport / citation | **The citation reads a cited run beside the run door.** `compose.py:252-262` refuses a folder holding more than one `.fdf`, so a flat relaxation of several stages — which jobset makes — cannot be cited, and its remedy (*"Keep one deck"*) tells the person to edit a run folder; the labels come from that deck's block or a lone `sidecars_in` file (`:445-468`), not from `runs.declared(run)` (`runs.py:308`), the one reader of a run | `engines/transport.md` § 3.1 | **built 2026-10-09 with Q17-c**: a cited folder is the run it speaks for (`runs.run_of`), its deck `Run.deck`, its geometry `runs.openable` and `view_of`, its labels `runs.declared` — the one-`.fdf` refusal and the sidecar-beside fallback gone (`compose._classify_run`, `labeled_citation_structure`) |
| **R11** | execution / tests | **Launch's test-only renderer fallback.** Three test files call the wrapper renderer; two rely on its fallback, and both hand-write `environment.json` — a hand-built record `testing.md:54-55` retires: `tests/test_launch_door_gate.py:37-46`, `tests/test_warm_file_inventory.py:30-55`. Then `machine_record` becomes required in `render_run_wrapper` and the fallback goes (`runwrap.py:2944-2945`, `:2968-2970`); launch's own (`_render_sbatch_for`, `runwrap.py:4224-4228`) stays | § 0b (W54) | open — judge the two tests first |
| **W57-G** | execution / tests | **Rules no test holds since the W57 retirements**: a killed run leaves no conclusion marker (an engine-tier kill); a stage's `overrides` (no road key for it); the PySCF effective-parameters fence (an engine-tier assertion on the H₂ log); `set -e` across the preamble; a warm retry's own launch record (`retry_of`, 0 test hits); MPS and ranks-per-GPU (only the `run_sh_lacks` side, `gpu_contract.toml:188`) | W57 (2026-10-07) | open — each an assertion on a real run of M2n's batch, never a fake |
| **W57-M** | execution / PySCF | **A stated PySCF memory cap reaching `max_memory`** — only the unset rows exist (`launch_values.toml:206-216`); none for a stated cap reaching `max_memory = N` (`pyscf/input.py:494`) | W57 | open |
| **W57-T** | tests | **Tests built on hand-made inputs, for retirement**: `tests/parse/test_tbtrans_out.py` writes empty files named by TBtrans's rule (`:13-22`); `tests/test_jobset.py` builds job sets by hand (`:56-59`, `:107-115`) — `testing.md:50-55` retires both, never sweeps them; and the ten test files that render through `spec_for` + `render_deck` directly, swept to prep's road | W57 | open |
| **W57-O** | execution / records | **Old-data tolerances**, each an explicit error under *build molbuilder, don't clean up handcrafted input*: `record.py:1323-1324` skips the conflict check when a copy names no machine; `script_generation` refused by name in three doors (`runtime_config.py:705-715,958`; `jobset/machine.py:137`; `envs/initconfig.py:253-259`) — one door; `Environment.from_dict` drops unknown keys (`record.py:363-398`) | W57 | open |
| **W57-K** | execution / library | **A kind left to a default.** Fifteen functions default their kind to `"optimization"` — `template.py:1003`, `siesta/input.py:555`, `script_emit.py:1043`, `pyscf/stages.py:33`, `validation/siesta.py:648`, `validation/__init__.py:122`, `validation/stages.py:49,117`, `pyscf/input.py:171`, `_shared.py:540`, `siesta/stages.py:91`, `kmesh.py:130`, `validation/siesta.py:197`, `validation/pyscf.py:166`, `_shared.py:711` — every caller on the road states it; ~150 calls, most in tests, lean on the default. And `template.MIGRATE_HINT` (`template.py:2290-2293`) names `--bundle` without a complete command | W57 (explicitness) | open |
| **W56-H** | tests / PySCF | **A PySCF relaxation with a held atom, end to end** — its progress log's `# frozen_atoms` line and a trajectory read with the atom held; only the vibration e2e holds one today (`test_vibration_e2e.py:501-537`) | W56 | open — M2n's batch |
| **PL-R** | execution / layout | **`project-layout.md`'s owed rulings**: *must every stage be measured, and how is a verdict whose environment changed shown* — the user's; `summarize` reporting the latest of two runs without saying so (`:687-693`) — a defect under the explicitness rule, fixed, not ruled. (*May one folder hold two ladders* is no question: one `task.json` per folder) | `execution/project-layout.md` | **answered 2026-10-13** — *must every stage be measured*: no, the contracts say so already (a run needs no benchmark, and any stage may be measured, which one a judgement — `web/task-setup.md` § 6.2b, § 11); *a verdict whose environment changed*: shown with the environment it was measured in, and a difference from today's said beside it — never set aside or applied for the person (finding is not permission, `generator.md` § 4.3a), written into `project-layout.md` with step 1's fix. **The defect, step 1**: `summarize` says how many runs a trial has and which it reports — `project-layout.md`'s own condition |
| **DOC-TOC** | tests / docs | **A test writes into the checkout when a document is new.** `tests/test_docs_tab.py::test_toc_returns_each_document_once` asks `/api/docs/toc` over the real `docs/`, and the toc builder persists an unlisted document into `docs/toc.json` (`web/blueprints/docs.py:329-345`) — the suite's checkout guard (`conftest.the_suite_leaves_your_checkout_alone`) fired on it 2026-10-09, when this consolidation added its archive. The same rule is held on an isolated tree by `:200` and `:249` | found 2026-10-09 | open — judged redundant: retire it |
| **Q11** | execution / scheduler | **The named-queue check in two doors** — `placement.launch_refusal` (`placement.py:28`) and `scheduler/place.place` (`scheduler/place.py:78`) | the queue of 2026-10-02 | open — step 2: one door, asked by both |
| **Q8** | configuration | **Ruled 2026-10-02, not built**: one `proxy.trust` setting replacing `auth.trust_proxy` and `rate_limit.trust_proxy` (`configuration.md` § 8); a benchmark's grid stated by the description, refused when it is not — no grid the machine proposes | the queue of 2026-10-02 | ruled, not built — step 2 |
| **Q2f** | code hygiene | **Absolute imports, package-wide** *(user, 2026-10-03: "i agree with a, but like a focused session for that with agents and reviews. let's finish this main goal of transport and framework related tasks")* — every `from ..x import y` inside `molbuilder/` becomes `from molbuilder.x import y`, the rule written into the conventions; **and the shipped files' imports in the same session** *(ruled 2026-10-06, W57 decision 1: "B, with Q2f as one session")*: each zip (`mb_monitor.pyz`, `mb_vibration.pyz`, `mb_pyscf.pyz`) keeps molbuilder's folder layout so the package's absolute import resolves inside it (user, 2026-10-06: "for the first form of import why can't we use absolute path?"); the try/except ImportError pairs leave the 22 shipped files. Until then a NEW module is written absolute | the queue of 2026-10-03 | agreed (option a) — its own focused session in step 3, as the user asked: a script with an agent review, then the full batch |
| **PDB-T** | model / codec | **`to_pdb` writes a title longer than 70 characters past column 80** of the TITLE record — invalid PDB (`structure.py:1978`, `{self.title:<70s}`; `model/structure.md` § 2.2c) | found 2026-10-09 | open |
| **CARRY** | various | **Smaller items carried by archived sections**, each to become its own row when taken: `tools/classify_source_reads.py`'s `_OVERRIDES` re-keyed by anchor, not line (`:102,117`; § 5h); TS6's `setInterval` pin (`tests/test_results_state_contract_spectra_js.py:238`; node v24 present — unblocked) and TS9's undocumented-test count to re-measure (§ 5m); `generator_mismatch`'s severity (`validation/siesta.py:149`), `detect()` unable to say why it refused (`parse/registry.py:46-89`), Task setup acting on `onChange` (`task-setup/viewer.js:3811-3839`) (§ 10); P6 — an A/B swap passes `tests/test_backends.py:306`; P7 — the `ByElement(("Au",))` mutant lives in `tests/test_atom_selection.py:342-355`; P12 — `lib/projects/checkpoint.js:33`'s no-poll property is prose; § 11.5a's error surface, unwritten; `LAYER_TOL_ANG` untested (§ 11); the route-catalogue sweep (`web-api.md`: 11 documented routes that did not exist, 9 live ones undocumented, 2026-10-08 — re-measure after 2026-10-09's corrections); the noted items in M2a (an unreadable zip or an exception escaping `main` reports in Python's words; a failing `ending` exits 1, which the wrapper treats as 2), M2b (the Build route's peptide branch collects no warnings, so the C-terminus warning reaches no web user) and M2c (`create_app(config={})` reads the disk config through the snapshot); N5d (the label handed to the wrapper through a `Resources` field — waits for a yes); `BlockSize` as a benchmark axis — built as a value axis (`prep_inputs.py:94-106`), re-derive what is left | the archived sections | open |
| **TPL** | engines / template | **`engines/template.md` § 12.1's open rows without one of their own**: row 3 — G3's `kind`, which no layer dispatches on; row 4 — `read_by` declared and unconsumed (no caller asks `select(read_by=…)`); row 5 — `kind = "monitor"` with no items; row 7 — BENCH-MARKS SIESTA-only. (Row 1 is W47, row 8 W44, row 6 closed) | `engines/template.md` § 12.1 | open |
| **SMALL** | various | **Smaller open items found in the contracts** (2026-09-29, each searched in the plan and not found): the Raman/IR finite-difference steps untested for convergence, ΔF ≫ σ_F not estimated, two textbook references owed (`normal-modes.md`); V1.22's dropped warning half; `dm0` seeding (`pyscf.md`); a SIESTA road run whose held-first sort reorders atoms, `info.calculation` on the PySCF road (K18), the v1 reader's residue (`vibration.md`); the G-5a `gpu_used` read-back (`engines/overview.md`); § 6b's two questions (`stages.md`); multi-node MPI and `config init --site`; hand-typed wrapper comment claims (`script-preparation.md`); a test `gpu.md` lists that does not exist; a PySCF residual shown with no tolerance (`run-reports.md`); the per-trial SCF plot (`bench-summary.md`); five known gaps in `projects.md`; `ttl` / `detail` / a `success` level for notifications; whether the markdown editor returns (`presenters.md`); legacy response keys (`web-api.md:76,246-251`); an unexercised APPLY arm and the getter/setter bridge (`results.md`); the calculation-wide block editable only by hand (`task-setup.md`); a per-layer overlay refresh and `.pdb` on the download row (`molview.md`); `title` carried by hand at ten sites (`structure.md`); `MOLBUILDER_LOG` (A1.15); the SLURM-shaped `Resources` (`backend-architecture.md` § 7); the gcc 15 migration against the GPU SIESTA pin (`installation.md`); `references.bib`'s unverified entries | the contracts, 2026-09-29 | open — re-derive each when taken |


---

## 5e. Engine additions — a person's own engine text, as an INPUT

*(Your ruling, 2026-09-05: option 2 — a distinct input to the writer, not a
catalogue extension; **engine-specific**; and **the person is told the
consequence**. Contract-first: this section is the design. **Not started.**
No code has moved.)*

*Re-checked 2026-10-09:* the measurements hold in shape, at new lines —
`render_deck`'s concatenation is `script_emit.py:1382`. Two of the known
defects named below have moved: a stray marker no longer drops text silently —
`check_deck` refuses a deck holding more than one pair of USER-CUSTOM markers
(`script_emit.py:1522-1543`); and transport has the zone — its decks are
rendered by `render_deck` like every other. § 3.5's *byte-for-byte* now reads
*line for line, unchanged* (`job-contracts.md`). `engines/template.md` § 9.2
still describes a `user_custom` catalogue item — the carrier this ruling
rejected; it is corrected when this is built (§ 2, W47).

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


## 5q. The engine offset — scope and order of work *(W33, 2026-09-25)*

*The rule, the name, the operations and the checks are the contract's:
[`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.0.
This section is the scope and the order of work, and restates no rule; its
row is **W33** in § 2. Trimmed 2026-10-08 after the validation walk: P0–P3
are built and P5 is closed by TD8; what stays is P4's remainder (§ 5q.6).
The measured facts, the data-structure and file-access tables, the risks and
the decisions D1–D16 are archived with their verdicts.*

### 5q.3 Protocol agreement — the wire

* **Built** (`6c705058`): one periodicity block from `Structure.to_wire`
  (`structure.py:896–950`); MolView draws the box at `box_corner`, verbatim
  (`render-engine.js:313–317`, `model.js:879–884`); the Cell page's origin
  group sends `box_corner` (`modify/periodicity.js:508–621`).
* The load/save payload's door is `validate_periodicity`
  (`web/blueprints/_shared.py:36–37, 354–385`).
* The Results door reads the run deck's ENGINE-OFFSET record for the axis
  kinds (D5) and states 0. The `.source` pair fallback for a run made before
  the record was retired by B12 4b (§ 0c unit 4): a run reads its own deck.
* ~~The transport deck viewer has no home since step 9 closed~~ — built 2026-10-11 on the Results ladder, for every calculation (§ 2, W30 ④).

### 5q.4 Validation

* **The checks** (contract § 6.0): *the atoms fit* (fractional span `< 1`) at
  the edit; along the transport axis the check is TD3's
  `cell.transport_vacuum` (`validation/__init__.py:459–500`) — a gap at the
  face beyond one layer spacing of the lead is **refused**, measured from the
  lead. T4's *d/2 warning* and R1 are superseded by it.
* **Where the tests live**: T1 `tests/test_engine_offset_reaches_every_deck.py`;
  T2 `tests/test_siesta_flat_run_e2e.py:168–191` (the Results load of the
  measured flat H2: the box at 0, each stage reading its own deck);
  T3 `tests/test_results_export_e2e.py`; T5 `tests/test_cell.py:543–576`.
* **"Nothing translates by hand"** is checked by a code-text review of
  § 5q.5's rows at the end of P4, not by a lint test.

### 5q.5 The inventory — what P4 still reaches

| site | role today | phase |
|---|---|---|
| `structure.py` `to_ase` (`:2034`), `cli.py --pyscf-atom-block` | design coordinates beside a lattice that implies the box at the origin — `to_ase` alone now: `to_extxyz` is gone (`61db1660`) and `to_pyscf` writes no lattice | P4 |
| `transport/deck.py:657`, `validation/__init__.py` | no validation subject: the transport validators judge design coordinates in `resolve_cell()`, which for a lead that states no cell is not the deck's box | P4 |
| `transport/wizard.py:396-400` | the lead's z hand-shifted by its lowest layer — harmless (re-centred), and a translation outside the one door | P4 review |
| multi-frame pairs | **built 2026-10-09**: `Structure.frame_at(i)` states the set's one offset — frame 0's — so every frame reaches `cell.to_engine` with it (`structure-periodicity.md` § 6.0); each frame's deck built 2026-10-10 (Q17-d, `prep._at_frame`) | — |

**The documents to sweep** in P4 — the owner first (the line numbers are 2026-09-25's; re-derive them, and § 6.0's own list was brought to the code on 2026-10-09): `structure-periodicity.md`'s
superseded clauses are still in the text — § 6 (`:606`), § 6.1 clause 4
(`:697`), § 6.1a's corner column (`:781`), § 6.2 (`:961`) — deleted, as § 7
and clause 5 were on 2026-09-25. Then the restatements, counted at planning:
`web/molview.md` (11), `model/structure.md` (11), `web/web-api.md` (6),
`model/structure-molstruct.md` (5), `engines/transport.md` (3),
`engines/vibration.md` (2), `architecture.md` (2), `science/normal-modes.md`,
`README.md`, `model/overview.md`, `backend-architecture.md` (1 each);
`model/parse.md` (the premise *"no deck shifts a structure that states one"*),
`model/structure.md` § 2.2 (the removal rule D2 relied on),
`execution/job-contracts.md` § 3.1 (the block grammar lacks ENGINE-OFFSET),
`engines/vibration.md` § 5.2a, `engines/siesta.md` (centring described for
the derived box only). Dated plans and handovers are history and are not
rewritten; `plans/plan.md`'s live rows are.

### 5q.6 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0–P3** | the name and the contract; data structure + file access; every emitter through `to_engine`, each deck carrying its record; the readers, the wire, MolView and the Cell page | **Done** 2026-09-25/27 (`a2901f4d`, `dcfb371b`, `1df5cc24`, `4adc6832`, `6c705058`, `4a3172e1`, `dba66a9a`); the detail is archived. T2's pin is `tests/test_siesta_flat_run_e2e.py:168–191` (§ 5q.4) |
| **P4** | the checks, the test retirement, the document sweep | the test retirements and the contract's own tables **done 2026-09-25**; the clearance check is TD3's (§ 5q.4). **Open — three items**: ① the document sweep (§ 5q.5's list, the owner's four clauses first); ② the design-frame exports (§ 5q.5 row 1); ③ the transport validation subject (§ 5q.5 row 2) — `wizard.py:395` read in the same review. Done when the review of § 5q.5's rows finds no hand translation and no design-frame export, and the four clauses are gone |
| **P5** | acceptance | **Closed by TD8** (§ 5u.2): the acceptance ladder is `claude-au-bdt-au`, run to its record on 2026-09-29 (§ 5u step 4). The one piece P5 left — the Results tab's view of each rung's engine frame — is § 5x R9 (the 3D viewer's default view, B6) |

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
2. ~~**The four live defects that need no ruling** — A1.11 (`X`), A1.12 (the
   H-ratio on raw labels), A1.5's `.pdb` travelling name (fix `files()`'s
   `fmt`, and A1.13's root falls with it), A1.4's silent `JSONDecodeError`.~~
   *Done: A1.12 2026-10-11; the other three 2026-10-14 (2.9.6).*
3. **V1.10 — the generator renders both halves.** The only structural change
   in the set; it also closes the half-written pair (a NaN in `info` raising
   at the sidecar write after the `.xyz` is on disk). Do A1.3's doc fix in
   the same commit — both are in `structure.md` § 2.4's four-clause block.
4. **A1.1**, ~~A1.4~~ (done 2.9.6) — the rest of the data-loss and uncaught-exception set;
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
10. ~~**A1.2**~~ — superseded 2026-10-09 by the comment-line ruling (`61db1660`): a lone stream is atoms and coordinates, and the read says so.
11. **A1.18** — the line-number pass, mechanical; then its behavioural list.
12. **A1.13** — each at its owner, never at the instance, starting with the
    `replace()` guard (A1.6), which is what makes the rest safe to touch.
13. **A1.17** — the three unpinned rules first (the `.pdb` one before all: a
    regression with its numbers already written down), then the six blind
    tests (worse than absent — they read as coverage), then the duplicates,
    re-running each cluster's mutant to confirm the keeper still goes red.
14. **A1.20** — residue last, and only what has a clean step-0 verdict.

**A1.21** (audit #2) comes after steps 1–4. **Standing on its own:** the 15
hand-built `structure_hash` fixtures (A1.17) ~~agree with the writer only
because the gate is loose — convert each as its file is touched for another
reason~~ — converted 2026-10-14 with V1.9: every pair a test writes goes
through the codec, and a file of an older version is written as that version
wrote it.

### 5r.2 Do NOT "fix" these — confirmed non-findings

Recorded so nobody fixes them by analogy. Re-read against the code 2026-10-08.

- **`frozen_atoms`'s shape is benign.** A data descriptor sends every read and
  write through `regions[FROZEN_LABEL]`, so the two cannot disagree in any
  order; `replace()` handles it by not re-passing it. **Not the defect `pbc`
  was** — the init-field spelling exists so `Structure(…, frozen_atoms=[…])`
  reaches the one place that spells the reserved label.
- ~~**`resolve_cell` really is the one resolver.**~~ **Untrue (2026-10-09):**
  `transiesta._compute_cell_from_extents` computes one for a transport lead
  (`transiesta.py:171,333`; `wizard.py:392`) — X4 ③.
- **The companion-lookup merge stays withdrawn.** What forbids it is a test
  pinning a deliberately more permissive guard, not `parse.md` § 5.3; a shared
  helper with the guard as a parameter would keep both.
- **The second multi-frame XYZ reader is forced**, not duplicated: it
  tolerates a torn final frame (`parse/engines/pyscf.py:405–439`), which every
  live geomeTRIC run has and ase refuses. Unifying it would break live-run
  viewing.
- **Registry overlap: none** — 14 parsers against a 29-file synthetic run
  directory, no file claimed twice.
- **Bare atom-index arithmetic in `parse/`: none that carries data.** But
  `transport/transiesta.py` has real ones (A1.14, still open:
  `:500`, `:580` on 2026-10-09), in the block whose own docstring says an
  off-by-one *"computes transmission through a region that is not the
  molecule, and converges while doing it."*
- **The CLI's seam verdict** was raised and dropped on the user's ruling
  (2026-09-22: *"people working with CLI would know what they're doing… leave
  that out"*). X3 is the browser half only.

---


## 5ab. The remapped state — the mapping contract stated at every output *(RMP; decided with the user 2026-10-14)*

> *(user, 2026-10-14)*: *"there is a mapping contract, all things will return
> to original index from input, no mapping is exposed at the output"*; *"it
> should (1) clearly mark the output result as "remapped", and (2) if the
> result finished, the generate the "restored" mapping result from that as the
> final result. (3) if the prep does not see that final result as "restored"
> it either should error, or warn for a conversion first"*; *"i would make do
> warning and convert (if mapping index record is there, otherwise error)"*;
> *"in the modify/molbuild tab, when loading a data set with "remapped" meta
> state, we should get a warning and if the mapping index is present we can
> allow a restore operation"*; *"that means parser should be reviewed and
> revised accordingly?"*; *"maybe we should enforce that remapping index to be
> saved into the .json if the meta data shows "remapped" state"*.

**The contract it completes** — [`model/overview.md`](?doc=model/overview.md)
§ 2.2: the program sorts a COPY at prep, records the permutation both ways,
and maps every number that comes back through the inverse before a person
sees it; the sorted order never leaks.

**What is true today** *(read 2026-10-14)*. Two kinds prepare their decks from
a reordered copy: a **SIESTA vibration** — every stage, its `relax` stage
included, because the force-constant stage starts from `relax`'s density
(`DM.UseSaveDM`, `engines/vibration.md` § 5.3), stored atom by atom in the
deck's order (`jobset/prep.py`, the SIESTA vibration rung; key `held-first`) —
and **transport** — every rung from the composed junction sorted for
TranSIESTA (`transport/compose.py`; key `transport`). A standalone structure
optimization is never reordered. Two readers apply the record: the vibration
finish (the spectrum is written in the input order) and the transport
composer reading its own record. **Nothing else does**: the parsers, the run
door (`runs.declared`, `runs.view_of`), the Results tab (a reordered run's
frames and labels shown in the engine's order, unmarked, while the same
calculation's spectrum is in the input order), a structure exported from such
a run, `xv2xyz --from-run` — and the force-constant prep, which copies
`relax`'s final coordinates row by row (V1.41: right only while both preps
held the same atoms).

**The rule, decided:**

1. **Stated, never silent.** A structure or per-atom result in an engine's
   reordered order carries the state **remapped**, its index with it. The run
   door marks what it builds from a reordered run; the parsers stay the
   engine's file view, and nothing is reordered silently while it is read.
2. **"Remapped" is never stated without its index.** A sidecar in the remapped
   state carries the full index — the reordering's key and both directions,
   in the shape of `atom-permutation.json`, read by its one reader
   (`atom_permutation`): one data structure, not a second format. Sidecar v14,
   the key optional (absent: the person's own order). A remapped state
   without a complete, self-consistent index is refused on read, by name — only
   a hand edit makes one.
3. **A run on a reordered copy leaves its restored result** when it finishes:
   its final structure in the input order, with the person's labels, as a
   pair (the name to settle; proposed `<label>_<stage>-run<N>.restored.xyz` and
   its sidecar). A force-constant run's restored spectrum already is one
   (`spectra.json`).
4. **A door that builds from a reordered run's output** — prep (V1.41's
   force-constant stage among them), a transport citation, the Molbuilder
   tab's append — **restores** through the index and says so; a remapped state
   with no index cannot exist (rule 2). The Results export and
   `xv2xyz --from-run` build nothing: they carry the state and its index.
5. **The Molbuilder tab**, loading a remapped structure, **warns and offers
   restore**: the atoms back in the input order, every per-atom field with
   them.

**The survey** *(2026-10-14, two read-only agents at `8c5b6be1`, each
finding re-read before it is used here)*. Every reader of a reordered run
shows the engine's order unmarked — the SIESTA parsers (`.out`, `.MD.nc`,
`.XV`, the molwatch log), the run door (`runs.declared`, `runs.view_of`), the
Results load and export, `xv2xyz --from-run`, the transport report and its
PDOS selections, the transport citation of a reordered run (any finished run
of ours may be cited — a vibration's `relax` included, its order then sorted a
second time, never composed), and the force-constant prep (V1.41, which also
imports a parser class directly, against `model/parse.md` § 3). The labels are
not wrong: the deck's ATOM-METADATA block is written from the sorted copy, so
labels and coordinates agree — in the engine's order, unmarked. Handled today:
the vibration finish (through the record's one reader), the transport
composer (reading the record, then mapping by hand in `wizard._atoms_named`
and `compose.py`, against § 2.2's *never by hand*). **Where the index lives
today does not hold**: `atom-permutation.json` is written at the
calculation's root at every prep — in the flat layout every run's own folder,
so each prep overwrites the one before it — and the shared package copies it
into every attempt, the transport electrode rungs and benchmark trials
included, whose atoms it does not describe. The browser rebuilds the metadata
it sends field by field (`model-jobs.js`), so a new field is dropped on every
round trip unless both directions carry it. A pair cited for transport must
carry `info.calculation`, which a job cannot compose. Stray engine indices in
otherwise restored results: `info.relaxation.held_atom_idxs` (V1.42) and
`engine_metadata.fc_range_1based`.

**The revision plan — approved by the user 2026-10-14** (*"yes to all, go with your recommendations"*).

*The constraints every part meets* *(the user, 2026-10-14: "your design need
to guarantee a unified api access, no handcrafted code, the json and xyz
travel, read and write together, no split handling")*:

- **The pair is one unit.** The remapped state is a field of the structure,
  so it is read and written only with its atoms, by the codec, in the
  `.xyz` and its `.molstruct.json` together; no index travels in a file of
  its own (`atom-permutation.json` retires), and the restored result is a
  pair the codec writes.
- **One door for each act.** A structure from a run comes through the run
  door (`runs.view_of`) — never a parser class imported for it (V1.41's
  `SiestaParser`); a restore is `Structure.take` through the record's one
  reader — never an index mapped by hand (`wizard._atoms_named`,
  `compose.py`'s `start[list(...)]`); every write is the codec's.
- **No field listed by hand.** The two places that list a structure's
  metadata field by field are fixed, not extended: the deck's ATOM-METADATA
  block is written from the structure's own serialiser
  (`metadata_to_dict`), not the hand list in `script_emit.emit_atom_metadata`
  / `apply_atom_metadata`; the browser carries the metadata block whole
  through one function each way, not the key-by-key rebuild in
  `model-jobs.js` (`structureFromServer`, `structureForServer`) — so a field
  the structure gains reaches every road with no edit.

*Where the index lives — one home for each thing:*

- **A structure** carries it as one field, `atom_order` — the key and both
  directions, in the record's shape, checked by the record's one reader (split
  into a reader of the dict that the file and the field share). **Present
  means remapped**: the state cannot exist without its index. It rides
  everything the structure's fields ride — `replace`, `to_dict`, the sidecar
  (v14), the codec — and the browser's two directions.
- **A run** states it in its own deck, in the ATOM-METADATA block (the
  sidecar's shape, so the same field), read through `runs.declared`: each run
  its own index — the flat layout's overwrite and the electrode rungs' false
  copy end with it.
- **`atom-permutation.json` retires**: the vibration finish reads the index
  from its deck; the transport record's junction carries it in its sidecar;
  `.transport.nc` carries the index on its atom axis, not a file's name.
- **The identity order is not remapped** — nothing moved, no state.

*Who says it, and who restores:*

- **The run door marks**: `view_of` — the one builder of a structure from a
  run (the Results load, `xv2xyz --from-run`, the citation) — sets the field
  from the run's deck. The parsers stay the engine's file view; the bare file
  readers (`watch parse`, `watch tail`, `runtime-info`, bare `xv2xyz`) read a
  file, not a run, and the contract says so.
- **The Results tab** shows a reordered run in the engine's order, marked
  (the key named); an export carries the state and its index (v14).
- **The Molbuilder tab** warns while a structure is remapped (a condition of
  every answer, not a load-only notice, since notices clear on an edit) and
  offers **restore** — a server operation, `Structure.take` through the
  index, the field cleared, the selection cleared with it. An edit that adds
  or removes atoms on a remapped structure is refused, naming restore (as a
  frame set's is).
- **A door that builds** restores with a warning — the force-constant prep
  (V1.41: `relax`'s restored result, or its final coordinates through its
  deck's index, read through the run door, not a parser class), a calculation
  whose structure is a remapped pair (`_structure_for`), the transport
  citation (restored before the transport sort, so the record is one
  permutation of the person's order — and a run that is not a relaxation, a
  force-constant stage, refused as `CITATION_CONDITION` already says),
  `xv2xyz --from-run`. A remapped state with no index cannot exist (the field
  is the index).

*What a run leaves:*

- **A SIESTA vibration's `relax` stage writes its restored result** when it
  finishes: `<label>_<stage>-run<N>.restored.xyz` and its sidecar, the
  relaxed geometry in the input order with the person's labels and box — its
  job's finish, through `StructureCodec.write_moved`, the payload made by
  prep, the final coordinates from the `.out` by the travelling reader. The
  run records (`info.calculation`, `info.relaxation`) are composed on this
  machine by the door that hands the pair on, as an export does today. A
  restore that fails is said in the session log and does not change how the
  run ended — the doors restore with a warning. The force-constant stage's
  restored result is its spectrum, as today; **transport rungs** are single
  points whose structure is the record's junction — no restored pair, the
  junction stated remapped with its index; the electrode rungs carry none.
- **Stray engine indices**: `held_atom_idxs` cut (V1.42);
  `fc_range_1based` stated as the engine's, under `engine_metadata` where it is.

*The validation round* *(2026-10-14: two fresh agents read the approved plan
against every reader and writer at `03b09e7c`; each finding re-read before it
is used here)*. What it changed — ◆ marks a choice the round forced, for the
user's word with R0:

- **Reordering a remapped structure composes** ◆: `Structure.take` composes
  its order into the field's index (the new `sorted_to_original[j]` is the
  old `[order[j]]`) and clears the field when the composition is the
  identity — so a restore is `take` through the field and a second sort is
  one composed permutation (§ 2.2's *one copy, one permutation*), never a
  stale index under a new order. `SortResult`'s tuples become a view of the
  sorted structure's field — one source from R2 on.
- **One restore, one sentence** ◆: `Structure.restored()` — `take` through the
  field — is called by every door (the route, prep, the citation, the
  append, the CLI's `modify`); an edit that adds or removes atoms on a
  remapped structure is refused by one constant naming it, as a frame set's
  is (`_carry_nonatom`); `concat` (the Molbuilder tab's append, reached by a
  load into an open tab) restores a remapped addition and says so.
- **The warning is the structure's state, not a notice** ◆: the browser's
  notices clear at every settle (`model.js`), a label edit or a Results poll
  among them; the Molbuilder tab and the Results tab show the state from the
  structure's field, read at every settle.
- **Restore clears the selection**: an `OPERATIONS` column for a reorder (a
  count kept, every index moved), and `molview.md` § 11.1's table gains it.
- **The browser's third hand list**: `getStructure()` (`model.js`) lists the
  master copy field by field too; one function each way, all three.
- **The index rides inside the deck's ATOM-METADATA block**, the block
  written from `metadata_to_dict` and read wherever it is read
  (`emit_atom_metadata`'s gate, `apply_atom_metadata`'s early return,
  `Declared.atom_metadata_for`, `render_deck`'s note) — never a `RunView`
  attribute of its own (`watch._frame0_structure` rebuilds the view from the
  block); a field refusal surfaces (`watch.py`'s blanket `except` goes).
- **The codec's engine-own branch is a run door**: SIESTA's own `<label>.xyz`
  in a reordered run (`WriteCoorXmol`, on by default) opens through the run's
  view, marked.
- **A run made before RMP** ◆: its index is in `atom-permutation.json` and its
  deck states none. A door that builds refuses it by name — *prepared before
  this version recorded its atoms' order; prepare the stage again* — as the
  vibration finish already refuses an old deck; a viewer says it and shows
  the engine's order; an old transport record is not loaded
  (`load_compose_record` answers why). No reader of the old file remains.
- **One source for a door that holds the run** ◆: prep and the citation
  restore through the run's deck index, always; the restored pair is the
  person's copy, for the Results tab and the Molbuilder tab — so V1.41's
  deck bytes never depend on which of two sources exists, and R3 does not
  wait on R4.
- **The citation restores both geometries**: `_run_geometry`'s relaxed
  structure and its starting geometry (the leads' *unmoved* gate compares
  the two), and the cited-pair branch too; **a run that is not a geometry
  relaxation is refused** through the verdict `_classify_run` already reads
  (`CitedRun.verdict`), as `CITATION_CONDITION` states.
- **The `relax` finish** ◆: keyed on the deck written from a remapped
  structure; each finish's failure policy a column of `_FINISH_BUNDLES` — the
  force constants' failure the job's, as now, the restore's said in the
  session log and never stopping the relaxation, its load check a warning;
  the payload in a deck block read by `deck_record`, built by
  `pyscf/input._sidecar_for`'s rule — the **run's box**, the engine's origin
  stated, never the person's design cell; the final rows permuted through the
  deck's index before `write_moved`; the catalogue row kept out of the run's
  result roles (`runs.about`), so the run door never puts the deck's sorted
  labels on it.
- **The density handed from `relax` to `freq`** ◆: the `.DM` is ordered as
  `relax`'s deck; when the two decks' indices differ (the held set grew
  between the preps), `freq` starts its reference SCF cold
  (`DM.UseSaveDM .false.`) and says so — the force constants are unchanged.
- **Stray engine indices**: `held_atom_idxs` cut in R1 (it outlives a
  restore otherwise); `fc_range_1based` is the engine's numbering wherever it
  is written — `spectra.json`'s `engine_metadata` and `fc-sweep.json`'s stage
  rows, where the sweep reads `.FC` with it — and is named so.
- **Stages prepared before the upgrade** and not yet launched are refused at
  launch (`submit._as_described` compares the re-planned deck); the refusal
  names the version that prepared them, not *the template changed*.
- **Placement**: `atom_order` in `METADATA_FIELDS` — a v14 sidecar writes
  `null` for the person's order, so "absent" reads "null or absent"; the
  field's key one of `SORT_KEYS`, required; the identity read as no index,
  the split reader answering it.
- **Untrue in the survey, corrected**: transport writes its record's index
  when it composes, not at every prep.

*The order, each phase closed as a milestone (§ 0a):*

| phase | what | tests |
|---|---|---|
| **R0** | the contracts, for the user's review before code — `model/overview.md` § 2.2; `model/structure.md` § 2.1–2.2e (and § 2.2 step 4's *two translators*); `model/structure-molstruct.md` (v14); `model/structure-annotations.md` § 4a; `model/parse.md` § 3, § 5.3, § 5b.1; `execution/job-contracts.md` § 3.1, § 3.4, § 6.1, § 6.3; `execution/project-layout.md` § 1.6.3, § 5.1; `execution/architecture.md` § 3.2; `execution/job-system.md` § 3; `engines/vibration.md` § 5.2, § 5.2a, § 5.5, § 6.5, § 6.9, § 7 (I7), § 8, § 10; `engines/transport.md` § 2a.12, § 3.1; `science/normal-modes.md`'s order note; `web/web-api.md`; `web/molview.md` § 6.2, § 6.8, § 11.1; `web/results.md`; `configuration.md`'s bundle rule | — |
| **R1** | the model: `atom_order` and the record's dict reader; `take` composing; `restored()` and the one refusal; `concat` restoring; sidecar v14; the codec; `held_atom_idxs` cut; the browser's three hand lists made one function each way; the state shown from the field; `/api/modify/restore` with its `OPERATIONS` row and the CLI's | `geometry_pin.toml` (v14 rows, an incomplete index refused), `frame_sets.toml` (restore over a set), the model and browser round trips — every order a **3-cycle** (O O C with the carbon held: `[2, 0, 1]`), so the two directions differ |
| **R2** | said at the source: `sort_by` setting the field, `SortResult` a view of it; the deck's block from `metadata_to_dict`; `declared` and `view_of` marking; the codec's engine-own branch; the Results tab's state; the export and `xv2xyz --from-run` carrying it; a pre-RMP run said or refused; launch naming the version | the Results route's rows |
| **R3** | restored where it builds: V1.41 through `relax`'s deck index (the run door, not a parser class); `_structure_for`; the citation (both geometries, the pair branch, the kind refused); the by-hand maps through the record's reader | the SIESTA vibration e2e on T-CO2's ladder (O O C, a held set grown between the preps) — not `prep_protocol.toml`, whose road runs no engine |
| **R4** | what a run leaves: the `relax` finish (its policy, its payload block, the run's box), the catalogue row, the cold `freq` start when the orders differ | the same e2e |
| **R5** | `atom-permutation.json` retired: the vibration finish reading its deck, the transport record's junction carrying the field (`record.py`'s provenance key and its schema, `compose.record_files`, `.transport.nc`'s atom axis), no reader of the old file left | the vibration and transport e2e — the transport junction's input order **permuted**, so its sort is not the identity |
| **R6** | the transport report: a server door restoring the junction for drawing (`junction.xyz` stays sorted on disk — the device's regions and the leads are read from it); the PDOS selections mapped in, and `outside_device` back out, through the record | a partial PDOS selection on the permuted junction |

**Decided with it** *(the user, 2026-10-14)*: the field's name and home as
above; `atom-permutation.json` retired; no restored pair for transport rungs;
a failed restore not failing the relaxation; the identity order not
remapped; the Results tab marking (not restoring) a reordered run's view;
R6 drawing the junction in the person's order. RMP is new functionality (a restore operation, sidecar v14): its own middle
number, after 2.9.6 is pushed — before V1.39, whose structures it reads.

## 5s. The electronic state — charge and spin as one answer *(W34, 2026-09-25)*

*The contract is [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md)
§§ 2a–2b; this section is the order of work. The row is **W34** in § 2; the
milestone is **M6**. Amended 2026-09-28 (decisions 8–9): spin is decided by the
same class as charge, and Auto-detect's fill is retired. Trimmed 2026-10-08
after the validation walk: P0–P3 are built, P4 all but one site; what stays is
P5 (ES10), decision 4's refusal site, and § 5s.4's open items. The measured
causes, decisions 1–3 and 6–9, the P0–P3 rows and the review record are
archived with their verdicts.*

### 5s.1 The design

The contract, § 2a: four items, one resolver, ten rules (ES1–ES10), a capability
table and the engines' semantics, each fact read from the engine's source. § 2b:
the species.

### 5s.2 Decisions *(user, 2026-09-25: "go with your recommendations on all seven")*

*Decisions 1–3 and 6–9 are built and archived. Two are partly verified:*

4. **Transport's spin defaults from the cited run** (`transport/citation_defaults.py:56–72,
   126–129, 161–194`), and a cited run carrying a net charge is refused.
   **Located 2026-10-09**: `citation_defaults.py:196-207` refuses a charged
   citation by name (ES7) — built.
5. **A vibration built from a relaxed structure inherits its state**
   (`electronic_state.py:253–283`, `_recorded`), and a change is warned (the
   record check, `validation/spectra.py:638–640`) — PARTLY on 2026-10-08.

### 5s.3 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P0–P3** | the contract and its pointers; the items and the class; the checks; the forms | **Done** 2026-09-29 (`chemistry-correctness.md:271–720`; `electronic_state.py:160–252`; `tests/test_electronic_state.py`, `tests/test_chemistry_card_e2e.py`; `lib/detection-chip.js`; `web/blueprints/build.py:122–151`); the rows are archived |
| **P4** | the hand-over: `parse/fdf.py` reads `NetCharge`/`Spin`/`Spin.Fix`/`Spin.Total`; transport defaults its spin from the citation and refuses a charged one; the relaxation record carries the state, the vibration defaults from it, and the record check compares it | **Built** 2026-09-29 — the fdf reader `parse/fdf.py:158–175`; the citation's spin `citation_defaults.py:56–72, 126–129, 161–194`; the inherited state `electronic_state.py:253–283`; the comparison `validation/spectra.py:638–640`. the charged citation's refusal `citation_defaults.py:196-207` (decision 4) |
| **P5** | the read-back: SIESTA's `.out` (net charge, fixed or converged moment) into the run record; PySCF's class, ⟨S²⟩ and stability recorded; the transport record reads both TBtrans channels; the Results tab shows asked against used, and a difference is a finding | **Open** (ES10): no reader of SIESTA's moment or charge, none of PySCF's ⟨S²⟩. The TBtrans half is § 5x B6 / R1 — `transport/record.py:453`'s own glob replaced by `tbtrans.transmission_files`, which knows the channels. Done when a spin-polarized run's moment and a UKS run's ⟨S²⟩ appear on the Results tab, and a two-channel transmission is read |

### 5s.4 Found on the way — named here, fixed only where a phase says so

* A fresh PySCF bundle's first `prep` says *"this calculation is already under way
  here: warm files at the root: <name>.source.xyz"* — the structure copy `init`
  writes is read as a warm file. **Re-check**: likely obsolete since D20 (the
  pair is written through the hand-over's own call, named for the label —
  § 0c unit 1, done 2026-10-04); not reproduced on 2026-10-08.
* ~~The wrapper's failure hint (`runwrap.py`) still says *"SpinPolarized with
  Spin.Total unset or 0 on a d/f-shell metal also triggers IMAX=0"*~~ — **fixed
  2026-09-29** (P2): the retracted cause is gone (`runwrap.py:3289–3305`).
* ~~The transport tab's shared panel is not persisted~~ — **fixed 2026-10-11**
  (step 8): the shared panel and the rung tabs keep what they hold in one
  workspace note, `transport:form`.
* ~~The PySCF vibration record keeps the raw `net_charge`~~ — **fixed**: its
  `config` carries the resolved `electronic_state` block.
* **Left by the review (2026-09-29):**
  * **The state is resolved several times per prep** — 11 call sites of
    `electronic_state()` (2026-10-09) (`electronic_state.py:160–252`): the family, the
    SIESTA checks, the vibration record check, the deck writer and prep's
    sibling files. Each gets the same answer from the same function; resolving
    once in `validate()` and handing it down is a tidy-up, not a defect.
  * **A re-prepped stage folder keeps its `makov_payne_correction.py`**
    (`runfiles.py:838`) from an earlier charged prep after the charge goes
    to 0. Prep writes siblings; it does not remove the ones a deck no longer
    promises. Not reproduced on 2026-10-08.
  * **`restart = continue` keeps the run id across a template edit.** So a spin
    edited between continuations starts from the previous state's `.DM` — the
    hazard ES1 closes for stages, one level up
    (`execution/run-identity.md` § 2.1). **Check against § 5x's ruling**
    before keeping: M2h (the stage hand-over) is done 2026-10-05, and § 5x.3
    rules *every launch a new run; warm or cold; every run kept* — **closed
    2026-10-09** under the user's rule of 2026-10-07: *"run continue warm or
    cold is user's decision, and error or not, that's user's responsibility"*.


## 5t. The run record — scope and order of work *(W35, 2026-09-26)*

*The contract is [`model/parse.md`](?doc=model/parse.md) § 5d and
[`engines/transport.md`](?doc=engines/transport.md) § 2a.12; this is the order
of work. The row is **W35** in § 2 (M3 carries P2's remainder and P3–P6).
Re-read against the code 2026-10-08: what was built or overtaken is in the
archive; what stands here is open.*

### 5t.1 The design

`model/parse.md` § 5d: one record per attempt, four parts (computation,
setup, deck, verdict), composed on read;
three columns for every parameter; one grammar per engine family; the verdict
and its symptoms. `engines/transport.md` § 2a.12: the transport report from
the rung records, and the stated contour.

### 5t.2 Decisions *(user, 2026-09-26: "get the framework and code unified and finalized")*

Taken as a yes to the three recommendations made the same day, with the
user's own additions — what is still open of them:

* **2.** **The contour is stated**: a circle and a tail whose lower bound sits
  below the seed's lowest eigenvalue — the manual's rule — rather than a
  pole count guessed from one measurement. Open — M3 P4.
* **3.** **On divergence the monitor warns, and the wrapper does not warm-retry**;
  stopping stays the person's call. Open — M3 P4: the wrapper's warm retries
  (`runwrap.py:3343-3349`, any `scf_not_conv` stop) ask no divergence
  verdict, and none exists; `running-a-job.md` marks the rule not built
  (2026-10-09), `model/parse.md:1706-1708` to re-check.
* **4.** **The parameters are reported as optimization's are — and better**: with
  what the engine used, and with the deck as it ran (user). Built
  (`parse/dirs/setup.py:151-205`, `siesta_fdflog.py:126-135`,
  `parse/dirs/record.py:467`) but for the setup rows' `echo` column —
  `parse.md:1538` lists it, `setup.py` writes none. P2.

And the same day, after P1 and its review:

* **5.** **P2 is built with all eight of the review's corrections** (§ 5t.5; user:
  *"go with all eight"*) — 4 (the echo column) and 6 (P4's symptoms) remain.
* **7.** **The transport result is shown, not listed** (user: *"plots that show
  the convergence of the calculation and … the DOS … in a more graphical
  way"*): each rung's convergence, and T(E), the DOS and the eigenchannels
  as plots. TBtrans's outputs are on by default
  (`data/catalogue.template.toml:2301-2347`); T(E), the DOS and the I–V have
  their card (`lib/inspectors/transport.js:142`); the eigenchannels and each
  rung's convergence plot are M5 step 9 / § 5x B6.
* **9.** **Then the whole road in the browser** (user: *"an end-to-end test
  through the browser … all based on the web UI design and the correct
  contract … and inspect the graphical presentation of the results and the
  graphical placement of the elements in the settings of the jobs … using
  the correct CSS framework"*): P6 — W50 step 3 (§ 5w.4).

### 5t.3 Phases, each with its done-condition

| | work | done when |
|---|---|---|
| **P2** | **the record** — built: the Run panel (`web/results.md:646`, § 3a), the SCF plots by phase (`scfplot.js:89-108`), the parameters fence narrowed to the calculation's items (`parse/dirs/setup.py:151-153` → `script_emit.declarations:796-802`). **Owed**: the setup rows' `echo` column (§ 5d.3; `parse.md:1538` lists it, `setup.py:151-205` writes none); the launch rows asked against those run — `compare_asked_to_ran` (`bench/result.py:191`) is called by the bench alone (`jobset/summarize.py:178`) | on the dev server, an optimization, a vibration and each transport rung show their Run panel; the SCF plots draw each phase against its own criterion |
| **P3** | **the transport report** — **done** (§ 5x B6, M5 step 9; the record reads both spin channels and both `.TBT.nc` files, and the panels draw each spin and their total — K21, 2.4.0). What § 5t.0 measured and still holds goes with it: `energies_relative_to_ef` hard-coded (`transport/record.py:537`); the spin channels' files missed — `.TBT.AVTRANS_*` globbed alone (`:453`) | B6's and step 9's own |
| **P4** | **the contour, the divergence, and the monitor's warnings**: the device deck states its contour from the seed's eigenvalues; the settings gate refuses one that cannot cover the spectrum; no warm retry of a diverged run; the symptoms, which the monitor then warns on. *(One monitor for every engine, reading through the framework's shipped readers and reporting the run's state, is decision 10 — built; the archive.)* **Why** (measured 2026-09-26): no `contour.eq`, so TranSIESTA's 42-pole continued fraction — dQ −29 at step 1 and −584 at the cap; 123 poles on the same deck conserved the charge to 0.024 by step 4; the wrapper then warm-retried the diverged device from its own density for three more hours. **Doc drift**: `execution/running-a-job.md:481-484` and `model/parse.md:1706-1708` state the no-retry rule as built; `runwrap.py:3335,3367` retries with no divergence verdict to ask | a device deck carries its contour; a diverging output yields the symptom verdict, and the monitor warns on it |
| **P5** | **the document sweep** — left: the Makov-Payne script (§ 5t.4) and § 2's document rows | the review finds none |
| **P6** | **the whole road in the browser** (decision 9; W50 step 3, § 5w.4): an optimization, a vibration and a transport calculation, each from the UI through the hand-over, Task setup, the printed `prep` / `launch` verbs and the Results tab — every step through the designed doors, no script; the settings' layout and the results' plots inspected against `web/ui-contract.md` and the CSS framework | each road finishes in the browser, and the inspection's findings are fixed at their owners |

### 5t.4 Found on the way — named here, fixed where a phase says so

* **A TranSIESTA device's `max_force` is meaningless** — TranSIESTA prints
  *"TranSiesta will NOT update forces … ALL FORCES AFTER TRANSIESTA HAS RUN
  ARE WRONG"* — and `report_fields.py:58-61` offers it to every run, its
  comment saying a transport rung's single point states one. One line under
  § 5x B6.
* ~~A PySCF run's manifest promises the monitor, `util.csv` and SCF-timing
  files, which the wrapper never writes for PySCF (`runfiles.py`).~~ The
  monitor's two files (`.monitor.log`, `.util.csv`) are every engine's since
  decision 10 — the wrapper starts the monitor from its shared part; the
  SCF-timing tee alone is SIESTA's, and its row names the engine
  (`runfiles.py:1017-1026`). This row had said all three rows were SIESTA's.
* The SIESTA vibration record's `runtime_info` is empty, so its Host/CPU/GPU
  rows read *—* (`spectra/siesta_vibration.py:245` takes only `siesta_build`
  from the reading; `spectra/results.py:696`) — unverified 2026-10-08. P2.
* **The Makov-Payne post-process script** (`siesta/makov_payne.py`, written
  beside a charged deck) keeps its own `.out` regexes for `E_KS` and the
  cell (`:274`, `:301`), and defaults to `<label>.out` (`:367`) — a name no
  wrapper writes — before globbing `*-run*.out` for the newest (`:381`). A
  second reader of the SIESTA family and a guessed name; it should read the
  run the way the monitor does. P5.
* **Found reading the Results tab's code (2026-09-27)**:
  ~~loading one run parses its output three times~~ — the directory's
  metadata was composed twice per load, each composing the relaxation
  record by a full parse, and a stopped run's poll re-read the file for
  its stop reason; now the load's own parse serves all three (2026-09-27) —
  unverified 2026-10-08 (`_run_ending.py:338-371`).
* The runaway symptom (§ 5d.6) cannot be "|ts-Vha| exceeds 1 eV": the
  converging device swung −1.39 → +1.95 eV in its first two NEGF iterations.
  P4 sets the rule on the two measured runs. M3 P4.

### 5t.5 P2's design, corrected by the 2026-09-26 review — **decided: all eight** *(user, 2026-09-26)*

The review read the P2 drafts against every door the record would use and
the files real attempts leave. Five of its corrections are built and one
overtaken (2: the record is read through `runs.folder_answer`,
`runs.py:429`) — the archive; what remains:

* **4.** **The columns from the run itself.** *Default* and *asked* as the run
  recorded them — PySCF's fence; SIESTA's wrapper fence extended to list
  every item's default — not today's catalogue. *Used* never picks one of
  several readings, and shows the engine's own echo where one exists
  (`Number of poles = 42`). Rows by engine, calculation and stage; items
  sharing a block as one row. Owed: the echo column (`parse.md:1538`;
  `parse/dirs/setup.py:151-205`).
* **6.** **Symptoms are P4's**; P2 carries asked ≠ used only. Open with P4.


## 5u. Transport — the one list (M5, W43) *(consolidated 2026-09-29)* — DONE 2026-10-11 but for S13 (the convergence sweep, step 6) and E13 (the junction on Sol, the user's)

*(user, 2026-09-29: "after all work on spectrum is done and review is green,
continue to work on transport as the next item on plan remember to correctly
update the plan and consolidate")*

**What this section is now** *(trimmed 2026-10-08; re-checked 2026-10-09)*.
§ 5x — archived 2026-10-09 — took steps 1–6 and 9; what stays here: step 5's
last line, steps 7, 8 and 10 (the queue's Q5), the TD1–TD14 decision record;
step 11 is Q17 (§ 5z). The paragraph as it stood: **§ 5x holds the order for transport since 2026-10-08** — the sweep,
the record and the Results tab (B5, B6), the small composition rules (B7), the
milestone review (B8) and the document sweep (B9). What stays here: the steps
§ 5x did not take over — 5's remainder, 7, 8, 10, 11 — the two owed contract
texts, the TD1–TD14 decision record (cited across the plan), and the stale
text § 5x B9 takes as its transport input (§ 5u.3, § 5u.4). Steps 1–4 are
done and archived (`4c612262`, `b63f219d`, `325cb1d4`; step 4's ladder
`claude-au-bdt-au` is no longer on disk — § 5x.0's road junction is the
acceptance now); step 6 is superseded by § 5x B1. The rows this section
absorbed point here (§ 5u.5).

### 5u.0 What is true today — ARCHIVED 2026-10-08

*Superseded by § 5x: the road is the ordinary one and the three cases ran on the road junction (§ 5x.7's record, archived).*

### 5u.1 The order — each step closes as a milestone does (§ 0a)

| # | step | absorbs | done when |
|---|---|---|---|
| **5** | **Submission groups — one queue wait per rung, not per point** — TD2's ruling; its order against M2 is TD9. **The group built 2026-10-07** *(user: "group at prep")*: `prep task --stage seed --stage electrode_L --stage electrode_R` names a group — each member prepped as alone, one save, none building on another (refused by name), one shared allocation; `Job.group`; the group's header at prep; `launch task` by the same names sends one job walking them (`project-layout.md` § 1.6.6; `jobset/group.py`). Walked on the road junction (§ 5x.7: seed + leads, one group job). The bias-point walk is § 5x B4's one walker (`submit._walk_script`). **Left**: the group's header checked on a machine with a scheduler (Q13). The frame walk built 2026-10-10 (Q17-d): frames are points of one run (D4), walked as voltages are, a chain per frame (`transport.md` § 2a.11) | TR9; § 2a.7's *default grouping*; W32 ③'s *the gather runs per frame* | the frame walk on the group: the device and the transmission ONE submission per scan walking frames, each from the shared seed, continuing past a failed frame; members stay prepared runs with their own records; the allocation sized from the members' own requests, a member that cannot share it refused by name; the order and the stop-or-continue rule read from the ladder's own `stage_inputs` and the axis's declaration. Written into the contract first — `project-layout.md` § 1.6.6 and `task-setup.md` § 1 carry the group (2026-10-07); the frame walk is written in `transport.md` § 2a.11 and `job-system.md`. The group's header seen on a scheduler |
| **7** | **The rest of the TranSIESTA / TBtrans surface**, each through step 1's sections and checked in the engine's source. **Done 2026-10-11, at 2.1.0** (`6c518057`): the four new rows (`TS.SolutionMethod`, `TBT.SolutionMethod`, `TS.Hartree.Fix`, `TS.Contours.nEq.Fermi.Cutoff` with `at_least`), blank ≠ 0 on the two device broadenings with `written_by` and `jobset migrate` for the old zeros (user, 2026-10-11), TD11, A1.12's counting half, the `bloch`, reservoir-temperature and tbtrans-log notes (`transport.md` § 2a.13, `template.md` §§ 3, 5, 5.3, `changelog.md` 2.1.0). Two review rounds, findings fixed (a rung's old 0 in `task.json` refused and migrated too). **Open**: the spin-polarised device run (`TBT.Spin`, K21's two channels). **Tried 2026-10-11** on the user's word, the task made through the browser (Transport tab → Task setup → the printed prep/launch, `projects/claude-transport-walk/transport/h-spin`: the relaxed H-chain pair, unrestricted, the moment free, 0 V): the seed and both leads converged fully polarised (moment 22, and 9 per lead, from their first step); the device's NEGF cycle held at dQ = 2.3 and a moment of 19 through 1000 iterations and its warm retry — `SCF_NOT_CONV`. **Finished procedurally** on the user's word (*"relax the convergence parameter, and let it finish just procedurally"*): the device's charge check is a setting now (`ts_scf_dq_converge` / `ts_scf_dq_tolerance`, 2.3.0 — SIESTA writes `.TS.HSX` only on convergence), set off with the cycle cap at 100 on the device's row in Task setup; the transmission ran and `summarize task` read **both channels' transmission** (`channels: up, down`, the current their sum). **Found**: the DOS, PDOS and eigenchannels are not read for a polarised run — tbtrans writes `TBT_UP.nc` / `TBT_DN.nc` and the record looks for `TBT.nc` alone (`dos_why`); how a spin's DOS is shown is a question for the user. The Fermi levels read `nan`: SIESTA printed an overflow for this state (`Fermi = 2178 eV`, `**********`), which the reader passes on. TD14 retired (below). **A known limit**: the template's `written_by` dates the whole calculation, so a calculation described before 2.1.0 whose template never needed migrating (both broadenings set to numbers) reads a 0 typed on a rung today as an old one — refused at prep, and `migrate` writes it blank and says so. **Re-checked 2026-10-09**: `TS.Elecs.Eta` has its row (the catalogue's `:2558`); no `TS.Contours.nEq.Fermi.Cutoff` or `TS.Hartree.Fix` row; `electrodes_bulk` declares `shared` and no `stages` (`[item.electrodes_bulk]` — TD11); the transmission's SCF settled 2026-10-09 (§ 2, TD12); the transverse check measures per lead, not per layer (`validation/__init__.py:516-543` — TD14). **Re-checked against SIESTA 5.4.2's source 2026-10-11**: `TS.SolutionMethod` and `TBT.SolutionMethod` default `BTD` (`m_ts_options.F90:180`, `m_tbt_options.F90:490`); `TS.Hartree.Fix` defaults to the transport axis's letter, `C`, and any other spelling — `-C` included — prints *"Requesting a different hartree plane than the default! … YOUR responsibility"* (`m_ts_hartree.F90:50-72`); `TS.Contours.nEq.Fermi.Cutoff` is in kT, default 5, and below 2.5 the run dies (`m_ts_contour_neq.f90:141-144`). **Taken off 2026-10-11** (said to the user): `TBT.T.Out` — tbtrans's default is `N_Elec == 1` (`m_tbt_options.F90:634`) and a transport run is two-terminal (`transport.md` § 4), so there is nothing to state; a per-lead temperature (`TBT.ChemPot.<>.ElectronicTemperature`) — both leads take the one shared electronic temperature, and a per-lead one is a thermoelectric feature nobody designed; the chemical-potential block says so in a comment. TD13 retired (§ 5u.2) | W25; § 5o.6's open rows; TD11, TD12, TD14 | `TS.Elecs.Eta` has a row (today TranSIESTA's 1 meV applies unsaid); `TS.Contours.nEq.Fermi.Cutoff` has a row, 5 kT, refused below 2.5; `TS.Hartree.Fix C` written, fixed by the rung, never typed; the transmission header saying which program reads what and pointing to `tbtrans`'s own fdf log (TD12 — nothing is cut); `electrodes_bulk` carried by the two rungs that read it — `shared` with `stages` meaning one value for the named rungs (TD11); ~~the transverse check measuring each atomic layer of the lead (TD14)~~ — retired 2026-10-11; a spin-polarised device run end to end (`TBT.Spin`; its record reads both channels' `.TBT.nc` — K21 — and the run is e2e, on the user's word); the chemical-potential block saying both leads take the shared temperature; A1.12's counting half (the element, not the raw label — the warning itself stays as written, F9 saying it once per group); both solvers (`TS.SolutionMethod`, `TBT.SolutionMethod`, `BTD`, today unsaid) and `bloch 1 1 1` with its note stated in the deck — TD1's *every parameter explicit*, after step 10 so each new line carries its source |
| **8** | **One panel per engine** (step 3, which it waited on, done 2026-10-02). **Done 2026-10-11**: card 4's engine strip — TranSIESTA live, PySCF-NEGF disabled with its reason — from `transport.stages.engine_tabs`; the schema route and the describe door take the engine and refuse one that runs no transport (`transport.md` § 3.8.8). The shared panel kept across a reload too (the user, 2026-10-11: *"you should fix #2 … a framework, like using the correct prefix/tag when calling persistence module"*): what the shared panel and each rung's tab hold is one workspace note under `transport:form`, through the workspace's one door (`web/tabs.md`, `workspace.md` § 4) | W24 = W30 ⑤ = `engines/transport.md` § 3.8.8; § 5s.4's *the transport shared panel is not persisted* | the tab's engine strip with the disabled PySCF-NEGF tab, its fields from the catalogue (§ 3.8.8's *"`TransportConfig` keeps its name"* superseded by TD4); an engine dimension on the schema route (a PySCF-NEGF backend itself — its `spec_for` arm, catalogue rows and config — is `transport.md` § 8's follow-up, not this step) |
| **9** | **The Results surface, finished** — **→ § 5x B6.** Designed 2026-10-07, `web/results.md` § 2.5 (:440-463), approved (user: "yes, go ahead with the build"); ① record fields (DOS, PDOS by label, eigenchannels, per-rung SCF, NEGF figures, chain, caveat) + the shared SCF-plot module and ② the report's cards, tabs, two selections, PDOS of a selection by atoms and orbital type — **built**; ③ the frame axis waits on step 11; the rest is § 5x B6 (R1–R4, R6, R7, R9, the low-bias I(V)) | W10; § 5c.3 (e) and (f)'s rung half — (f)'s citation half is built (`transport.js` draws `provenance.slot`); TR10's chain; W30 ④ (the deck viewer) | § 5x B6's: `summarize task` and the Results tab on the road junction, the three cases. Named here from the first done-when, for B6 to carry: presenters read through a door rather than `JSON.parse` (a framework door, not transport's alone); the decks viewable per rung, run and bias point beside their `.validation.txt` (W30 ④) |
| **10** | **Every value says where it came from** — TD5's ruling, framework-wide. **Done 2026-10-11** (`bac8ccbe`, `2c951f48`): each deck's PROVENANCE block says who set each value it is rendered from (`parameters:`, `ResolvedConfig.deck_sources`, both render roads; `job-contracts.md` § 3.2); a setting left to the engine says so where its line would be, in the item's own `not_set` — the catalogue key the user named (*"unset may suggest removed or cleared by user"*), required on every optional engine/deck item (`template.md` § 5, § 6.6); Task setup's *What came over* lists every value shared by every stage with whose it is (`task-setup.md` § 3). Reviewed in two rounds by fresh agents, every finding re-read against the code and fixed (`048824a7`, and the commit after it): the table names the step that set a value after `resolve` and who answers a blank; both PySCF decks say their unset solvent and fitting basis; the diagonaliser and output items declare the SCF rungs, so the transmission's table and tab carry none of them | W30 ③'s remainder; `engines/template.md` § 6.6 obligations 2–4; the read-only echo of shared values | a template item carries `source` (`cited` · `person` · `default` — `record` retired, `template.py:1168-1172`; built, K7), written at `init` and by the describe door, read by every surface; each deck's provenance block carries the per-parameter table; an `optional` item left unset writes a line naming the engine default that applies |
| **11** | **The frame axis → Q17 (§ 5z)** — on step 5's group (its *Left*). **Done 2026-10-10** (Q17, archived) | W32 ②, ③, ⑤ (④ left transport by TD10's ruling: transport is handed a multi-frame structure and reads each frame's details from its `customized` section; the generator is V1.25's own procedure) | as W32 states them, the frame walk on step 5's groups; step 9 ③ with it |
| **12** | **A transport description's life** *(user, 2026-10-11, on the AuBDTAu walk; design B of the framework analysis — who writes which part, when)*: ① Describe is one act and the server writes it (`/api/transport/describe` writes the template and `task.json` into a folder holding none, through `describe.write_description` with `jobset init`'s two refusals) — `engines/transport.md` § 3.8.2a, `web/handover-procedure.md` § 4, § 6; ② once written, Task setup is its editor as every kind's: the Stages card's rows are the ladder's rungs, fixed and numbered — only the seed removed or added back (ruling Q4) — the picker naming the rungs that read each item, the shape shown fixed — what a kind fixes is `task.fixed_by_kind`, which the reader enforces (`web/task-setup.md` § 4, § 5.3); ③ the launch rows stand on every rung's run card, blank — the catalogue's `run_card` (§ 6.2b); ④ ~~the junction card's centring offer~~ — built at 2.5.0, retired at 2.6.0: the junction enters the calculation centred (row 13 ①); ⑤ the swap choice reaches the description (the sender never sent it) | — | built and reviewed twice; `AuBDTAu/transport/transport_lowbias`'s junction composes centred |
| **13** | **Found on the 2026-10-11 browser walk — the user's word first** (the road ran: Describe wrote the description through the server, Task setup echoed it, Preview refused until the launch rows were stated, then composed the AuBDTAu junction centred): ① ~~D7~~ — **ruled 2026-10-11, built at 2.6.0**: a structure entering a new calculation is centred unless its origin is a person's; a vibration's input stated relaxed keeps the engine's frame (`structure-periodicity.md` § 6.0, *Entering a new calculation*; the sidecar records whose the origin is, v12); D7, lost in `d7f37054` with no decision, is subsumed and 2.5.0's `centre_junction` retired; ② **closed 2026-10-13 (2.9.1): `stages = "scf"` on both, `template.fixed_answers` for the one-answer rule** — the transmission tab echoed `WriteForces` / `WriteCoorStep` as fixed on, though tbtrans writes neither — § 6.1b keeps their one answer for every rung on purpose; limit them to the SIESTA rungs (`stages`), or keep; ③ **closed 2026-10-13 (2.9.1): the locked box shows the list's points (`locked.per_point`, `formSchema.showPoints`)** — the device's and transmission's *Bias voltage* echo was an empty box — show the bias list's points; ④ **closed 2026-10-13 (2.9.1): counted against the heavy non-metals, none on a gold lead** — the H/heavy reference note counted gold as heavy — 0/75 on every lead; ⑤ **closed 2026-10-13 (2.9.1): it names Task setup's card, then the key and the flag** — the run card's refusal named `task.json` and flags, not the card on the page; ⑥ ~~at ~1200 px the Transport tab's main column is narrow: the rung strip overflows and the run card's number boxes show a dash~~ — **fixed 2026-10-11** (second walk): the `.tabs` strip wraps (tabs 4–5 were clipped out of reach), and the run card's boxes are typed text inputs, so the theme styles them. **The second walk, end to end on `claude-transport-walk/transport/e2e-walk{,2}`** (Describe → refused again → Task setup: stages edited, seed removed and added back, Save, run cards, Preview, Prep → Results), fixed: ⑦ the machine card and `STAGE-PLAN.md` listed `molbuilder.json`'s `launch._comment` as a value it supplied (`config_provenance`); ⑧ a Prep's answer stopped before the terminal's *next* line, and every rung's tab named its stage alone, so a group's one launch was shown nowhere — the answer carries `next`, and the tabs ask the status door's rule (`runstatus.launch_words`, D4); ⑨ a waiting rung read *"…from seed, whose newest run, 01_seed/run-0, which has not been launched"*; ⑩ the Transport tab drew all five rung forms three times on load — restoring the shared panel fired its change listener; ⑪ the describe refusals named the folder by its absolute path. **Open, the user's word first**: ⑫ **closed 2026-10-13 (2.8.0): a blank transmission mesh is the rule's, and its hint reads *not chosen · (3 x the shared k-point mesh)*** — the transmission grid's one-line hint read *"not chosen · 15, 15, 1 you set this"*: the value is the 3× rule's (`kmesh.transmission_start`) on the shared mesh the person set, and the line says neither the rule nor its purpose, which only ⓘ help gives (§ 0.3b) — proposed: *"3 × the shared k-point mesh (5, 5): T(E) needs a finer in-plane mesh than the density; converge until T(E_F) moves < 1 %"*. Traced 2026-10-11: the cited AuBDTAu run is 3 × 3 × 1, offset 0; the user's `transport_lowbias` set 5 × 5 × 1 (→ 15 × 15 × 1); the 4 × 4 seen on the walk was the walk's own test edit, since undone in the user's note. Parity and Γ, from SIESTA 5.4.2's source: the block form reaches `find_kgrid` with `firm_displ = .true.` (`kpoint_t.F90` `setup_mp`; tbtrans's `m_tbt_kpoint.F90` likewise), so the 0.0 offset is kept, and the index runs −(n−1)/2 … n/2 — Γ is a sample on an odd or an even count; an even count adds the zone edge. The 3× rule keeps parity (5 → 15, 4 → 12). A parity rule, if wanted, is chosen with the literature first; ⑬ ~~GPU binding on a machine with no queue~~ — **withdrawn**: the box is the calculation's own switch (`execution/gpu.md` G9, the person's column of § 1), and the queue card writes `allocation` whole (`web/task-setup.md` § 6); a machine fact decides nothing about it; ⑭ **moot 2026-10-13: the seed is always kept** — without the seed the stage folders renumbered (`01_electrode_L/`) while the rows read *2 · electrode_L*; ⑮ **closed 2026-10-13 (2.9.1): MolView's hidden radios positioned in their labels — they had stretched the document** — a focus jump (find-in-page, `scrollIntoView`) scrolled the window itself, leaving the header off-screen and a blank band below — `body`'s `overflow: hidden` reaches the viewport, which can still be scrolled by script | — | each ruled; the ruled ones built |
| **14** | **Groups are stages' numbers; prep writes every group; launch gathers** *(user, 2026-10-11, after the browser walk found the page treating a group's one job as three stages: "the prep UI clearly still treats each stage separately ... steps that are beyond logic"; "the checkbox for grouping is awkward: it cannot assign different groups"; "prep should be able to prepare all groups at the same time ... that's the launch status, not the prep")*. **Ruled 2026-10-11, the four questions as put**: ① each stage states a **group number** in `task.json` (`stages[i].group`), edited on Task setup's Stages card: the first stage is group 1, each later one repeats the number above it or adds one; a stage may not repeat the number of a group holding a stage it builds on; the defaults repeat where allowed (transport 1 1 1 2 3); ② one tab per group on Task setup, holding the group's one run card — what the job shares (ranks, cores per rank, GPU use and count, queue) written to every member's `execution`; the wall per stage, the job asking their sum; ③ **one prep prepares every group** — each group one job — and **the gather moves to launch**: `launch task` sends the next group, first checking each stage's upstream newest run finished and ran the deck described now, copying the inputs with their `.gathered-from`, refusing before anything is sent when not; ④ a stage **prepared and never launched may be prepared again** (no run's record to falsify). **The design, traced and reviewed: § 5aa** — V settled (the force-constant stage keeps its geometry in its deck), R answered by a save before a group's first launch (put to the user with the contract). Contract first: `job-system.md` (*The task*, the states, § 5.0, § 5.4, § 6.0), `project-layout.md` § 1.6.6, `stages.md` (the field), `transport.md` § 1.1, `task-setup.md` (§ 5.3, § 11, the run card), `web-api.md`; then code; then the user's `lowbias` setup proved in the browser (this machine; a Sol preview) | the walk's grouped-prep refusal on `AuBDTAu/transport/lowbias` (2026-10-11: the job's resources stated on the seed only) | proved through the browser on the user's setup, two review rounds — **done 2026-10-12** (2.7.0; § 5aa, archived) |

**TD12 — settled 2026-10-09** (§ 2, TD12): `engines/transport.md` § 6.1b states what tbtrans reads and the deck's header says so; the four SCF rungs carry their own SCF settings and the transmission none, its tab saying why (`RUNG_NOTES`), the user's word on B8 U1's cut.

**Owed by TD10's ruling** — **written** (`36a485e6`; 2026-10-09; `archive/2026-10-10-q17.md`, § 5z.2). The paragraph as it stood *(2026-09-29; open 2026-10-08 — `transport.md` names no `customized`)*: the contract text — `engines/transport.md` § 2a.9 (the frame axis: handed a multi-frame structure, each frame's details in `customized`), `model/structure-molstruct.md` § 6.1, `engines/vibration.md` § 5.6 and § 5.10 ③, `science/normal-modes.md` § 4b.6 G (the generator a separate procedure; its record in `customized`, not `info`) — written once the M11 reviews, which are reading those documents, report.

**Beside every step: the documents.** Each step fixes the false claims and
stale citations in the documents it touches; the sweep at the end is **§ 5x
B9**, whose transport input is § 5u.3's residue and § 5u.4.

### 5u.2 Rulings *(user, 2026-09-29, on the questions as they were put)* — the decision record; every one ruled, TD6's decision now due

*(Numbered TD — transport decisions — because D1–D7 already name § 2 rows. The
questions as put, and the user's words, are in the archive.)*

* **TD1** — the device's equilibrium contour (TranSIESTA's fallback 42 points lost the charge on `claude-w33`'s device; `TS.Contours.Eq.Pole 10 eV` = 123 points held it): **10 eV, always written, the count stated beside it** (an ENERGY — TranSIESTA counts its points by how far up the imaginary axis they reach, `N = int(E / (π·k_B·T))`); interim until M3 P4's stated `contour.eq` (§ 5x's parked P3). Step 2 (done); the rule is `transport.md` § 6.1c.
* **TD2** — grouping the ladder's submissions (TR9): *"not a hack but a sequence-aware design framework"* — **build it, on the chain submission**. Step 5.
* **TD3** — vacuum in a transport structure: refused along transport beyond one layer spacing of the lead (measured from the lead) and on a transverse axis declared periodic; **allowed** on an isolated transverse axis, from the structure's stated vacuum, and so is the box built from it; the k-grid along transport 1 on the device and the seed, dense on the electrode rungs. Step 2 (done); § 6.1c.
* **TD4** — `TransportConfig`: **retired** — the Results tab reads the record and the ladder's status, never the class. Step 3 (done).
* **TD5** — how a value records where it came from (`engines/template.md` § 6.6's three proposals): **all three, as proposed**. Step 10.
* **TD6** — per-rung convergence defaults (§ 2a.7): **decided after the first ladder**; every rung's convergence settings stay on its own tab, the person's to change. After step 4 — **due**: the ladders have run (§ 2, TD6).
* **TD7** — the bias treatment as a named choice: **build it** — superseded in form by § 5x decision 1 (one explicit switch, `low_bias_approximation`; B1, `7351a3ff`).
* **TD8** — one acceptance ladder *(user: "TD8 yes")*: `claude-au-bdt-au` closed step 4, E13's workstation half, W33 P5 and W35 P3's done-when; `claude-w33` and `claude-vib-ui`'s fake junction retired as acceptance targets. **2026-10-08: `projects/claude-au-bdt-au` is no longer on disk; § 5x.0's road junction (`claude-transport-walk/transport/au-dta-t`) is the acceptance now** — flagged to the user.
* **TD9** — step 5 against M2 (which attempt *concluded* — M2f; the bias chain's restart list — M2g; a seed skipped by removal — M2i; one placement record — M2k): **decided 2026-10-02** *(user: "#9 ok")* **and done** (Q4: M2f, M2g, M2i done; M2k's placement done, its GPU half M2k's own; step 5 built `0e11a018`). Recommended then: steps 1–4, then M2f, M2g, M2i, M2k, then step 5 onward. On 2026-10-08: M2f, M2g, M2i done; M2k partly; step 5's group built. Step 5.
* **TD10** — the displacement generator's one home: **transport builds no displacement.** It is handed a multi-frame structure (one pair, many frames); the person is responsible for the frames being one structure with only the relevant atoms moved; the structure's `customized` section (W39) carries each frame's parameter set — its displacement, the mode's frequency, whatever else tells that frame's details. The generator is a separate backend procedure (V1.25): the equilibrium relaxed structure and a normal mode in, the multi-frame structure out. W32 loses ④; W42 ③ uses the procedure. Step 11 · V1.25 · W39; the contract text owed above.
* **TD11** — `electrodes_bulk`'s scope *(user: "TD11 a")*: `shared` and `stages` declared together mean **one value, carried by the named rungs only**; `electrodes_bulk` declares both, and the shared panel says which rungs read it. Step 7.
* **TD12** — what the transmission deck carries *(user: "a comment or log showing what are the parameter taken by tbtrans would be enough … no point adding complexity between steps")*: **nothing is cut.** The header says which program reads what — `tbtrans` reads the `TBT.*` settings, the `TS.*` declarations and, by fallback, `ElectronicTemperature`; the SCF and output settings are the ladder's shared set it does not read — and points to `tbtrans`'s own fdf log in the run (`fdf.<time>.log`), the engine's word rather than ours. Step 7 (the header only). *B8 U1 then cut the SCF items from the transmission; the user settled it 2026-10-09 — the four SCF rungs carry their own, the transmission none, with its tab saying why (§ 2, TD12).*
* **TD13** — correcting a junction's transverse kinds without a new relaxation — **retired 2026-10-11** *(the user: "Retire TD13"; ruled "TD13 a" on 2026-09-29, a transport-side declaration of the transverse kinds)*: the code builds no recovery machinery for a wrong state (2026-10-07), so the kinds are what was cited — the relaxation's deck, or a pair's own. A wrong declaration is refused at the first prep, and the way out is the road that exists: relax again after declaring on the Cell page, or save the run as a pair, set the cell on the Cell page and cite the pair — the edit marks the pair's record outdated and keeps it, and the citation reads its settings (`transport.md` § 6.1c).
* ~~**TD14** — the transverse check measuring within each atomic layer of the lead~~ — **retired 2026-10-11** *(user: "i would leave scientific check to the user. our check should really be the most obvious, and consistency check. we are creating tools, not to invent or limit scientific exploration.")*: whether a lead's cell tiles is the person's to judge. Measured before building, the per-layer rule would have refused correct leads — a graphene sheet with transport along zigzag 2.0, a bcc(111) lead 1.63 against the 1.5 limit. The whole-lead check stays as it is: vacuum on an axis declared periodic, the structure against its own declaration (`transport.md` § 6.1c).

### 5u.3 / 5u.4 / 5u.5 — ARCHIVED 2026-10-08 (B9 done)

*The residue list, the stale citations and the absorption map are in the archive; B9 swept them (the commit after `93977f5d`). What B9 did not reach is a § 2 row.*

---


## 5aa. Groups, every group prepared at once, a run's inputs taken at launch — ARCHIVED 2026-10-13

Built from 2.7.0: a stage's group number in `task.json`, prep preparing every
group picked in one plan, a group's inputs gathered at its first launch, one
hand-over, and review rounds A and B. The design, its trace, its reviews and
its status are verbatim in
[`archive/2026-10-13-5aa-groups.md`](?doc=archive/2026-10-13-5aa-groups.md);
the rules live in `execution/job-system.md`, *The task*, and
`web/task-setup.md` §§ 5, 11. Nothing of it is open: the engine e2e batch
passed on the grouped code (302 tests, 2026-10-13), and two small code items
it named are in Q19's Python list, with its design calls for the UI, the CSS
and the Python.

## 5z. Q17 — transport over a set of frames — ARCHIVED 2026-10-10

*Done 2026-10-10 (Q17-c … Q17-f, each with its real-junction e2e — 41 passed
at the close; the milestone review's two rounds; an H2 frame set walked
through the road in Chrome). Its rulings, its build history and its design
(A–E: the parameters, the cycle, the folders, the launch) are in
[`archive/2026-10-10-q17.md`](?doc=archive/2026-10-10-q17.md). The rules it
wrote live in the contracts: `engines/transport.md` § 2a.9 (the input — a run
or a pair, one frame or a frame set), § 2a.11 (a point is a frame and a
voltage), § 2a.12 (the record, the data file and its notebook recipe), § 3.1
(what you can cite, *a cited pair is left alone*, *a calculation's folder is
its own*), § 3.8.1 (the frame-set card); `model/structure.md` § 2.2d–2.2f
(frames, `customized`, a mode's frame set's definition); `model/overview.md`
§ 2 (the index convention); `web/results.md` § 2.4–2.5 (the frame family, one
Plotly box); `web/task-setup.md` § 3. Open from it: the frame generator,
V1.25 (§ 2), its amplitude waiting on the user; the AuBDTAu workflow, the
user's, on Sol (the handover).*

---

## 5w. The M11 review — what is left of its mechanism classes *(W50, 2026-09-29; rewritten 2026-10-09)*

*The five static reviews of 2026-09-29 (Structure optimization on SIESTA /
PySCF, Spectrum on SIESTA / PySCF, Transport), each defect re-read in the code
and the engine's source, are
[`archive/2026-09-29-m11-static-review.md`](?doc=archive/2026-09-29-m11-static-review.md);
the classes K1–K22 and their rulings as they stood, with the 2026-10-09
verdicts, are in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
§ 5w. **The fix is the framework's, never the call site's** *(user,
2026-09-29: "the solution to the issues identified through the review should
be a systematic holistic and framework level with good api unification")*:
each class closes with one declaration in the catalogue and one door every
surface asks (`template.md` § 6.5, § 6.6).*

**Done:** K1–K8, K10–K12, K19, K20 (archive 2026-10-08); **K15** — the deck's
values written to two significant figures by one formatter, `as_written`
(`pyscf/layout.py:131-138`; the user's ruling, `pyscf.md:701-704`, `164ecbfe`);
**K16's record** — `script_emit.declarations` (`:804-821`), read narrowed by
`parse/dirs/setup.py:169-170`; **K14's name map** — `AS_RESOURCE`
(`jobset/model.py:57-62`), read by `resolve.py:148-157`, `prep_inputs.py:365`,
`placement.py:63`; its blank budget superseded — an unstated value is refused
(`jobset/model.py:66-70`, `runwrap.py:3445-3447`, the 2026-10-02 rule).

### 5w.1 Open, in this order (step 2 of § 0a's order, but K17's three named items in step 1 and K22 in step 6)

| | what is left | where it is | done when |
|---|---|---|---|
| **K13** | **one atomic writer** — private writers remain at `checkpoint.py:664`, `sidecars/molstruct.py:549-581`, `sidecars/spectra.py:159-180` (V1.37), `workingcopy_structure.py:435`; 25 raw `write_text` calls (e.g. `compose.py:829`, `jobset/summarize.py:369`, `jobset/_cli.py:445`, `script_emit.py:1460`, `:1934`) | `persist.py:68,163` | every file molbuilder writes goes through `persist`'s atomic writer |
| **K14** | **the engine's parallel model as data** — today a suffix branch, `runwrap.py:4242` `ntasks = 1 if suffix == ".py"`, also `:3443` | `job-contracts.md` § 6.2 | the model (MPI or OpenMP) declared per engine and read by the scheduler header |
| **K16** | **the wrapper fence narrowed** — the SIESTA fence lists every SIESTA item (`runwrap.py:1746`), PySCF narrows by calculation only (`pyscf/layout.py:313-314`), against `parse.md` § 5d.3a's *engine, calculation and stage* (marked not built) | `model/parse.md` § 5d.3a | the fence writes the rung's own items |
| **K18** | **a PySCF run's level of theory** — `parse/contract.py:28-46` reads SIESTA decks only (`pyscf.md:135`) | `parse.contract.contract_of` | a PySCF deck's basis, method, functional, charge and spin read by the one reader; the Results export carries `info.calculation` for a PySCF run |
| **K17** | **physics needing a build or a refusal**: SO-C10/C11 — `ParallelOverK` automatic over k (`siesta/input.py:443-446`), ignoring ELPA, its help silent on the deviation, and counted before time reversal (`kmesh.py:111-114`); SO-C12 — `find_psml`'s case and prefix fallback (`siesta/input.py:222-235`) against `pseudopotentials.md` § 9.1's exact name; **PO-C4 — the geomeTRIC log**: § 5w.3 of 2026-09-29 dropped its promise, but `runfiles.py:994-1000`, `pyscf/input.py:49`, `:1518-1522`, `pyscf.md:137`, `project-layout.md:2683` and `vibration.md:2093` still promise `_geom.log`, and PySCF 2.14 substitutes its stream-only `log.ini` when `geometric/log.ini` is absent (`geometric_solver.py:147-150`) — confirmed on an H₂ PySCF run, then the promise dropped everywhere; PO-C10 — the ECP read back; PO-C12 — a `-V` functional with D3; PS-C13 — mode numbers bounded at prep (`validation/spectra.py:107-134`); T-F4/F34 — Task setup's add and rename on a transport ladder (`task-setup/viewer.js:1253-1266`); the vibration's level-of-theory check skipping the recorded k-mesh (`parse/contract.py:63-68`); PS-C3 and PS-C2's builds (their refusals are in: `validation/spectra.py:386-415`, `validation/chemistry.py:378-391`) | as listed | each built, refused, or its promise removed |
| **K22** | **per-atom starting spins** — **designed in step 6, brought with its science**: where the starting moments live (the structure's per-atom data, or a template item) and how SIESTA is told them (`DM.InitSpin`). Measured 2026-10-08: an unrestricted hydrogen junction cannot run — every rung's moment floats, the aligned start leaves no minority density, the empty channel's GGA potential puts its states near 3000 eV; the suite's transport run is restricted until this is built (`tests/test_transport_on_a_real_junction_e2e.py:26-32`) | no `DM.InitSpin` writer | designed, ruled, built, and a polarized junction run end to end |
| **K21** | ~~**a polarized junction's panels**~~ — **done 2026-10-11**: the polarized run `claude-transport-walk/transport/h-spin` made through the browser and run on the road; both `.nc` files read (`tbtnc.tbt_files`, `spin_dos`), the report drawing ↑, ↓ and their total by colour and legend in T(E), DOS, eigenchannels and the I–V *(user: "use color/legend to correctly present up and down spin and total current")* — `web/results.md` § 2.5 | `transport/tbtnc.py` · `transport/record.py` · `lib/inspectors/transport.js` | ✅ |
| **K9** | **engine facts restated in prose** — each item's engine default declared once with its source, the help's deviation sentence, the template comment and the deck's *not set* line written from it, checked against SIESTA's fdf log of a minimal run; not started (`template.py:150-290` has `manual`, no default-with-source field) | `engines/template.md` § 6.5–6.6 | the declaration, then the check with the e2e step (ruled 2026-09-29) |
| **K11** | its road test — the leads and the seed concluded, `mesh_cutoff` changed, the device's prep refused — is not in `tests/`; judged now that § 5x B2–B4 are done: kept as an assertion on M2n's real run, or retired | — | judged |
| **K3, parked** | `transport/sort.py:74` keeps its own `TRANSPORT_AXIS` (`kmesh.py:28-34` says owed) | — | one constant |
| **K10, parked** | SS-C14's deck half: the relax rung's start-state comment says one field decides, `restart` (`siesta/input.py:1277-1285`), but `restart` is optimization's only (`catalogue:653`); on the flat shape a vibration's relaxation re-run after its force constants starts from the FC run's last `.XV` — unmeasured | — | the comment true; the flat case measured |

### 5w.2 Then the text

The doc-drift and nit findings are one sweep, **after** K9 — several of those
texts become data there. The list (the archive's § 5w.2) is partly stale —
T-F21 is done (`validation/__init__.py:393-395`), PS-C5 has no *auto-load*
text left — and is re-derived before the sweep. Then W50's step 3: the road in
the browser per track, with the reviewers' probes among its runs.

---

## 5v. Documents that lag the code — ARCHIVED 2026-10-09

*In [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
— its entries corrected on 2026-10-09 (the transport invariant table's
`species_order` row, I15; I14's owed clause; § 2a.13's `TS.HS.Save`;
`template.md` § 6.6; `junction-cell.md` against X3; `normal-modes.md`'s
comparison tool; `vibration.md` § 10; `stages.md`'s conversion;
`execution/overview.md`'s fork; `parse.md`'s *one test*; the archived plans
named as *the plan*); what is left is § 2 (W15's #102 / #103, TPL,
`checkpointing.md` § 13.4 → W45).*

---

## 5x. Q14 — Transport, made whole — ARCHIVED 2026-10-09

*Done 2026-10-08 (B0–B9, the review's two rounds clean); in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
with its decisions (§ 5x.5) and its road walks (§ 5x.7) — the contracts it
wrote are `engines/transport.md` § 2a.10–2a.12, § 3.1 and § 3.8. Its open
remainder is § 2's F9, F11 and F14 (F19b done 2.9.6; R2-15 closed).*

---

## 5y. Q15 — No fake engine — ARCHIVED 2026-10-09

*Done 2026-10-09 (`22ac10d9`, `4963406b`); in
[`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md).
The rule lives in `process/testing.md` — tier 1 (`:28-40`: no engine runs
in the basic suite, fake or real) and its patterns (`:757-779`): an API test
checks what molbuilder prepares, a test that reads an engine's output is
end-to-end on the real engine, and one end-to-end pass makes each run once.*

---

## 7. The document survey and the four indexes — OPEN; step 1 done (§ 5v, archived), steps 2–4 not started

*(User, 2026-09-07: "the documents are very fragmented and a lot of key
information cannot be easily found. we also should have an index that
summarizes key design aspects, such as api, module, data structure etc. we
also should have a reference list that gives the foundation of all the
constants, scientific validation foundation and paper citations summarized in
one place.")*

### 7.1 What is actually wrong — measured 2026-09-07, re-measured 2026-10-08 and 2026-10-09

**Coverage too, since 2026-10-08.** The 2026-09-07 measurement said every
`/api/*` route was named in `web-api.md` and planned no sweep. The
route-catalogue sweep (§ 0a *Unscheduled*) overturned it: **11** documented
routes do not exist and **9** live ones are undocumented, against **86** unique `/api` routes live on 2026-10-09 —
70 `@bp.route`, 10 `@bp.get` / `@bp.post`, 6 on the app itself
(`app.py:641-701`, `rate_limit.py:565-571`); the 71 counted only the first kind. And coverage was never truthfulness:
the 2026-09-29 check read every live document against the code and found
statements the code contradicts — § 5v holds that list, and the survey starts
from it rather than repeating it.

**Findability.** **78** live documents, **69,899 lines** (2026-10-09), nine directories
(`docs/` and eight below it; `archive/` and `plans/` excluded). The
information is there and a person cannot reach it.

**The existing index has the problem it is meant to solve.** `README.md`
§ Index (`docs/README.md:211`) is one row per DOCUMENT, and the longest row
runs 823 words (`:250`). To find out which document owns a fact you read
essays about documents. It is a good map of the tree and a poor answer to
"where is X".

**Two documents contradicted themselves in one day** (2026-09-07), both found
by accident while doing something else — `science/pseudopotentials.md` § 3
listed a check as not-done that its own § 2a.1 documented as done (since
corrected; § 3's note at `:470-479` records the removal), and
`process/testing.md` § 6 quoted a number after telling the reader never to
quote one (corrected since: `testing.md:702-704`). Neither is exotic.
Nothing systematic would have found them, which is the survey's real
justification.

### 7.2 The four indexes — organized by THING, not by document

Each is a lookup table, each row pointing at the document that owns the
detail. **None of them may become a second home**: a row says where a fact
lives, never what the fact is. (`README.md` R-W1, and the reason `docs/`
survived its 2026-06-02 over-compression.)

| index | one row per | the row answers | generated or written |
|---|---|---|---|
| **API** | route | what it answers · who calls it · which doc owns it | **generated** — the route list is derivable from the blueprints, and a generated index cannot drift. Two findings are the argument: W8 (2026-09-07: `web-api.md` claimed the Documents tab read `/api/docs/list`, which it has not since the commit after the one that added it) and the 2026-10-08 route sweep (11 documented routes that do not exist, 9 live ones undocumented) |
| **Module** | module | its role · its layer (L1/L2/L3) · its doc | **generated** from `architecture.md` § 3's index, the one list of each module's layer (the scan that classified every module, `test_layering.py`, was retired 2026-09-27, `a291fbbf`) |
| **Data structure** | persisted file / schema | its shape · its version · its one reader and one writer | **written** — the doors are a design fact, not derivable. The substance for a calculation's files exists: `execution/project-layout.md` § 5, every file, who writes it, the one door that reads it — **generated** from `runfiles.WRITTEN` by `tools/manifest.py`, so the index points at it |
| **Reference** | constant · citation | its value · its source · **why this value** | **written** — see 7.3 |

### 7.3 The reference index, which is the one with real content in it

Constants have one home already (`molbuilder/constants.py`), and the dialects
a quantity is written in have theirs (`molbuilder/units.py`); the rule for both
is `architecture.md` § 3.  There is no lint -- `082ba979` retired it in favour
of that written rule. What has no home is the **reasoning**, and it is
genuinely scientific. The worked example: `constants.py:101-106` defines
`HARTREE_BOHR_EV_ANGSTROM_ASE = 51.42208619`, the ASE/NIST value, and says why
the name carries the convention — `HARTREE_EV / BOHR_ANGSTROM` gives
51.422067476, 0.36 ppm apart, and a force and the threshold line drawn over it
must use the SAME one. The emitter imports it (`trajectory_log/emitter.py:40,
43`) so emitted forces line up with what a person reads in ASE, VASP and QE
logs. That is a real convention choice with a real justification, and it lives
in a code comment nobody will find.

Citations are split the same way: `science/references.bib` holds **34**
entries, **8** live documents reference it (`README.md`, `engines/pyscf.md`,
`engines/template.md`, `engines/tuning.md`, `engines/vibration.md`,
`science/normal-modes.md`, `science/vibrational-averaging.md`,
`science/refs/README.md`), and the rest cite
literature in prose, if at all. Two keys the roadmap called for — Reed 2006 and
Stokbro 2003 — are still in neither, though `engines/transport.md` § 8 cites both (row **E14**, reopened 2026-10-09).

So the reference index carries three things per row: the value, where it is
defined, and the sentence saying why that value and not the neighbouring one.

### 7.4 Order, and the one risk

1. **Survey first, and record findings without fixing them** — **done as
   § 5v** (2026-09-29): every live doc read against the code, findings in one
   list, an entry leaving it when its document is corrected. The route sweep
   (§ 0a *Unscheduled*) is the same kind of finding for `web-api.md`.
2. **The two generated indexes**, which are cheap and cannot go stale. Inputs:
   the 86 live routes; `architecture.md` § 3's module list.
3. **The two written indexes**, from the survey's findings. Input for the
   data-structure index: `execution/project-layout.md` § 5 is
   already the manifest (generated) of a calculation's files — the index points at it, it
   does not restate it.
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


---

## 8. One nature, fourteen instances — the reading doors have no guard — OPEN: rows 10, 11, 13

*(Consolidated 2026-09-07, after the selection/MolView review; re-read
2026-10-08 and 2026-10-09. Rows 1–9, 12 and 14 are fixed; the guard of § 8.3
is ruled out — a door's rival is found by reading the code, which is how every
row was found (`code-audit.md` § 1c (e)). The closed rows, § 8.2's account of
the four remaining guard tests and § 8.3–8.5 are in the archives of 2026-10-08
and 2026-10-09.)*

**The pattern.** A door exists. Someone needs it to behave slightly
differently for a local reason. They write a second implementation instead of
widening the first. The two then drift, and the copy is the one that is wrong.

| # | the door | the second implementation | how it drifted | state |
|---|---|---|---|---|
| 10 | the `forceScale` switch | the trajectory template's slider | **confirmed by reading 2026-10-09**: `trajectory/core.js:897-901` sets the switch from `#force-scale` before the load; `molview/ui-context.js:145-149` then runs `restore()` after the first structure and re-sets every saved switch (`:107-110`) — the saved value wins, and the slider (`#force-scale` / `#force-scale-val`, written only at `core.js:2526-2534`) shows another number | open — one owner, the switch; the slider reads it |
| 11 | the filter row kinds | two literal lists: the menu (`molview/ui.js:2695-2698`) and the translator `rowToRule` (`stores.js:353-380`), with the default at `stores.js:224` — `_atom.js`'s `KIND` is the atom-channel taxonomy, not these (the 2026-10-08 premise was untrue) | a kind added to one list and not the other is a menu entry that builds no rule; `structure-annotations.md` § 6 marks the kind menu read from `channelKinds` and the `value` kind's range predicate not built | open — one table, read by both |
| 13 | `molview.md` § 1.1 vs §§ 6.6 / 9.5 / 11.6 | the same facts stated twice | five stale claims on 2026-09-07; unverified since | open — § 7's survey |

*Row 14 closed 2026-10-09: `model/structure-annotations.md` § 6 names the real
homes (`_atom.js`'s `toDisplay`, `atomChannels`, `channelKinds`; `stores.js`'s
`buildRule`, `rowToRule`; `ui.js`; `/api/selection/eval`).*

---

## 9. ATOM as a pseudopotential validator — § 9.1 DONE, § 9.2 PENDING, not started

*(User, 2026-09-19.)* **§ 9.1 — done, archived 2026-10-08**: the user's two
rules — the pseudopotentials are explicitly provided, and the file for element
`E` is named `E.psml` — are refusals when the calculation is configured
(`validation/siesta.py:19`, `:59-100`; `pseudos.py:636`; contract
`science/pseudopotentials.md` § C1a, `:139`).

### 9.2 ATOM as an optional extra layer — PLANNED, not started

*(User, 2026-09-19: treat it the way 3DNA is treated — ask whether they have
it, look for it under the molbuilder root, and enable the extra validation
only when it is there.)*

ATOM 4.2.7, the SIESTA project's own pseudopotential program (Froyen /
Troullier / Martins, maintained by Alberto García). Not in the repo; the copy
read on 2026-09-19 is gone, and the facts below are from that reading.

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
| `x3dna*/` in `.gitignore` | `atom*/` likewise, added (`.gitignore:85-88`): the ignore has to exist *before* the folder does, or a redistribution-prohibited package lands in `git status` |
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
out of scope today (`:446-470`, which points back here), and the ATOM
tutorial's own warning is the argument for them: *"You should thoroughly test
a pseudopotential before using it."*

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


---

## Archived sections — pointers

- [`archive/2026-10-13-5aa-groups.md`](?doc=archive/2026-10-13-5aa-groups.md)
  — § 5aa (the groups) whole, as it stood when the plan was cleaned on
  2026-10-13.
- [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)
  — the rows of § 2 and the queue done, settled or closed by 2026-10-13
  (after 2.9.3), each verbatim under its verdict; and Q19's row as it stood
  after its phase 1.
- [`archive/2026-10-10-q17.md`](?doc=archive/2026-10-10-q17.md)
  — § 5z (Q17) as it stood at its close, its queue row, and the handover
  of 2026-10-09/10.
- [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)
  — the queue and milestone tables as they stood; § 0b; § 0c (W55, W56, W57,
  unit 12); *Unscheduled*; the whole of § 2 as it stood, with a verdict per
  row; the pointer sections § 5a–5c, § 5f–5p, § 10 and § 11 with what they
  carried; § 5v; § 5w as it stood; § 5x; § 5y; § 5z as it stood (Q16, F.1–F.7);
  § 8.2–8.5.
- [`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md)
  — §§ 1, 3, 4, 4a, 5, 5d/5i/5j, 5k, 5l and 6 (pointer sections since
  2026-09), and §§ 5a, 5b, 5c, 5f, 5h, 5m, 5n, 5o, 5p, 10, 11 in full.
- [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md)
  — the rows closed when § 2 became the one table.
