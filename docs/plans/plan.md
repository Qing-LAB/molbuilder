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

---

## ▶ HANDOVER — where we are, and where to start *(2026-10-13)*

*(user, 2026-10-09: "consolidate the plan and make sure we have all the
discussions persistent and ready so that when i come back here we can
continue without losing any context/details")* Read this first; each line
points at the row or section that holds the detail.

**Just done — 2.9.0 to 2.9.4** *(2026-10-13, each pushed)*: the k-point
wording in one vocabulary (2.9.0); the 2026-10-11 walk's leftovers (2.9.1);
the first phase of Q19's three passes — small UI, CSS, Python — each
reviewed by fresh agents on the full code and contract text, the none2e
batch run, the one confirm unified on the one dialog component (2.9.2);
the server's numerical threads capped at one (2.9.3, in effect on port
8888); Task setup as a checklist — every card folds, its head saying its
answer, the rail one card at a time, Save in a bar pinned to the window, and
the task's Prep after the group tabs (2.9.4, the user's request; `web/
task-setup.md` §§ 7a, 9a.1, 11). Q19's row holds the detail and what its
reviews left open.

**Before that — Q17, transport over a set of frames** *(2026-10-08 → 10; its
record archived, § 5z)*. A transport calculation cites a finished relaxation
run (one structure) or a structure pair that records its optimization — one
frame, or several: a mode's frame set (its weighted average beside the
frames) or a plain family (each frame its own curve). It runs a point per
frame and voltage; the record, the data file `<label>.transport.nc` (every
result on its grid), the Results tab's *one frame / all frames* on every
chart, the Transport tab's frame-set card and Task setup's echo; one Plotly
box for every Results chart. Reviewed in two rounds; walked through the road
on an H2 frame set in Chrome; the user's rulings on the review's calls built
(plain words on the card, the Setup card's line per form, a plain message
where the report is not on screen, *a cited pair is left alone*).

**Also done — Q18, the CSS sweep** *(2026-10-10)*: the defects its row
lists, one chart theme (`lib/plot-theme.js`) and one series palette
(`--series-1…5`) for every chart.

**molbuilder 2.0.0** *(2026-10-11, tag `v2.0.0`; the user: "we are closer
to release")*: the version has one home (`molbuilder/__init__.py`), every deck
and run script records it, and every version has its line in the changelog
(`process/conventions.md` § 2, *Versioning*; `process/changelog.md`). **The
numbering, disciplined 2026-10-13** (the user: "middle number means new
functionality, major number means major framework change"; "we will discipline
from now on"): the middle number for new functionality only, the last for
everything else, once per push -- 2.1.0 to 2.9.0 moved the middle number for
any change to a file, too loosely, and stand as given.

**Next, in order** (the queue, § 0a): Q19's later phases when the user
takes them; **Q5**, **Q7**, **Q9**, **Q12**, **Q13**, **Q11** — as the
queue orders them.

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

**Waiting on the user**:
- **The frame generator's amplitude** — V1.25's row in § 2 holds the
  proposal. Decided when the generator is designed.
- § 2's *the user's word* rows: E7, EL-W, PL-R's first question, K22's design,
  V1.26 (M10).

**Owed by the next e2e pass** (M2n; on the user's word): § 2's DOC-TOC (a
docs-tab test writes into the checkout) — retire.

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

### THE QUEUE — what is in hand, in order *(2026-10-02; rewritten 2026-10-09)*

> *(user, 2026-10-02: "consolidate your plan so we don't lose track of these
> items and correctly prioritize and plan in a persistent plan and correctly
> update as we move down the list"; "we need to get the transport task
> finished asap")*

**Read this first.** It is every item in hand, in the order it is worked; the
milestone table below and § 2 keep each item's detail. A row moves in the
commit that moves it, and a row is never re-ordered without saying who ruled
it.

| # | item | state | next |
|---|---|---|---|
| **Q19** | **Three passes — CSS, small UI, Python** *(proposed 2026-10-13 from three read-only surveys; the user: "go with your recommendations … think at framework level or larger scope where the snippet sits, need a systematic point of view for fixing"; for Python: "make sure final outcome to be checked against design contract, and reviewed in their full code/text and validated correctly")* — **small UI**: P1 the four rule defects and the fence rule (one `pageBusy.run`, every Task setup edit through it; V1.14's suffix, R2-15, MV-RES, LOAD-T; stale comments), P2 the design calls (the shape shown fixed with its way back, one answer `prep.shape_fixed`; the chip's budget advice removed; the five generators on one chain), P3 the prep report served as lines — **the wording to keep is the user's call**, P4 the small shared helpers; and the four found while measuring, fixed with P1 as proposed: the stale comments (F3, F4), one "x ago" (F7, `lib/time-ago.js`), the markdown editor's `beforeunload` guard gone (F9, `runtime.md`'s retirement said for every page), the transport road said once — the Describe line names the *After you describe* card (F5); **CSS**: P1 one dialog system (`lib/dialog.js`, `warningModal.confirm` the one confirm loaded by the header, no native `prompt`/`confirm`/`alert` — the 16 alerts now notifications, `ui-contract.md` § 11), later phases on the user's word with decisions 1–8 as recommended; *found while measuring, open*: three animations ignore the reduced-motion setting the contract says all honour; the layout test claims every page and skips Transport, This machine and JupyterNB; **Python**: P1 pyflakes to zero, `notify.sign` → `monitor.sign_report`, the doc drift, a full-code-text review by a fresh agent; | **small UI P1 and P2, CSS P1 and Python P1 done 2026-10-13** (2.9.2, `22ae300d`): the small UI pass reviewed in two rounds, CSS P1 and Python P1 in one each, by fresh agents on the full code and contract text — every finding re-read in the code, then fixed or answered; every new or changed test broken once, red. **The one confirm unified** *(user, 2026-10-13: "yes, unify the confirm")*: `warning-modal.js` is a module built on `lib/dialog.js` — one dialog at a time with every other, Esc and a click off it answer no; its 16 tests became 7 rows. **Open from the reviews:** MolView's export dialog is a third modal, a `div` overlay outside the component (`lib/molview/ui.js`), with CSS's later phases; the layer index has no row for `template`, `runrecord`, `kmesh`, `workingcopy_structure`, `frameset`, `resolve`, `placement`, `describe`, `projects`, `jupyter`, `auth_setup`, `spectra/`, and four L1 modules import upward inside functions (`electronic_state` → `template`, `task` → `jobset.commands`, `warmfiles` → `jobset.model`, `runfiles` → `sidecars.molstruct`; `trajectory_log/emitter` → `pyscf.end_lines` at the top) — the Python pass's phase 5 | the none2e batch ran 2026-10-13 (7,218 tests; its one failure a real duplicate — the SMILES and name panels' `generate` — moved into the shared chain, `panelDeps.generateText`, its slot checks with it); the dialogs walked headless on an isolated tree (no browser dialog, nothing written); **pushed as 2.9.2** (`22ae300d`, on the user's word); the server's 40 numerical threads **capped at one, 2026-10-13** (the user: "one thread may be enough for handling the light tasks"; `ops/deployment.md` § 1.0f — the supervisor's child alone; the server's numpy paths 0.1–4 ms either way, a job-side eigensolve 100× slower on forty threads under the machine's load) — both supervisors, one environment (`reload_protocol.server_environment`); `serve start`'s child no longer names `--no-supervise`; 2.9.3 (`fa6bc2ce`), in effect on port 8888 since its stop and start at 13:55 that day (the child: one thread, `OPENBLAS_NUM_THREADS=1`). **Next: the passes' later phases, on the user's word** |
| **Q5** | **Transport's surface, the rest** — § 5u.1 steps 7 (the TranSIESTA / TBtrans items still unsaid), 8 (one panel per engine) and 10 (every value says where it came from); the deck viewer W30 ④, which lost its home when step 9 closed | **open — in hand from 2026-10-11**: step 10 done, step 7 done (2.1.0), the deck viewer done 2026-10-11 — every calculation's ladder, not transport's alone (W30 ④); step 8 next, on the user's word *(said to the user 2026-10-11; TD13 retired the same day)* | § 5u.1 |
| **Q7** | **M4 — the engine offset finished**: P4's three items (§ 5q.6) | open | § 5q.6 |
| **Q9** | **M11's remainder** — K13, K14's parallel model, K16's fence, K18, K17's open list, K22 (the user's design word first), K9, then W50's step 3 | open | § 5w |
| **Q12** | **the rest of the work order below**: M2j closed at its review, M2k's GPU half, M2m, M2n, M3, M6 P5, M8, M10 | open | the table below |
| **Q13** | **The task's left items** — `prep task` / `launch task` built 2026-10-08 (`214cc19d`, `91629b67`), Task setup's Prep card seen in a real browser (`1a28e67f`) | **left**: `tests/test_task_setup_prep_e2e.py` drives the per-tab Prep buttons that are gone — judged and rewritten or retired (e2e on the user's word); a group's header under a real scheduler (Sol); `prep_group` re-plans each member after the save and prints each member's notes (§ 2, F9) | § 2, F9 |
| **Q11** | **The named-queue check in two doors** — `placement.launch_refusal` (`placement.py:28`) and `scheduler/place.place` (`scheduler/place.py:78`) | open | one door, asked by both |
| **Q8** | **Ruled 2026-10-02, not built**: one `proxy.trust` setting replacing `auth.trust_proxy` and `rate_limit.trust_proxy` (`configuration.md` § 8); a benchmark's grid stated by the description, refused when it is not — no grid the machine proposes | ruled | one config milestone |
| **Q2f** | **Absolute imports, package-wide** *(user, 2026-10-03: "i agree with a, but like a focused session for that with agents and reviews. let's finish this main goal of transport and framework related tasks")* — every `from ..x import y` inside `molbuilder/` becomes `from molbuilder.x import y`, the rule written into the conventions; **and the shipped files' imports in the same session** *(ruled 2026-10-06, W57 decision 1: "B, with Q2f as one session")*: each zip (`mb_monitor.pyz`, `mb_vibration.pyz`, `mb_pyscf.pyz`) keeps molbuilder's folder layout so the package's absolute import resolves inside it (user, 2026-10-06: "for the first form of import why can't we use absolute path?"); the try/except ImportError pairs leave the 22 shipped files. Until then a NEW module is written absolute | the user's word — its own session | after the transport and framework goal — a script with an agent review, then the full batch |

Done rows, archived (`archive/2026-10-09-plan-consolidation.md`, with the
earlier ones in `archive/2026-10-08-plan-consolidation.md`): **Q2d** (W55, units
1–12; 12b-2 dropped, 12c/12d built `ac2d5741`, 12e defined nowhere), **Q2g**
(the file manifest; D29's document list corrected 2026-10-09), **Q4** (TD9),
**Q6** (step 9 done — § 5x B6; step 8 → Q5; step 11 → Q17), **Q10**, **Q14**
(§ 5x, B0–B9), **Q15** (§ 5y, `22ac10d9`, `4963406b`), **Q16** (`archive/2026-10-09-plan-consolidation.md` § 5z,
`a19ddfcc` and after), **Q17** (§ 5z, archived 2026-10-10), **Q18** — Q17's and Q18's rows in
[`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md); before them Q1, Q2, Q2c, Q2e, Q2h, Q3.

**Why transport comes before M11's remainder** *(2026-10-02)*: the user's word
above. The K items left are framework-wide (writers, allocations, the run
record's rows) and transport's decks pass through them, so each is done when a
transport step needs it, the rest after.

**How a milestone closes:**
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
5. this row records the commits, the review and its outcome — and only then
   does the next milestone start.

A full test batch (`tools/testrun.py`) runs at the end of M2 (M2n) and at the
end of each programme after it, with nothing changing under it.

| # | milestone | items | done when | status |
|---|---|---|---|---|
| **M2j** | status owns the ladder | W38 F8 | every described stage in status; launched-and-unfinished attempts never hidden; the Results tab reads status | **looks built** (2026-10-09 validation): the Results door reads `jobset_status` (`results.py:198-211`), `runstatus.every_run` lists every attempt (`runstatus.py:240-275`), `status <stage>` lists its attempts (`_cli.py:723`); the queued-attempt question dropped by the 2026-10-07 ruling — **closed by its review**, with M2k |
| **M2k** | what a job runs with | W36 ⑦, W38's GPU half | one placement and one record (done, unit 11a: `placement.py:182,236,130`, `model.py:267`); the precedence table (`running-a-job.md:430`); the GPU request (9b part 4); **the GPU claim / match** | partly — the claim / match half and W36 ⑦ open |
| **M2m** | the structure's identity | V1.9 (with X4 ⑤ and W39's hash) | the identity rule written in `structure-molstruct.md` § 3 — two hashes, where each is minted and checked — then built: the hash checked on read, prep refusing a structure whose hash moved; a frame-set citation pinned by its two files' sha256 meanwhile (`archive/2026-10-09-plan-consolidation.md`, § 5z.8 F.3) | open — W39's own part (the `customized` section, the API, the pane) built `c93da223` |
| **M2n** | the text, then the batch | W36 ⑪; the document rows of § 2 (DOC-*) | the full batch green; then the engine e2e runs owed by closed milestones (M2b⁗, M2b⁵, and the six files M2d did not run), the Sol memory measurement, and the sweep for tests that read text | open — the last e2e batch ran 2026-10-05 |
| **M3** | the run record, the rest | W35: P2's remainder, P4, P5, P6 | W35's own done-conditions (§ 5t.3) | open |
| **M4** | the engine offset, finished | W33 P4 | § 5q.6 | open |
| **M5** | transport | § 5u.1 steps 7, 8, 10; Q17 | each step's done-when | steps 1–6 and 9 done; step 11 is Q17 |
| **M6** | charge and spin | W34 P5 | § 5s.3 | P0–P4 built (decision 4's refusal: `citation_defaults.py:196-207`); **P5 open** |
| **M8** | MolView sealing, the CSS | W15; W1–W6, W13 | each row's own | open |
| **M10** | from a mode to the current | W42 (V1.24–V1.27 folded in) | V1.26's decision; then W42's order P1–P6 | decided 2026-09-28 (W42: D1–D5, D7, D8; D6 as recommended); V1.26's decision owed; after the work order. V1.25 — the frame-set generator — is unblocked by W39 and feeds Q17 |
| **M11** | every parameter, end to end | W50 → § 5w | W50's own: a static full-text review per track, each finding verified and fixed at its owner, then the road in the browser | the reviews done 2026-09-29; their 22 mechanism classes (K1–K22) — K1–K8, K10–K12, K15, K19, K20 done; the rest § 5w |

Done milestones, archived: M1, M2a, M2b–M2b⁵, M2c, M2d, M2e, **M2f** (9b, 2026-10-03), **M2g** (9b part 3), M2h, M2i, M2l, M7, M9 (folded into M10).

---

## 2. OPEN — the one list

**Every open item, in one table** *(consolidated 2026-09-10 at the user's instruction: "consolidate plan and to do, archive finished things and untrue things, clean it up so we have one list"; re-consolidated 2026-10-09)*. A design section (§ 5e–§ 9) keeps the *why* and the order of its own work and sends its rows here; the queue (§ 0a) says when.

**A row is evidence of when it was written** — re-derive it against the code before acting (the rule, `archive/2026-10-08-plan-consolidation.md` § 5a). Each row below was re-read at its cited place on 2026-10-09; *Re-checked* says what was found. The long histories of the rows that had grown to tens of lines (W36, W38, W41, W30, W32, X4, W52, W54) are in [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md), verbatim; the row here is what is left.

**Rows that were done, withdrawn or measured untrue are not here** — [`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md), [`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md) [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md) and [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md), each with what it turned out to be. Archived ids still cited elsewhere resolve there: A1.12, A1.16, A1.2, A1.3, A1.7, B2, B3, B4, D2, D4, DOC-TEX, E1, E10, E11, N5, N9, R5, S-T2, S18, S3, TD10 text, TR1–TR3, TR4–TR6, TR7–TR8, TR9–TR10, TD6, TD12, R2-15, SCF-RUNG, CHIP, V1.31, W10, W20, W27, W37, W39, W46, W51, W52, W53, W54, X1.

| # | area | item | from | state |
|---|---|---|---|---|
| **N1** | identity | **Rename molbuilder to Benchmate, with the H5 icon.** The name a person sees and types changes (title, icon, command, import name, conda environments, config directory, `BENCHMATE_*` variables); every name stored in existing files stays (`.molwatch.log`, `.molstruct.json`, `molbuilder/<kind>@<n>` schemas, `mb_*.pyz`). Six phases, each ending in a working state, with old names read as fallbacks until Phase 6. The detail, the measured inventory and the icon are in [`plans/rename-benchmate/`](?doc=plans/rename-benchmate/README.md) — this row is R3's pointer to it | user, 2026-10-18 | open — Phase 0 (plan written; decisions D1–D6 for the user) |
| **E7** | engine / science | **D7's cluster half** — the prep→submit→watch loop for a **run** through SLURM on Sol. **2026-09-10, the user: "we have tested it on Sol" — and the row must not contradict that.** What is actually on disk here: Sol slurm records exist (`optimization/sol/AuBDTAu-slabcorrected/01_coarse/bench/launch/slurm.62380919.out`, real ids) and every `kind: "run"` ledger entry in `projects/` is `workstation`/`direct`. That is the absence of a LOCAL record, which the earlier wording ("No `kind: run` submission exists on any Sol tree") presented as the absence of the WORK. A tree on Sol is not in this repo. **This row needs the user to say whether it is closed, not another file scan** | roadmap § 1 | needs the user · *Re-checked 2026-10-09:* open, the user's word; the run's folder is now `projects/Au-BDT-Au.old/optimization/sol/AuBDTAu-slabcorrected/`. |
| **T1** | front end / results | **A live poll that must rebuild does not update the movie — the cause found by reading, 2026-10-09.** Found 2026-09-07: grow the feed 4 → 6 while the frame at `oldLen − 1` moves, so `canAppend` (`lib/trajectory/core.js:2159`) refuses and `applyNewData` (`:2136`) rebuilds; the status says 6 frames and the bar holds 4. **Why:** the Results viewer is mounted read-only (`core.js:595-596`); `rebuildModel` re-installs without `enforce` (`:915-921`); MolView's § 9.4 guard returns `null` for a read-only viewer already HOLDING a structure (`molview/model.js` ~`:1112-1116`) — so every rebuild after the first, and the two resync paths (`:2290`, `:2311`), is a silent no-op. `web/molview.md` § 8 marks it | found 2026-09-07; `web/results.md` § 4.1, `web/molview.md` § 9.3–9.4 | **open — a defect against § 9.4**: the rebuild installs through the read-only door's own replace (`enforce`), and a test drives the 4 → 6 rebuild in the browser model |
| **E13** | engine / science | **The first real junction walk — BDT–Au on Sol. DROPPED BY THE CONSOLIDATION.** Named in the roadmap and again in `transport-design` § 7's *"order of proof"* as the run that follows P6. plan.md has the machine-blocked infrastructure (E5 — archived 2026-09-10 — and E7) and the browser and deck walks (W12 — archived 2026-09-10 — and E11) but no row for the transport composite's first real science run | roadmap · `transport-design` § 7 | **the Sol half only** — after § 5u step 4; the workstation half ran 2026-09-29 (the Au–BDT–Au ladder to T(E) on `claude-au-bdt-au`, TD8's acceptance ladder) and is archived *(2026-10-08)* · *Re-checked 2026-10-09:* open — the Sol half only; the workstation half closed by TD8 and Q15's run E. |
| **E14** | science / references | **Reed 2006 and Stokbro 2003 into `science/references.bib`.** `engines/transport.md` § 8 cites both (and Solomon 2008 beside them); none is in the `.bib`, and the rule is that an entry is added when a passage cites it. The 2026-10-08 archive reason — *no live document cites them* — was untrue | roadmap § 7.3 | **reopened 2026-10-09** — each entry checked against the paper before it goes in (§ 7.3) |
| **V1** | engine / science / web | **The vibration calculation — one path on two engines.** The contract is [`engines/vibration.md`](?doc=engines/vibration.md) (§ 10 says what stands); the science is [`science/normal-modes.md`](?doc=science/normal-modes.md); the design and the first audit are archived (`archive/2026-09-24-*`). Built 2026-09-21 → 24: one harmonic path with the rank rule and its gate against PySCF, one mass convention, stationarity on the free atoms, the runs write the pair, the Hessian over the free atoms with the two measured corrections, the SIESTA arm (deck on a sorted copy, `atom-permutation.json` with its key, the `.FC` reader, `summarize run`, the warm-file section, the kind's start state), the equilibrium block optional, the wrapper's banner. **Open, the sub-rows:** | `engines/vibration.md` § 10 | open · *Re-checked 2026-10-09:* a container: V1.11, V1.15 and V1.38 narrowed, V1.25 unblocked, V1.31 archived. |
| ↳ | **V1.9** | **The structure identity rule** (settled in discussion 2026-09-22, written into no contract yet — this row is its full statement until it has one): **two hashes, one home** — a *geometry hash* over the atom lines as **verbatim text** (the numbers are already text, so there is no float question) plus the lattice and the per-axis periodicity from the sidecar, with `none` as a real hashed value when there is no lattice; and a *broad hash* over labels, regions and identity columns, for provenance. **Never the comment line** (a human title from one writer and a derived `Lattice=` from another; adopting it would promote a derived value into a stored one). **Out:** the cell origin (a gauge choice — shifting it moves no periodic image), labels, timestamps, the job name. **Joins check the geometry hash only**; a broad-hash difference is reported, never blocking. **Minted at three gates** — Modify → Save to project, Results → export, the CLI on request — then **carried and never recomputed**; **checked at three points** — prep, results load, joining two runs. **Derived structures** (a reordered copy) mint their own identity and record the parent's plus the permutation. **A run's output is a record, not a gate** — it becomes an input only by passing through one (ruled 2026-09-22). Today: `spectra.json`'s `structure_hash` has the job name as line 2 (the atom count is line 1) and never matches the codec's pair hash, which is over the document's bytes — a different scheme; and `workingcopy_structure.py:278`'s `keep_sidecar = (not _metadata_is_default(meta))` writes no companion for all-default metadata, so *Save to project* can still emit a bare `.xyz` at the very gate meant to guarantee the pair (both measured saves got one only because they carried labels) | `vibration.md` § 4.3 | open · *Re-checked 2026-10-09:* open — M2m. Both facts hold: the spectra hash puts the label on line 2 (`sidecars/spectra.py:85-96`); a structure with all-default metadata is saved with no sidecar (`workingcopy_structure.py:304-307`). No contract states the rule yet; `structure-molstruct.md` § 3 defers the identity hash to M2m, and a frame-set citation is pinned by its two files' sha256 meanwhile. |
| ↳ | **V1.10** | **The pair writer renders both halves** (ruled 2026-09-23, the unification audit § 1.1a): `pair()` returns text, so a sidecar that cannot be written (a NaN in `info`) stops the write before the `.xyz` is on disk. **The deck's half is done (2026-10-05):** the PySCF deck writes its pairs with the codec itself, imported from `mb_pyscf.pyz`, and its own serialiser — the third — is deleted | unification audit § 1.1a | open (the codec's half) · *Re-checked 2026-10-09:* open — `pair()` returns the sidecar as a dict (`workingcopy_structure.py:298-308`) and `_write_pair` writes the `.xyz` first (`:420-446`); `structure.md` § 2.4 marks both-or-neither not built. |
| ↳ | **V1.11** | **Untested and open**: the reduced Hessian with a GPU mean field; a composed permutation for a structure sorted for two reasons (ruled one record, no caller yet); `transport/compose.py` writing its record through `write_permutation` and stamping its key (U10; reading through `read_permutation` done 2026-09-28) | `vibration.md` § 4.4, § 5.2 | → **§ 5u step 3** *(2026-09-29: compose writes the permutation by hand and calls the sort directly, so no key is stamped; `sort_by` / `write_permutation` exist for it)* — **the compose half built 2026-10-02 (M5 step 3)**: it sorts with `sort_by(…, "transport")` and writes through `write_permutation`, the key stamped (`test_transport_prep.py` reads it back) · *Re-checked 2026-10-09:* the compose half done (`transport/compose.py:757` `sort_by`, `:831` `write_permutation`); the test the row names (`test_transport_prep.py`) went in `93977f5d`, so no test reads the transport permutation key back. Untested: the reduced Hessian with a GPU mean field; one record for a structure sorted for two reasons. |
| ↳ | **V1.12** | **Not done**: a mode-by-mode intensity cross-check against an external code (Gaussian / ORCA / Turbomole); absolute intensities carry the caveat until then | `vibration.md` § 9 | open · *Re-checked 2026-10-09:* open (`vibration.md` § 10). |
| ↳ | **V1.14** | **From the UI walk of 2026-09-23, not this kind's**: the Molbuilder tab's save prompt doubles a typed suffix (`x.xyz.xyz`); the `#`-labelled provenance regions (`O#`) warned as unconsumed on every SMILES-built molecule — needs a ruling; the vacuum notice advising 8 Å on a gas-phase PySCF run | unification audit § 1.18 U3, U4, U7 | **the suffix done 2026-10-13** — the door adds the extension to a name typed without it and takes one typed with it, any case (`molview-doors._askProjectPath`, both the structure and the picture; `projects.md` § 3; `test_molview_files_door_js.py`, mutation red); *open:* the vacuum advice · *Re-checked 2026-10-09:* the doubled suffix stood — `lib/projects/molview-doors.js:114-117` appends `.xyz` to a typed `x.xyz`, the save route writes the path as given; the 8 Å vacuum advice is engine-blind (`cell.py:398-405`); the `#`-label warning is settled by *labels are the user's* — nothing to build. |
| ↳ | **V1.17** | **Four held systems through the whole road**, today rank rows only: acetylene with both carbons held (the collinear trap end to end), NH₃ with its three hydrogens held (nothing removed), an empty held list reproducing the free path *exactly* (what makes "the free case is the held case with nothing held" a fact), the water dimer with one molecule held (the over-removal guard) | `vibration.md` § 9 | open · *Re-checked 2026-10-09:* open (`vibration.md` § 10). |
| ↳ | **V1.24** | **Mode matching across runs** (needs a design): the overlap of eigenvectors in the shared free subspace, mass-weighted — the active-region convergence test (Models A/B/C) and the PySCF-against-SIESTA comparison of one molecule are the same calculation | `science/normal-modes.md` § 4b.6 F, § 4b.3 | open · *Re-checked 2026-10-09:* open — M10. |
| ↳ | **V1.25** | **The mode-displaced frame set** *(decided 2026-09-24, user)*: a generator that writes ONE multi-frame pair — the base as frame 0, then `R_A(Q) = R_A⁰ + Q·L_canonical` with `R_F` unmoved at the zero-point amplitude and its thermal growth — from a spectra file, for a named mode; the rule (run, mode, amplitudes) recorded in the pair's `info`. It is the interface to transport's frame axis (W32); a person's own script writes the same pair. Level one of `vibration.md` § 5.6. *W42 (draft 2026-09-28) proposes one set for many modes, at the Gauss–Hermite nodes of the thermal distribution (D6, D7)* | `engines/transport.md` § 2a.9 · `model/structure-molstruct.md` § 6.1 · `science/normal-modes.md` § 4b.6 G | **contract written 2026-09-24; re-ruled 2026-09-29 (TD10)**: the generator is a separate backend procedure, designed on its own — the equilibrium relaxed structure and a normal mode in, one multi-frame structure out — and **each frame's parameter set goes in the structure's `customized` section (W39), not `info`**; transport is its consumer, never its builder. Open, after W39 · *Re-checked 2026-10-09:* unblocked — W39 built in `c93da223`. The contract is `vibration.md` § 5.10 ③: the generator, its own module, writes one frame set for **one** mode (user, 2026-10-09) — `with_frames`, the mode on the structure's rows, each frame's node and `weight` through `set_customized(…, frame=i)`, frame 0's weight included, the set's summing to 1. **Its amplitude, to design** (the user, 2026-10-09: *"how much should the max displacement of a selected vibrational mode use? ... temperature involved? how about local EF enhanced (such as SERS in Raman etc) mode? what's the good starting point"*) — proposed 2026-10-09: the spread is the mode's own, `σ² = (ħ/2ω)(2n+1)` — thermal `n` from the experiment's temperature (zero-point above ~300 cm⁻¹ at 300 K, temperature-driven for contact modes), a stated effective `n` / `T_eff` for a pumped mode (SERS: incoherent, still Gaussian, wider), the max displacement then an output; a coherently driven mode is the one case where the max displacement `A` is the input — the arcsine distribution, Gauss–Chebyshev frames at `0, ±0.866 A`, weights `⅓` each; always a warning past a stated fraction of the nearest-neighbour distance. Its consumer, Q17, is built (a person's script writes the same set today, as the real-junction e2e's does). |
| ↳ | **V1.26** | **`Δρ_ν(r)` maps** (needs a decision): SIESTA's density grid at `±Q_ν`, differenced — the discussion's intermediate quantity before any oscillator strength on a metal | `science/normal-modes.md` § 4b.7 | open — M10; its decision is owed · *Re-checked 2026-10-09:* open — M10; its decision owed. |
| ↳ | **V1.27** | **The PySCF probe** (decided — W42's D8): five points per mode instead of two; the coupling per zero-point amplitude `g_ν = ∂ε/∂Q · √(ħ/2ω)` in meV written beside `ΔE/(2A)`; a molecule-projected frontier quantity for a cluster whose HOMO and LUMO are metal states. *Decided with W42 (D8, 2026-09-28): the probe runs on the thermal nodes* | `vibration.md` § 4.8; `science/normal-modes.md` § 4b.5 F–G | open — M10 · *Re-checked 2026-10-09:* decided (D8), not built — M10. |
| ↳ | **V1.33** | **The Task setup tab re-derives the vibration ladder in JavaScript** (`proposedFromHandover`, `task-setup/viewer.js:2775`, spells `relax`/`freq` and the box rule by hand; `_afterRunLines` is gone) while Python owns it (`pyscf/stages.py:120`, `vibration_stages`). A second copy of one rule; the hand-over or the folder answer should carry the proposed ladder computed server-side, and the page should ask the description which rung is the force-constant one. Found by the 2026-09-24 review | `vibration.md` § 2.2, § 5.2a | open *(re-cited 2026-10-08)* · *Re-checked 2026-10-09:* the JavaScript copy is now `task-setup/viewer.js:2807-2838`; Python's `pyscf/stages.py:120`. |
| ↳ | **V1.34** | **The response screen — which mode to take to transport** (decided with W42, 2026-09-28): the vibration kind's half of W42 — each mode's character at load, the charge each mode moves from the force-constant run, the frame set at thermal nodes (V1.25), the PySCF probe on them (V1.27) | `vibration.md` § 5.10 · `science/normal-modes.md` § 4c | open — M10 (W42 decided) · *Re-checked 2026-10-09:* open — M10. |
| ↳ | **V1.37** | **A vibration's result is written readable by its owner alone.** Found 2026-09-29 on the Au–BDT–Au spectrum: `aubdtauvib.spectra.json` is `0600` while every other file of the run is `0664`. `sidecars/spectra.dump_spectra_json` writes through its own `tempfile.mkstemp` + `os.replace` and never widens the mode (`sidecars/spectra.py:163-180`); `persist.write_bytes` -- the one writer -- does (`persist.py:68-88`), and says why (*"mkstemp creates 0600, which is not what a shared artifact should end"* up as). The finish runs it job-side from the bundled copy (`spectra_sidecar`), which may be why it keeps a writer of its own: whether `persist` can travel in the bundle decides the fix -- one writer, or the same widening in the bundled one | `persist.py` · `sidecars/spectra.py` | open · *Re-checked 2026-10-09:* the writer is `write_spectra_payload` (`sidecars/spectra.py:98`, its mkstemp `:159-180`); `persist.py:88-129` is the writer that widens the mode — K13's. |
| ↳ | **V1.38** | **A δ-convergence comparison that also varies the mesh.** `engines/vibration.md` § 9 still names it as owed; V1.23 closed as built with the δ axis only (the displacement sweep), so the mesh half — the one that separates the grid's error from δ's (§ 9's H₂ measurement) — has no row. Found by the 2026-09-29 coverage check | `vibration.md` § 9 · `science/normal-modes.md` § 4b.6 C | open · *Re-checked 2026-10-09:* narrower: the sweep takes a mesh stage already (`spectra/displacement_sweep.py:8,17,152`; `vibration.md` § 5.9); what is left is the measured δ+mesh comparison on H₂. |
| **W33** | structure / engines / web | **THE ENGINE OFFSET — one placement rule for every engine** *(user, 2026-09-25: "always adjust it before sending to siesta or other engines that the coordinates of all atoms are centered inside the cell"; "we don't have to have special logic to treat isolated, periodic, transport axis_info differently")*. Every engine receives the design coordinates + `engine_offset` with the cell at the origin; the offset is computed from the cell and every atom (the fractional span centred, never re-wrapped), recorded in every deck, and stated 0 for an engine's own output, so the Results tab draws what the engine had. Retires `cell_origin` (an origin the person assigns is kept, stored as the offset), the derive-on-null corner and its per-axis rules, calibrate, and the two hand translations. Found by the fake-junction ladder: a TranSIESTA device refused on a flush corner, a Results box the engine never had, the molwatch step-0 jump (the 2026-09-25 audit's finding X4 — not § 2's row X4) | `model/structure-periodicity.md` § 6.0 · § 5q | **P0–P3 done** (P3's T3 on 2026-09-27, `tests/test_results_export_e2e.py`; its dev-server check on 2026-09-25)**; P4 open** — P5 closed by TD8, the Au–BDT–Au ladder run to T(E) on 2026-09-29 (§ 5q.6 is the per-phase record). **D1–D16 settled** (§ 5q.8) · *Re-checked 2026-10-09:* P4's three items (§ 5q.6) — M4. |
| **W34** | science / engines / web | **THE ELECTRONIC STATE — charge and spin as one answer per calculation** *(user, 2026-09-25: "put spin and charge setup in to a unified framework so that these information can be produced consistently and systematically for different engines"; "investigate holistically ... from template to validation ... a design gap/framework level investigation rather than a patch"; "documentation should have an explicit discussion on species with these properties")*. Four engine-neutral template items — `net_charge`, `spin_treatment`, `unpaired_electrons`, `method` — resolved once with what the structure adds (the charge's source, the electron count, finite or repeating) and read by every deck writer, check, hand-over, form and read-back; restricted-open explicit; a species-by-species chapter. Found by the transport ladder's rung reports (a gold lead and a formate ion both told "switch to open-shell") | `science/chemistry-correctness.md` §§ 2a–2b · § 5s | **P1–P4 built** (M6, 2026-09-28/29; decisions 1–9, § 5s.2); **P5 open** · *Re-checked 2026-10-09:* P5 open — the TBtrans two-channel half built (`transport/record.py:157-165`, through `tbtrans.transmission_files`); no reader of SIESTA's charge and moment, none of PySCF's ⟨S²⟩. |
| **W35** | parse / execution / web | **THE RUN RECORD — what ran, with what, and how it went, for every run; one SIESTA-family reader; the transport report** *(user, 2026-09-26: "include in the report of transport results ... all indicators that can provide scientific information, progress tracking and symptom of non-convergence"; "parameters used for calculating transport should be similarly reported ... check what we did for the other tasks, and find the best way to honestly and fully record what is the computation setup and scientific setup, and how result evolves in the record, and output report"; "fdf parser need unified upgrade"; "get the framework and code unified and finalized")*. One record per attempt, composed on read — computation, setup (default · asked · engine used), the deck, every iteration of every phase from one grammar, the verdict with its symptoms — read by a Run panel for every kind and by a transport report built from its rungs; the NEGF contour stated rather than left to TranSIESTA's fallback. Found by a device that diverged for eleven hours with every reader saying otherwise | `model/parse.md` § 5d · `engines/transport.md` § 2a.12 · § 5t | **P0–P1 done (`a4c1d28e`); decision 10 done — the monitor, the wrapper's endings and the report fields through the framework; P2: the record and the Run panel built (2026-09-27), the rest below** · *Re-checked 2026-10-09:* M3 — P2's remainder, P4, P5, P6 (§ 5t.3). |
| **W36** | execution / packaging | **The run-independence review (2026-09-27), agreed item by item** *(user: "i need to go through all of them so we have agreement before i let you just go ahead to fix all")* — its full text, ① to ⑪ with each agreement, is in the 2026-10-09 archive. ①–⑥, ⑧, ⑨ and ⑩ are built. **⑦, open: the job inherits the whole shell** — `submit.py:1232` passes `{**os.environ}` and `:307` `--export ALL`, and the wrapper honours `OMP_NUM_THREADS` (`runwrap.py:2224-2236`); the agreed fix is *keep inheriting, guarded*: launch hands `-np` / `-omp`, which win (`submit._run_sh_args`), so an inherited count cannot resize a launched run — what is unbuilt is the by-hand path and the log naming an override. `running-a-job.md` § 2.0a states this; its § 3.2–3.3 precedence still puts `OMP_NUM_THREADS` second, against ⑦'s *"a bare OMP_NUM_THREADS no longer sets it"* — one table, written when ⑦ is built. **⑪, done 2026-10-09**: the `$MOLBUILDER_ROOT` text is gone from `job-contracts.md` § 2.5, `repo_root`'s docstring names its real callers | `execution/running-a-job.md` § 2.0a, § 3.3 | ⑦ open — M2k |
| **W38** | execution / front end | **The multi-door review (2026-09-27): one fact, two or more code paths that can answer it differently** *(user: "we are looking for framework level unification and consistency")* — F1–F9 and M1–M5, its full text in the 2026-10-09 archive. Built: F2 F3 (M2f), F4 F5 (M2i), F7, F9, M1–M5, F1 F6's one placement and one record (unit 11a), the bias chain's copy list from the one door (`submit.py:2014-2022`, the warm list's `along` row), F8 (M2j — the Results door reads `jobset_status`, `results.py:198-211`; `every_run`, `runstatus.py:240-275`). **Open: M2k's GPU claim / match half** | `execution/gpu.md` § 1 | the GPU half open — M2k; M2j closed at its review |
| **W40** | execution | **`jobset launch --mode direct` exits 0 when the job it ran failed.** Found 2026-09-27 by M1's fixture: a PySCF run died at activation, the decision ledger recorded `"status": "failed", "returncode": 1`, and the command still exited 0 -- so a script (or a test) that trusts the exit status is told the run succeeded. No contract states launch's exit status for a direct job (`_cli.py:1993-2010` sets none, 2026-10-08) | `execution/running-a-job.md` · `jobset/_cli.py` | **needs a decision** · *Re-checked 2026-10-09:* open — the launch report (`_cli.py:2031-2071`) returns 0 whatever the run's rc, and no contract states the exit status; under the explicitness rule a foreground direct launch exits with its run's status (`--background` reports *started*) — written in `running-a-job.md` first. Its proposed home M2f is done; it goes with the next launch work. |
| **W41** | engines / results | **A SIESTA vibration's finish, the remainder** *(user, 2026-09-28: "siesta only gave some optimization ...")* — the finish built (M2b′, M2b″: the job derives its spectrum, `mb_vibration.pyz`); the measurement and the rulings are in the 2026-10-09 archive. **Left:** the force-constant `.out` titled *"SIESTA optimization"* — the title guessed from the suffix (`lib/inspectors/trajectory.js:108-115`), against `web/results.md` § 0.3; I7's sorted positions; the ⚠ beside an in-tolerance stationarity sentence; Makov–Payne run by hand; the finish parsing the whole output where FC step 0 suffices; PySCF leaving `ir_fd_step_ang` out of its record — written `null` under the explicitness rule. Re-running a failed finish is settled by rule (no recovery machinery; launch again) | `engines/vibration.md` § 5.5, `web/results.md` § 0.3 | open |
| **W42** | engines / science | **FROM A MODE TO THE CURRENT — which vibration to take to transport** *(user, 2026-09-28: "think about this and how the spectrum calculation can be enriched by the additional electronic state/PDOS calculation, and propose an improvement in the vibration calculation contract/document/design and see how we can move forward ... this is after all the other issues are done, so this is just a draft for next step, do not implement")*. The science, `science/normal-modes.md` § 4c (proposed): at a metal the molecule's orbitals are resonances and the junction has a Fermi level, so the current sees `T(E)` near `E_F`, not the HOMO–LUMO gap; a mode acts on a level `ε_r` or on the coupling `Γ` (the contact modes, whose exponential coupling gives a large positive curvature); a DC measurement, thermal or light-driven, sees the second derivative, `⟨ΔG⟩_ν/G = ½ T″σ²/T₀`, and the two scores — the level shift at `σ`, the conductance change — are kept side by side; three frames at the Gauss–Hermite nodes of the thermal distribution give the curvature and the second-order average at once; one transmission at the base screens every mode by the rigid-shift estimate `T″ ≈ g_ε² ∂²T₀/∂E²`; the free atoms' projected DOS and their own states (the extended molecule's MPSH) stand in for HOMO and LUMO. **The design, proposed:** `engines/vibration.md` § 5.10 — ① each mode's character (element shares, the free part's whole-body share, the bond stretches at `σ`), derived at load, no calculation and no new label; ② the charge each mode moves between the held anchor and the free part, from the force-constant run's own displaced SCFs; ③ V1.25's frame set for many modes at the Gauss–Hermite nodes — *made by V1.25's separate procedure, each frame's parameters in the structure's `customized` section (TD10, 2026-09-29)*; ④ an `electronic` kind — one SCF per frame, each job's finish writing `E_F`, the charges, the projected DOS and, with sisl, the levels; ⑤ `summarize` over a frame set, electronic or transport, writing `<label>.mode-response.json` — slope, curvature and thermal average per mode, levels followed by overlap, the rigid-shift estimate beside the explicit value; ⑥ the ranking table in that record's presenter; the PySCF probe on the same nodes (V1.27). Folds in V1.24–V1.27. **Decided with the draft** *(user, 2026-09-28: "the anchor is the fixed atoms. we don't specify the anchor, the atoms is labeled already with fixing or unfixed")*: D1 — the anchor is the held atoms and every projection is over the free atoms, per element, with no new label; D2 — then not needed, the result already carries the free and held atoms. **Decided 2026-09-28, the user's answers:** D3 — yes, the charge each mode moves, *"and result presentation should include the correct data and UI design to present it"*; D4 — yes, an `electronic` kind, *"and make it clear the difference between pySCF and siesta, from setup and calculation and presentation"*; D5 — sisl in every SIESTA env, *"so it is always available when we need them ... fix the install-env and other env related setup to get this consistently done - do not reinvent wheel as we have a full system of install implemented"* (built with the M2b‴ milestone); D7 — the thermal nodes, as recommended (the displaced structures the electronic and transport runs compute on — not the animation, which the user asked about); D8 — the PySCF probe on the same nodes, as recommended. **Taken as recommended unless the user objects:** D6, one displaced-structure set for many modes. **Registered with it:** E2 — a vibrational-only result records no pressure (built with M2b‴); E3 — a command that re-runs a failed finish on its attempt, the user's yes owed. **Order, once agreed:** P0 the decisions; P1 ① (no engine); P2 ② — measured first, whether SIESTA prints the populations at every FC step; P3 ③ with its displacement statement, checked by the frame set's citation promises (W32 ②); P4 ④ and ⑤ for the `electronic` kind — the frame axis for one kind first; P5 transport on the frame set (W32 ③–⑤, behind W32's gate) and ⑤'s transport half; P6 ⑥. Validation H₂, then the carbon-chain junction, then Au–BDT–Au. **Found while drafting, fixed with M2b‴:** the SIESTA deck's commented population hint spelled the retired `WriteMullikenPop` (`siesta/input.py`); it names 5.4.2's `Charge.Mulliken` and `Charge.Mulliken.Format` | `science/normal-modes.md` § 4c · `engines/vibration.md` § 5.10 · `engines/transport.md` § 2a.9 | **draft 2026-09-28 — decided D1–D5, D7, D8; D6 as recommended; after the current work order** · *Re-checked 2026-10-09:* decided, M10; V1.26's decision owed; the frame set it uses is V1.25's — **D6 (*one set for many modes*) reversed 2026-10-09: one mode a set, each its own transport calculation**. |
| **W43** | engines / front end / results | **TRANSPORT — THE ONE LIST (M5).** Transport's open work, consolidated 2026-09-29 from W10, W24, W25, W27, W30, W32, § 5c.3, § 5o, § 5p and `engines/transport.md` § 3.6 / § 3.8.5 after an inventory read each against the code: eleven steps in order, seven decisions that are the user's, the claims the documents make that the code contradicts, and 54 citations of a document that no longer exists *(user: "continue to work on transport as the next item on plan remember to correctly update the plan and consolidate")* | § 5u · `engines/transport.md` § 6.1b | → **§ 5x** (Q14 — transport made whole; its *state at a glance* is the current state) and **§ 5u.1** (the step table); the Au–BDT–Au ladder ran to T(E) on 2026-09-29 and is the acceptance ladder (TD8) *(pointer refreshed 2026-10-08)* · *Re-checked 2026-10-09:* a pointer — § 5u.1. |
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
| ↳ | **A1.4** | **Two silent reads of a broken metadata block.** A malformed annotations channel escapes `StructureCodec.load` as a bare `KeyError('kind')` naming no path (`apply_atom_metadata` the same); and `read_atom_metadata_block` (`script_emit.py:1953`) returns `None` on `JSONDecodeError`, so a corrupted fence reads as *no labels* with nothing said | audit § 1.7, § 1.18 | open · *Re-checked 2026-10-09:* `structure.py:420` raises a bare `KeyError`; `deck_record.read_json_block` (`:128-137`), reached through `script_emit.py:1819-1821`, returns `None` on broken JSON. |
| ↳ | **A1.5** | **The pair's doors re-derive one rule.** Eleven spellings of *"is this a structure path"* (`workingcopy_structure.py`, `files.py:257`, `selection.py:99` — dead, `siesta/input.py:1872,1902`, `build.py:240,722,813`, `cli.py:93,773`, `structure.py:94`) and a fourth in the browser (`task-setup/viewer.js:1144` derives `<stem>.molstruct.json`). Rename strands labels: `water.xyz → notes.txt` leaves `notes.molstruct.json`, and the reverse adopts a foreign sidecar unchecked. A `.pdb` source travels under three names — describe records `c.source.pdb`, the codec writes `c.source.pdb.xyz` (`files()` never passes `fmt`), prep looks for `c.source.xyz` → *"the structure this calculation describes is not here"* | audit §§ 1.8, 1.8a, 1.18 | open · *Re-checked 2026-10-09:* the browser derives the sidecar name itself (`task-setup/viewer.js:1053`); a `.pdb` target is not replaceable (`workingcopy_structure.py:117,330-333`); `files('mol.pdb')` writes `mol.pdb.xyz` (A1.13). |
| ↳ | **A1.6** | **`Structure.replace()` still hand-enumerates.** Nine fields in a literal, five more from `_carry_nonatom()`. Completeness is pinned by a test iterating `dataclasses.fields()` — but its fixture has `annotations={}`, so a `replace()` that drops annotations passes it. Deriving the carried set from `dataclasses.fields()` makes it complete by construction; `frozen_atoms` stays excluded (a property over `regions`, no storage). Read `replace()`, `_carry_nonatom()` and `__post_init__` end to end first — it is the most load-bearing method in the model (`structure.py:1091`, `:1778`; 2026-10-08) | audit § 1.16e; the 2026-09-22 handover | open · *Re-checked 2026-10-09:* `replace()` lists the fields by hand (`structure.py:1272-1336`). |
| ↳ | **A1.8** | **Rules re-derived at the call site.** `estimate_partial_charges` (`chemistry.py:1392`) is label-blind (water `O1,H2,H3` → **0.0 D**; `_DEFAULT_EN = 2.20` is hydrogen's value); **twelve dead `axis_kind` fallbacks** (13 on 2026-10-08) (11 × `("isolated",)*3`, 1 × `()`, all unreachable, plus `validation/siesta.py:734–737` resolving the opposite way — and `transiesta.py`'s was `("periodic",)*3`) — pure deletion, first; the k-sampling hint measures the gap in two frames (hexagonal cell: the hint says ~5.5 Å, the perpendicular gap is 4.16, `_min_image_distance` 6.5) | audit § 1.12a–c | open · *Re-checked 2026-10-09:* twelve dead `axis_kind` fallbacks (e.g. `validation/geometry.py:116`, `validation/__init__.py:115`); `chemistry.py:1392`. |
| ↳ | **A1.9** | **`load()` restamps `schema_version`.** A sidecar written at 7 reads back as 9 through `molstruct.load` — so no reader can say what version a file was. W33 moved the schema to v10 and inherited it: `parse/sidecars/molstruct.py` stamps v10 on every read | audit § 1.13 | open · *Re-checked 2026-10-09:* `_normalised_dict` stamps the current version (`parse/sidecars/molstruct.py:110,236-249`) — v11 now. |
| ↳ | **A1.10** | **Placeholders stored as facts, and two builder defects.** The backbone check keys on `rid − 1`, so a 5P duplex is refused (*"residue 4 O3' → residue 5 P 16.74 Å"*) with the blame on `$X3DNA`; rdkit-added hydrogens carry `(1, MOL, A)` and are persisted as real identity (7 of 12 atoms); `smiles.py:155,187` and `_common.py:35,40` spell index names `C1, O3, H4…` as stated identity, so every SMILES-built molecule ships a sidecar; `_amber.py:80` warns *"requested B-form … not enforced"* on every build. The placeholder must be carried apart from the data — a shape decision before a fix | audit §§ 1.15, 1.18 | open · *Re-checked 2026-10-09:* open. |
| ↳ | **A1.11** | **The ghost element `X` is a legal element.** `resolve_element('X')` → `X`, `atomic_number('X') = 0`, `atomic_mass('X') = 1.0`; it passes `check_species_labels`, contributes Z = 0 to the electron count, and `render_fdf` writes `%block ChemicalSpeciesLabel / 1 0 X` — the defect `chemistry.py`'s own docstring says it exists to end. An unstated rule to state with it: an element denotes Z ≥ 1 | audit § 1.18 | open · *Re-checked 2026-10-09:* `resolve_element` accepts `X` because ASE lists it (`chemistry.py:281-283`). |
| ↳ | **A1.13** | **One rule, several enumerations — and two strictnesses.** ~~Three `_CONTAIN_EPS`/`_EPS` (one unread); `cell._contains` re-implements `Structure.cell_contains_atoms`~~ — gone with W33 (`6c705058`): containment is one distance tolerance, the hand-off's; `affine`/`concat` hand-list the columns; `vacuum=-5` is accepted by the model and refused by the gate; `annotations` accepts a `str` index `regions` refuses; `set_channel` installs then validates, so a refused channel stays and breaks `copy()`; `files('mol.pdb')` writes an XYZ inside; `apply_to_structure` on a partial payload resets the cell. Fix each at its owner, never at the instance | audit §§ 3, 4 | open · *Re-checked 2026-10-09:* open — `files('mol.pdb')` writes `mol.pdb.xyz`. |
| ↳ | **A1.14** | **The same shape, latent** (unreachable or harmless today). A second PDB reader (`builders/backends/_common.py:57–93`: `Mg → M`, `Cl → C` on a blank element column); a second deserialiser (`selection.py:131–173`; `_shared.py:124–165, 1349–1406`); eight metadata-dropping rebuilds (`add_hydrogens`, `protonate_phosphate_oxygens`, `_drop_overlapping_hydrogens`, `relieve_clashes`, `_strip_5prime_phosphate`, `select_chain`, `_patch_residue`, `_fix_methylene_hydrogens`); `describe.write_description(struct=None)`; `_reset_to_derived`'s own 1e-6 threshold; `_validate_transport_kind` reading the raw cell; `transiesta.py:601,705` bare `+ 1` into the deck; `pyscf/input.py:1575` serialising with `json.dump`; `chemistry.py`'s `_adjacency` on raw `"H"` and a second periodic table; `emit_atom_metadata` dropping a kind-less channel silently; `build.py:410/1296`; the backend set spelled in seven places | audit § 1.18 | open — its transport half is the electrode block's hand-written `+1` (`transiesta.emit_electrode_declarations`, `:432`); § 5r's order *(2026-09-29: not done by § 5u step 1, which had claimed it)* · *Re-checked 2026-10-09:* the transport `+1` sits at `transiesta.py:500,580`. |
| ↳ | **A1.15** | **Rules nobody wrote down** — the stale comments are fixed (`structure.py:5-16` and `workingcopy_structure.py:26-32` read right on 2026-10-08; `structure.py:892` and `validation/sidecar.py:3–7` not re-read). To state: what the CLI's single stdout stream carries; `cell: null` in a load body; a partial `apply_to_structure` payload; a `.XV`'s companion sidecar; whether *Delete file* pairs the sidecar; `Frame.lattice` against `Structure.cell` (which frame a run artifact is in is W33's) | audit § 1.18 | open · *Re-checked 2026-10-09:* the single-stdout-stream item is answered by the 2026-10-09 ruling (a lone file is read as atoms and says so, `cli.py:421-424`); `Frame` still exists (`frame.py:64`). |
| ↳ | **A1.17** | **The tests: the count must come down, and three rules are unpinned.** Unpinned: `write(struct, "x.pdb")` producing a readable pair (restoring the pre-2026-09-07 bug leaves 460 passed); geometry-before-sidecar, both-or-neither (reversed → 441 passed); the explicit-cell centring branch (deleted → 441 passed — W33 makes it the rule and pins it). Blind or shape-asserting: ~~`test_periodicity_gate.py:1126` (inverted — fails on cosmetics, passes on deletion), `test_cell.py:202` (`"a " in message`)~~, `test_cell.py:588` (a signature), ~~`TestDocMatchesTheDoor` (blind to a shrinking `OPS`)~~ — the struck three fixed or retired with W33 (`6c705058`), which also pins the explicit-cell centring branch as the rule (T5, T1). ~~~24 duplicates in twelve clusters — 16–17 tests carry *"the default isolated vacuum gap is 3 Å"*~~ (gone, 2026-10-08). `structure_hash` is still hand-built in test fixtures (18 test files name it on 2026-10-08; 15 fixtures in September), three matching a different error than they name | audit § 5a | open · *Re-checked 2026-10-09:* 19 test files name `structure_hash`; the front-end half of the atom-number invariant (`model/overview.md`; `structure-annotations.md` § 6 now names the real homes, `_atom.js` `toDisplay`) is this row. |
| ↳ | **A1.18** | **The contracts' line numbers are 15 % right.** 6 of 39 `file.py:NNN` references resolve (re-derived exhaustively, 2026-09-22); seven symbol names have never existed; five retired concepts are written as current. `model/parse.md:355` already states the rule — a line number is a pin; the fix is one mechanical pass, then the behavioural list | audit § 2 | open · *Re-checked 2026-10-09:* open. |
| ↳ | **A1.19** | **The test harness can report a green suite that is not green.** In `tools/`, open and not held: ~~`run lf` with nothing to rerun shouts NOT GREEN (exit 5 on a green last run — `exit 4 \| 0/0 ran` appears 302 times in history, so the guard gets trained away)~~ — **closed 2026-09-30** with the spread runs' review: `run lf` runs only when pytest's last-failed list names a test file here, and otherwise says *nothing to rerun* and exits 0 (`testing.md` § 6.1a); ~~`testrun.py failed` emits node-ids with a `[teardown]` suffix pytest refuses; the head line stops summing (`2/2 ran \| pass 1 FAIL 3`)~~ — fixed (`testrun.py:319`, `:234`; 2026-10-08). Same class, open: `progress_plugin.py:68`'s `except OSError` silently disables the writer and `cmd_status` returns 0 for no-data; the env canary's *DISARMED* goes through `warnings.warn` with no hook, so a run whose canary proved nothing reads like one that proved everything | audit § 0c | open · *Re-checked 2026-10-09:* open — `tools/progress_plugin.py`'s `except OSError: pass`; `conftest.py:401`. |
| ↳ | **A1.20** | **Residue, with the step-0 read done** (`process/code-audit.md` § 1d): ~~five `_enumerate_files` buckets with no reader, built on the Watch polling path with four directory scans~~ (gone, 2026-10-08); three legacy shim classes and the five test-only names that depend on them (one decision); `sha256_of_file` (`sidecars/molstruct.py:220`), whose docstring calls it the `structure_hash` pin (X4 ⑤); `selection_rules` (`script_emit.py:391-448`), a format field with no producer and no consumer; `sidecars.molstruct.load_text` (×2 on 2026-10-08 — re-read before acting); ~~the `*-electrode` convention two modules advertised and `sort.PARTITION_LABELS` could not compose~~ (gone 2026-10-02, § 0b item 3). Last in § 5r's order | audit § 5 | open · *Re-checked 2026-10-09:* open — `sidecars/molstruct.py:230,706`. |
| ↳ | **A1.21** | **Audit #2 — the rest of the tree, planned and not started.** ~70,000 of ~110,000 lines were outside the audit: `web/static/lib/` (33k — a language boundary), `jobset/` (12k — a process boundary), the rest of `web/blueprints/` (~9k), `runwrap.py` (5k), `runtime_config`/`template`/`monitor`/`checkpoint`/`task` (~9k), `config/` + `validation/` (~8k). Kept separate because the failure shape differs (one fact per side of a boundary, not one door per operation) and so does the evidence (a JS finding needs no Python fixture; a wrapper one needs a submitted job) | audit § 7c | planned · *Re-checked 2026-10-09:* open, with A14: the 44 hand-built run-file names counted 2026-09-07 predate `runfiles.RunNames` — re-measure by review. |
| **S1** | architecture seams | **`runwrap` reaches into the engines.** The wrapper writer branches on which engine it is writing for — what a cold restart clears, how the label is read back out of a deck, how the launch line is formed. Until it moves, *adding an engine edits `runwrap.py`*, which is exactly what `generator.md` § 7's *"adding an engine adds files and edits none"* exists to catch | `backend-architecture.md` § 5 (its **W1**) | **measured open** — three engine branches in `runwrap.py` and 22 lines carrying an engine-name literal (2026-10-08; four branches and 128 literals when last counted, 2026-09-06) · *Re-checked 2026-10-13:* conditions on an engine at `runwrap.py:592`, `:772`/`:816`, `:1914`, `:1962`, `:2116`, `:3171`, `:3183`, `:3432`, `:3920`; 22 lines carry an engine-name literal. Closed by the Python pass's phase 4 as proposed (Q19: one wrapper hook per engine on `jobset/engines.engine_seam`), when the user takes it. |
| **S13** | architecture seams | **Transport convergence sweep** — auto-vary transverse-k / `MeshCutoff` / electrode thickness and report where `T(E_F)` stops moving. `transport.md` § 2 already tells a reader not to trust a single point blindly, so the document promises what the code does not offer | `engines/transport.md` § 8 | **measured: not built, re-verified 2026-09-20.** Nothing in the tree names it at all now — even the `transport/wizard.py` comment that used to is gone, so the only record that it is owed is `engines/transport.md` § 8 and this row · **→ after § 5u step 4, unscheduled** (§ 5u.5: a sweep needs a ladder that has produced a curve) · *Re-checked 2026-10-09:* open; `transport.md` § 8 now cites this row. |
| **N10** | parse / front end | **A CALCULATION ROOT IS NOT A RUN DIRECTORY, and the Results tab has only one notion.** Measured on the real transport ladder: `jobset_status` answers 5 stages all `pending, prepped, not launched`; `run_status` -- which `/api/results/dir` calls unconditionally -- answers `running, no result file yet`, so the tab tells a person a calculation nobody launched is running. The tab offers that root its own INPUT structure as the result, and says nothing about the five stages. `transport.md` § 2a.12 has required the ladder's state, the curve with its treatment named, and the provenance chain since before the surface was built. **The predicate already exists** -- `checkpoint._is_bundle_root` -- and a second copy in `parse/dirs` would be instance 14 of § 8 | § 5c.3, `transport.md` § 2a.12 | **RE-MEASURED 2026-09-23 — the headline defect is CLOSED; (e) folded into § 5u step 9 (2026-10-08); the two-predicate question open.** § 5c.3's two premises are both stale: `/api/results/dir` no longer calls `run_status` unconditionally (`results.py:185-229` asks `calcdirs` first — `record` / `read` / `root_of` — and builds the ladder there; a container gets `status: None`), and the public owner that landed is a NEW module, `calcdirs`, **not** the `checkpoint._is_bundle_root` step (a) proposed — which is still private at `checkpoint.py:379`. So (a) is superseded, (b) is done in substance but via `place == CONTAINER` rather than a `jobset_status` ladder (`jobset_status` has zero hits in `results.py`), (f) is DONE (`inspectors/transport.js:180-205` renders the provenance chain), and **(c) the payload and (d) the ladder view are DONE 2026-09-24** (`ladder` on `/api/results/dir`, the empty-state card's table, `results.md` § 2.4); **(e) the inspector's parser → § 5u step 9** — `transport.js` no longer calls `JSON.parse` (2026-10-08); what remains of the reader is step 9's. **AND A NEW, SMALLER QUESTION:** there are now TWO predicates for *is this a calculation root* on DIFFERENT evidence — `_is_bundle_root` tests for `task.json`/`job-set.json` existing, `container_or_run` reads `task.json`'s `shape`. The § 8 duplication § 5c.3 warned about is real in a milder form, found by a browser walk that 2,800 passing tests missed · *Re-checked 2026-10-09:* `container_or_run` is gone (`runs.place_of`, `runs.py:56`); two root checks remain on different evidence — `checkpoint._is_bundle_root` (`:379-382`, task.json or job-set.json) and `calcdirs.root_of` (`:109-130`, task.json). No failure on molbuilder's road (job-set.json always sits beside task.json): the unification only. |
| **W24** | front end / transport | → § 5u.1 step 8 (one panel per engine) | — | pointer |
| **W25** | engines / transport | → § 5u.1 step 7 (the TranSIESTA / TBtrans items still unsaid) | — | pointer |
| **W30** | front end / engines | **The transport parameter surface** (`engines/transport.md` § 3.8; its full text in the 2026-10-09 archive) — ① ② built 2026-09-24; ③'s remainder → § 5u.1 step 10; ⑤ → step 8; **④ the deck viewer — built 2026-10-11, for every calculation** *(user: "we should expand this to other tasks as well where we may have different stages and different runs")*: the Results tab's ladder gives each run — each attempt, frame and voltage — an *input* and a *check report* button, opening the deck it read and the `.validation.txt` prep wrote, in the sidebar's file viewer (`web/results.md` § 2.4; `runs.Run.prepared` / `Run.report`) | `engines/transport.md` § 3.8 | ④ **done** |
| **F9** | execution | **A group prep prints its pseudopotential copy lines and the `geometry.h_ratio` note (a warning until 2026-10-11) once per rung** (three times each) — found on the road walk 2026-10-08, not reached by B8 | § 5x.7 F9 | open · *Re-checked 2026-10-09:* `prep_group` (`prep.py:2970`) does not de-duplicate the members' notes; its h_ratio half is A1.12. |
| **F11** | execution | **The wrapper's kill line is unseen on a killed point**, and two more mechanisms have not been seen on the road: a **cold sweep** (`launch task --stage device --cold` on a finished sweep) and the **take-over hop** (a point taken over from a run that took it over). One walk closes all three — ~45 minutes of this machine (a device point ≈ 15 min), launched through Task setup and left to run; **waits for the user's go** | § 5x.7 | open — the user's · *Re-checked 2026-10-09:* the cold sweep is seen on Q15's run E (`tests/test_transport_on_a_real_junction_e2e.py:238-240, 396-409`); the kill line and the take-over hop are not. |
| **F14** | parse | **The unconstrained max force and a periodic slab's pressure are never shown** (the monitor and wrapper print the constrained max); TranSIESTA's *FORCES WRONG* is not captured | § 5x.7 F14 | open · *Re-checked 2026-10-09:* no `FORCES WRONG` capture anywhere; `report_fields.py:58-61` offers `max_force` to a TranSIESTA device (§ 5t.4). |
| **F19b** | execution | **A retry's `exec` inherits the first try's tee**, so run0's session log holds run1's session too; the tbtrans wrapper says *Retry policy: up to 1 retry on non-convergence* for a program that converges nothing | § 5x.7 F19 | open · *Re-checked 2026-10-09:* open. |
| **F-MDNC** | parse | **A flat stage launched again shares SIESTA's history file with the run before it**: after a second run in a flat folder `<label>.MD.nc` holds more rows than that run wrote (12 rows, where each run's output has 8 frames; measured on the end-to-end flat calculation 2026-10-08), and pairing the second run's output with it (`siesta_mdnc.align_to_reference`) matched rows 0–5 and then row 11 -- the earlier run's rows first. Whether the reader takes the newest run's rows, or a run that starts over sets the old history aside, is a design question. *(2026-10-10, the Q17-c review: the transport citation of a flat relaxation reads the same output and history — and a flat root speaks for its highest stage launched, so an earlier stage launched again after a later one leaves the later stage's output beside the earlier's history.)* | § 5y, the flat module's second launch | open · *Re-checked 2026-10-09:* open. |
| **T-ENV** | tests | **Installer tests answer the environment manager in-process**: `tests/test_envs_one_answer_about_an_env.py` (H6, `dispatch_into_env` replaced to return a typed PySCF verify answer) and the stubs `tests/test_envs_install.py` writes -- a command's answer swapped in-process, which tier 1 forbids (`testing.md` § 0); judged under § 5y's rule in their own pass | § 5y | open · *Re-checked 2026-10-09:* open (`tests/test_envs_one_answer_about_an_env.py:182`). |
| **PYF-T** | code hygiene | **The tests, `tools/` and `scripts/` are not pyflakes-clean**: 56 findings — 52 in tests, 3 in `tools/`, 1 in `scripts/` (47 unused imports, 7 unused locals, 2 f-strings without a placeholder), measured 2026-10-13 with `python -m pyflakes $(git ls-files '*.py' ':!:molbuilder/*' ':!:tests/data/*')` — the package is clean (`conventions.md` § 1) | Q19's Python P1 | open — each read before it is removed: an unused import in a test can be a fixture pulled in by name |
| **L2** | tests | **`tests/field/test_ask_the_target.py` fails instead of skipping when invoked outside its batch** (`. does not read as a machine record`: the record path is unset) — a field test gates itself on its backend (`testing.md` § 3) | B8 round 2 | open · *Re-checked 2026-10-09:* open — `tests/field/test_ask_the_target.py:49-51` asserts instead of skipping. |
| **W56-2** | transport / citation | **The citation reads a cited run beside the run door.** `compose.py:252-262` refuses a folder holding more than one `.fdf`, so a flat relaxation of several stages — which jobset makes — cannot be cited, and its remedy (*"Keep one deck"*) tells the person to edit a run folder; the labels come from that deck's block or a lone `sidecars_in` file (`:445-468`), not from `runs.declared(run)` (`runs.py:308`), the one reader of a run | `engines/transport.md` § 3.1 | **built 2026-10-09 with Q17-c**: a cited folder is the run it speaks for (`runs.run_of`), its deck `Run.deck`, its geometry `runs.openable` and `view_of`, its labels `runs.declared` — the one-`.fdf` refusal and the sidecar-beside fallback gone (`compose._classify_run`, `labeled_citation_structure`) |
| **EL-W** | transport / labels | **An electrode label that is not frozen is a warning** (`validation/sidecar.py:107`); making it an error is one word (§ 5p.3g, parked 2026-09-17) | `engines/transport.md` § 4 | the user's word |
| **R11** | execution / tests | **Launch's test-only renderer fallback.** Three test files call the wrapper renderer; two rely on its fallback, and both hand-write `environment.json` — a hand-built record `testing.md:54-55` retires: `tests/test_launch_door_gate.py:37-46`, `tests/test_warm_file_inventory.py:30-55`. Then `machine_record` becomes required in `render_run_wrapper` and the fallback goes (`runwrap.py:2944-2945`, `:2968-2970`); launch's own (`_render_sbatch_for`, `runwrap.py:4224-4228`) stays | § 0b (W54) | open — judge the two tests first |
| **W57-G** | execution / tests | **Rules no test holds since the W57 retirements**: a killed run leaves no conclusion marker (an engine-tier kill); a stage's `overrides` (no road key for it); the PySCF effective-parameters fence (an engine-tier assertion on the H₂ log); `set -e` across the preamble; a warm retry's own launch record (`retry_of`, 0 test hits); MPS and ranks-per-GPU (only the `run_sh_lacks` side, `gpu_contract.toml:188`) | W57 (2026-10-07) | open — each an assertion on a real run of M2n's batch, never a fake |
| **W57-M** | execution / PySCF | **A stated PySCF memory cap reaching `max_memory`** — only the unset rows exist (`launch_values.toml:206-216`); none for a stated cap reaching `max_memory = N` (`pyscf/input.py:494`) | W57 | open |
| **W57-T** | tests | **Tests built on hand-made inputs, for retirement**: `tests/parse/test_tbtrans_out.py` writes empty files named by TBtrans's rule (`:13-22`); `tests/test_jobset.py` builds job sets by hand (`:56-59`, `:107-115`) — `testing.md:50-55` retires both, never sweeps them; and the ten test files that render through `spec_for` + `render_deck` directly, swept to prep's road | W57 | open |
| **W57-O** | execution / records | **Old-data tolerances**, each an explicit error under *build molbuilder, don't clean up handcrafted input*: `record.py:1323-1324` skips the conflict check when a copy names no machine; `script_generation` refused by name in three doors (`runtime_config.py:705-715,958`; `jobset/machine.py:137`; `envs/initconfig.py:253-259`) — one door; `Environment.from_dict` drops unknown keys (`record.py:363-398`) | W57 | open |
| **W57-K** | execution / library | **A kind left to a default.** Fifteen functions default their kind to `"optimization"` — `template.py:1003`, `siesta/input.py:555`, `script_emit.py:1043`, `pyscf/stages.py:33`, `validation/siesta.py:648`, `validation/__init__.py:122`, `validation/stages.py:49,117`, `pyscf/input.py:171`, `_shared.py:540`, `siesta/stages.py:91`, `kmesh.py:130`, `validation/siesta.py:197`, `validation/pyscf.py:166`, `_shared.py:711` — every caller on the road states it; ~150 calls, most in tests, lean on the default. And `template.MIGRATE_HINT` (`template.py:2290-2293`) names `--bundle` without a complete command | W57 (explicitness) | open |
| **W56-H** | tests / PySCF | **A PySCF relaxation with a held atom, end to end** — its progress log's `# frozen_atoms` line and a trajectory read with the atom held; only the vibration e2e holds one today (`test_vibration_e2e.py:501-537`) | W56 | open — M2n's batch |
| **PL-R** | execution / layout | **`project-layout.md`'s owed rulings**: *must every stage be measured, and how is a verdict whose environment changed shown* — the user's; `summarize` reporting the latest of two runs without saying so (`:687-693`) — a defect under the explicitness rule, fixed, not ruled. (*May one folder hold two ladders* is no question: one `task.json` per folder) | `execution/project-layout.md` | the first the user's; the second open |
| **DOC-TOC** | tests / docs | **A test writes into the checkout when a document is new.** `tests/test_docs_tab.py::test_toc_returns_each_document_once` asks `/api/docs/toc` over the real `docs/`, and the toc builder persists an unlisted document into `docs/toc.json` (`web/blueprints/docs.py:329-345`) — the suite's checkout guard (`conftest.the_suite_leaves_your_checkout_alone`) fired on it 2026-10-09, when this consolidation added its archive. The same rule is held on an isolated tree by `:200` and `:249` | found 2026-10-09 | open — judged redundant: retire it |
| **Q11** | execution / scheduler | → the queue, Q11 (the named-queue check in two doors) | — | pointer |
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
hand-built `structure_hash` fixtures (A1.17) agree with the writer only
because the gate is loose — convert each as its file is touched for another
reason.

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


## 5u. Transport — the one list (M5, W43) *(consolidated 2026-09-29)*

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
| **14** | **Groups are stages' numbers; prep writes every group; launch gathers** *(user, 2026-10-11, after the browser walk found the page treating a group's one job as three stages: "the prep UI clearly still treats each stage separately ... steps that are beyond logic"; "the checkbox for grouping is awkward: it cannot assign different groups"; "prep should be able to prepare all groups at the same time ... that's the launch status, not the prep")*. **Ruled 2026-10-11, the four questions as put**: ① each stage states a **group number** in `task.json` (`stages[i].group`), edited on Task setup's Stages card: the first stage is group 1, each later one repeats the number above it or adds one; a stage may not repeat the number of a group holding a stage it builds on; the defaults repeat where allowed (transport 1 1 1 2 3); ② one tab per group on Task setup, holding the group's one run card — what the job shares (ranks, cores per rank, GPU use and count, queue) written to every member's `execution`; the wall per stage, the job asking their sum; ③ **one prep prepares every group** — each group one job — and **the gather moves to launch**: `launch task` sends the next group, first checking each stage's upstream newest run finished and ran the deck described now, copying the inputs with their `.gathered-from`, refusing before anything is sent when not; ④ a stage **prepared and never launched may be prepared again** (no run's record to falsify). **The design, traced and reviewed: § 5aa** — V settled (the force-constant stage keeps its geometry in its deck), R answered by a save before a group's first launch (put to the user with the contract). Contract first: `job-system.md` (*The task*, the states, § 5.0, § 5.4, § 6.0), `project-layout.md` § 1.6.6, `stages.md` (the field), `transport.md` § 1.1, `task-setup.md` (§ 5.3, § 11, the run card), `web-api.md`; then code; then the user's `lowbias` setup proved in the browser (this machine; a Sol preview) | the walk's grouped-prep refusal on `AuBDTAu/transport/lowbias` (2026-10-11: the job's resources stated on the seed only) | proved through the browser on the user's setup, two review rounds |

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


## 5aa. Groups, every group prepared at once, and a run's inputs taken at launch — the design *(plan § 5u.1 row 14; user, 2026-10-11; two review rounds 2026-10-11, revision 3)*

> *"we want to combine certain stages ... as one grouped job ... but the prep
> UI clearly still treats each stage separately"* · *"the checkbox for grouping
> is awkward: it cannot assign different groups"* · *"the prep should be able
> to prepare all groups at the same time ... that's the launch status, not the
> prep"* · *"the stages should have a group number assigned to it, either
> repeat the last group number or add one"* · *"consolidate such this is a
> framework that would work for other tasks as well - we don't need a patch"* ·
> *"think hard to find the best balance between flexibility, logical, and
> cleanness"*.

**Ruled 2026-10-11** (the four questions as put): a stage's group number lives
in `task.json`; a group's wall is its members' walls summed; the upstream files
are copied **at launch**; a stage prepared and **never launched** may be
prepared again. **Settled by the trace:** V (§ 5aa.4 — option (a) withdrawn)
and R (§ 5aa.5 — option (i) withdrawn; the answer, a save before a group's
first launch, is put to the user with the contract).

### 5aa.1 The model

* **A group is a job.** Each stage states a number; the stages with one number
  are one job, launched together; the numbers run in ladder order.
* **Two kinds of hand-over, told apart by what the next stage's deck needs.**
  - **Files** — the next run reads them when it starts: an optimization
    ladder's `.XV`/`.DM`/history (SIESTA) or `_optimized.xyz`/`.chk` (PySCF);
    transport's `.DM`, the leads' `.TSHS`, the device's `.TS.HSX`. The deck
    does not depend on them, so **it is prepared any time, and the files are
    taken when the run is launched**.
  - **Values written into the deck** — one case: a SIESTA force-constant
    stage, whose deck holds `relax`'s relaxed coordinates and cell (§ 5aa.4).
    **It is prepared once `relax` has finished**, its source run decided at its
    prep; its file hand-over (relax's `.DM`, read under `DM.UseSaveDM .true.`)
    is copied at its first launch **from that same run**.
* **Prep makes what a stage IS** — deck, run script, header, job row — and
  reads another stage's run only for a value hand-over.
* **Launch makes a RUN and fills it with what it starts from** — the file
  hand-overs, decided at the launch that uses them, recorded in the run.
  **A group's first launch saves the folder's state first**, as prep does
  before it writes: the state a redo of that group restores — every run
  upstream already in it (§ 5aa.5).
* **Nothing starts a stage but you, and you still look first** (`job-system.md`
  § 5.4, *the gap is the feature*): the gap moves from before prep to before
  launch, where the run is spent.

### 5aa.2 What the code does today (traced and cross-checked 2026-10-11; file:line at that date)

| fact | where |
|---|---|
| a stage is **ready** to prep only when what it builds on has a newest run that finished — transport by `gather_sources`, every other kind by `continuation_answer`; else **waiting** | `jobset/ready.py:60-95` |
| prep opens `run-0` with the stage's own files (deck, run script, header, pseudos, `atom-permutation.json`, bundles) and **copies in what it continues from** (`warm_carry`, `.continued-from`) or **gathers** (transport, `.gathered-from`), per point for a swept rung | `prep._open_attempts` 1327-1353 → `materialize.prepare_attempt` 678-812 (`bring_files` 591-633, carry 783-790); `prep.transport_inputs` 1766-1930, `_gather_into` 1933-1940; `open_sweep_run` (materialize 648-675) |
| **every run folder holds its own copy of the deck and run script** | `materialize.bring_files` 591-633 (seen: `01_seed/run-0/e2e-walk_01_seed.fdf`) |
| the first launch **runs in prep's `run-0`** (an unlaunched last attempt is reused) | `materialize.resolve_attempt` 500-521; `submit._plan_member` 748-752 |
| a relaunch opens `run-<n+1>`: warm from its own latest run when the stage continues (`Job.relaunch_continues`: not a force-constant stage, not `restart: clean`), else empty; cold from none; a single-run transport rung **re-copies the inputs its prep gathered** from the run `.gathered-from` names (refused when one is gone), a swept one from its previous run's copies | `continuation.relaunch` 501-537; `submit._plan_member` 753-778, `_carry_the_gather` 799-827, `_carry_sweep_gather` 2159-2182 |
| `--from` exists on **prep only**; `launch --cold` on a stage never launched is refused | `_cli.py:1083`; `submit._refuse_cold_first` 632-644 |
| launch checks **no upstream run's state** (prep does) | prep: `continuation._by_default` 468-498, `prep.transport_inputs` |
| **an earlier run's facts are read from the stage's one job row and today's description, not from the run**: the optimizer a `.CG` may follow (`warm_carry` against the job row), a swept rung's points (today's bias list, the frame record) | `model.warm_carry` 453-490; `materialize._source_job`; `transport/stages.points_in` 247-252; `runstatus._sweep_status`; `transport/record.py:352` |
| prep places each member of a group alone and renders the header with the summed wall without admitting the sum; launch admits it | `prep.prep_group` 3141-3168; `submit._sbatch_request` 270-275 |
| the printed ways back name *"the state saved before X's prep"* (`commands.rollback`); `continuation._by_default` prints `prep … --cold` / `--from` | `submit.py:494, 642, 730, 822, 2003`; `prep.py:1824, 1922`; `continuation.py:444, 495` |
| launch writes its planned files **before** it sends (`carry_out`), so a refused `sbatch` leaves them | `submit.py:1161` |
| a group: several `--stage` at one prep — any pick none of whose stages builds on another and that share one allocation (`envelope`: queue, ranks, cores per rank, GPUs, GPU use, binding the same; wall summed; memory largest); `Job.group` (names) + `launch/<g>.sbatch`. **Stages named together at launch also go as one group**, a header rendered then | `prep.prep_group` 3080-3224; `group.py` 36-112; `submit._plan_group` 2252-2357 |
| with nothing named, `launch task` offers a relaunch **only when nothing prepared is waiting** | `_cli._launch_unit` 1211-1261 (1238-1247) |
| "prepared" = the stage has a row in `job-set.json`; prepared again is refused, naming the rollback to the state before its prep | `prep.prepared_already` 2468-2507; checkpoint 2a 2696-2702 |
| a checkpoint restore **overwrites the folder**: runs made after the state are gone unless saved | `checkpointing.md` § 7 (A5) |
| the SIESTA force-constant deck is written **at `relax`'s relaxed coordinates and cell**, read at prep from relax's output, with the relaxation record in its `vibration` block | `prep._vibration_stage_geometry` 349-469; `_vibration_block` 318-346 |
| it writes `MD.UseSaveXV .false.` because **a force-constant run writes its LAST DISPLACEMENT to `<label>.XV`** (measured 2026-09-23) — a deck reading it would take a nudged geometry as the stationary point; a re-run restarts at `FC.First` (`[vibration] resumes = false`) | `siesta/vibration_deck.py:108-127`; `siesta/warm-files.toml:155-170` |
| the vibration finish reads the reference geometry from FC step 0 of its own output, the cell from the deck, masses, criterion and relaxation record from the deck's block | `spectra/siesta_vibration.py:228-255, 286-308` |
| a transport frame count is read from the composition record on disk — "no record yet means nothing is prepared: 1" | `transport/stages.frames_of` 203-212 |
| transport's fourth gate compares the upstream's **ran** deck with what the upstream renders **now**; nothing compares the downstream's own prepared deck with now | `prep.transport_inputs` 1886-1924 |
| a benchmark is its own lane: trials opened cold, nothing copied in; a force-constant stage's trials written at relax's geometry read at `prep bench` | `prep.py:2805-2823, 650-654`; `submit._walk_of` 1791-1797 |
| every file hand-over is read at run start — SIESTA `.XV` (`MD.UseSaveXV`), `.DM`, `.CG` for the same optimizer; its history (`.MD.nc`, `.MD`, `.MDE`, `.ANI`) is appended to, never read; PySCF `_optimized.xyz` and `.chk` when the stage continues; transport `.DM`, `.TSHS`, `.TS.HSX` | `siesta/warm-files.toml:45-115`; `pyscf/input.py:437-460`; `transport/stages.stage_inputs` 255-290 |
| a task's stage has `name`, `overrides`, `execution` (the run card, which holds `restart`); `task.py` is the stdlib-only description layer | `task.py:17-31, 306-331, 1138-1187` |

### 5aa.3 Every situation, and what the design does

**Kinds and shapes.**

| # | situation | today | the design |
|---|---|---|---|
| K1 | SIESTA optimization ladder, hierarchical | prep medium only after coarse finished; copies `.XV`/`.DM`/history (+`.CG`, same optimizer) into `run-0` | defaults 1, 2, 3; prep writes all; the first launch of medium takes coarse's newest run, which must have finished (D5: an older one never stands in), and copies the same files |
| K2 | same, flat | nothing copied; prep requires the stage before finished and writes the flat `.continued-from` | prep writes all; the first launch requires the stage before's latest run finished and writes the flat `.continued-from` (launch already does on a flat relaunch) |
| K3 | a stage set `restart: clean` | builds on nothing | may share the group above; nothing copied |
| K4 | PySCF ladder | `.chk`, `_optimized.xyz` | as K1 |
| K5 | SIESTA vibration, relax → freq | freq prepped after relax finished; deck at relax's geometry; relax's restart files copied into `run-0` at prep (`.DM` read) | freq is preparable once relax has finished, its source run decided at its prep (`--from` a relax run); prep-all prepares the groups before it and says freq's group waits for relax before it can be prepared; its first launch copies relax's files **from the run its prep took** (recorded on its job), never the newest |
| K6 | SIESTA vibration stated relaxed (freq only) | the structure as given | one group, nothing upstream |
| K7 | SIESTA displacement sweep | each FC stage after relax; flat refused | defaults 1, 2, 2 …; each FC stage as K5 |
| K8 | PySCF vibration | one rung | unchanged |
| K9 | transport | seed + leads one group; device after they finish (gathering); transmission after the device | defaults 1 1 1 2 3; prep writes all five; the first launch of group 2 gathers (the three gates, `transport.md` § 6.1: prepared · its newest run finished, every point done, and the deck it renders now · the file there) into `run-0`, per point for a swept rung |
| K10 | transport without the seed | | ~~defaults 1 1 2 3~~ — withdrawn 2026-10-12: every ladder keeps its seed (`task.fixed_by_kind`) |
| K11 | transport bias sweep (self-consistent) | per point gathers at prep | per point at the first launch; the walk and a relaunch's point takeover unchanged |
| K12 | transport over a frame set | frame × voltage points; the frame count read from disk | **the frame count comes from the junction composed in the same prep** (the plan holds the composition record before any member is planned), never from disk — else a prep-all plans the device and transmission with one frame |
| K13 | benchmark | its own lane | unchanged and outside groups; a force-constant stage's benchmark still waits for relax (K5). **Found, to verify by a test before any fix (separate item):** a force-constant trial's deck keeps the kind's `DM.UseSaveDM .true.`, which `agreement.check_trial_starts_cold` refuses at `launch bench` |

**Groups.**

| # | situation | the design |
|---|---|---|
| G1 | the rule | the first stage is group 1; each later stage repeats the number above it or adds one; a stage may not repeat the number of a group holding a stage it builds on (`group.upstream_of`). **Where checked:** the reader (`task.py`, stdlib) checks the numbers' shape — whole numbers from 1, each the one above or one more; the building-on rule needs the template and the engine, so it is prep's preflight (`stages.md` § 6.6) and the folder route's answer to the page, said beside the row |
| G2 | defaults for a new description (`jobset init`, the hand-over's Save, the transport describe door) | repeat where allowed — transport 1 1 1 2 3, an optimization ladder 1 2 3, clean stages 1 1 1 — written before any run card exists |
| G3 | one job's allocation | as `envelope` today; the group's run card writes each shared value into every member's `execution` **when you set it**; members that already disagree are shown disagreeing, never overwritten unasked, and prep's `envelope` still refuses a disagreement by name |
| G4 | numbers changed for groups prepared and never launched | **groups are compared by membership** (`Job.group`, the names), never by number — numbers are positions, and removing a stage shifts them. A prep whose picked group's membership changed, or that touches a prepared group whose membership changed, forms every such job again in the same prep, the stale header removed; `prep --group N` naming only some of them is refused, naming the others (nothing prepared unseen) |
| G5 | a launched stage's group membership changed | refused at prep (it reads the run records): a launched stage's group is fixed, as its deck is; a renumbering alone (a stage removed above it) is no change |
| G6 | relaunch | a member alone (D4), or its whole group; **stages named together at launch no longer form a group** — a group is its number (`submit._plan_group`'s launch-time header retired) |
| G7 | a launch whose prepared jobs' memberships disagree with the description's groups | refused, naming the prep that forms the jobs again (G4) |
| G8 | a group's job admitted | prep admits the **group's** request — the summed wall, the largest memory, the shared ranks/cores/GPUs — against the queue it names, as a run's is (`placement.admitted`), so a prep-all that passes is one launch takes (R9) |
| G9 | the first launch of one member of a group of several | refused: a group goes whole the first time; a member alone is a relaunch (D4) |

**Prep.**

| # | situation | the design |
|---|---|---|
| P1 | `prep task` with nothing named | shows the groups, asks once — every group not prepared and preparable pre-selected; one save; each group one job. With nobody to ask, refused, naming `prep task --group N …` — nothing prepared unseen |
| P2 | `prep task --group N` (repeatable) | those groups |
| P3 | what may be prepared | every described stage, except a stage with a value hand-over (K5) whose upstream has not finished; the description's checks, the machine, the placement and every deck's gates as today |
| P4 | a stage prepared and never launched, prepared again (ruled ④) | allowed — its job row replaced, its deck/scripts/header rendered again, its `run-0` reused and emptied of anything an earlier prep or a refused send put there; the folder saved first. **A group is prepared again whole.** Reverses 2026-10-02's *"refuse it, redo via rollback"* for a stage never launched |
| P5 | a launched stage, prepared again | refused, as today — earlier runs' facts are read from the stage's job row and the description (§ 5aa.2), so its runs must keep the deck and row they ran with; the way back is § 5aa.5's |
| P6 | the plan identity | no upstream copies at prep: stable while upstream runs end |
| P7 | the machine | set at the first prep, as today |
| P8 | what prep says | each group, its job, its files; for a file hand-over, *"at launch: takes <files> from <stage>'s newest run, which must have finished"*; for K5, the run it was written at |

**Launch.**

| # | situation | the design |
|---|---|---|
| L1 | `launch task`, nothing named | the prepared groups not launched whose upstream has finished — one: sent, several: asked (D3); none of those: the stages launched before, offered to launch again (today's rule, `_launch_unit`), beside what each waiting group waits for |
| L2 | `launch task --group N` | refused before anything is sent if a member's upstream has not finished — in the status door's words, with the command |
| L3 | a stage's FIRST launch, file hand-over | the one door, asked at launch: the stage before's newest run, which must have finished (D5), or the transport gather; `--from RUN` / `--cold` as prep takes them today, for a launch of one stage (L3a); copied into `run-0`, recorded (`.continued-from` / `.gathered-from` → `run.json`), ledger `continues` / `gathers`. K5: its files from the run its prep took |
| L3a | `--from` / `--cold` on a first launch | a launch of ONE stage; a group's members each take their default |
| L4 | **the prepared stage is still what the description gives now** | at every launch that takes inputs — a first launch, and a relaunch with `--from` / `--gather` (L6) — each member's prepared files (deck, run script, header, job row: its resources included) are compared with what prep's own plan would write now (`Plan.identity`, stamps masked; the allocation included, since a deck's values follow it — `BlockSize` from `mpi_np`; K5 planned at the relax run its prep took); a difference is refused, naming the prep again (a stage never launched) or § 5aa.5's way back (a launched one). Without it, an edit after prep-all — a template value, a run card's ranks, a re-prep of group 1 alone — would run the old job on new files, unsaid. A plain relaunch (warm or cold, nothing taken anew) runs the stage as prepared, as today |
| L5 | a relaunch | unchanged: warm from its own latest run, `--cold` none; a transport rung keeps its gathered inputs |
| L6 | a launched stage takes a newer upstream | `launch --from RUN` (an independent stage) and `launch --gather` (transport: the upstreams' newest finished runs, the three gates) on a relaunch — decided at the launch that uses it, recorded; warm or cold is the person's (2026-10-07) |
| L7 | the plan shown and asked | per member, what its run takes and from where, before the one question; `--dry-run` shows it, writes nothing |
| L8 | a refused send | an unlaunched `run-0` is filled anew at every launch until one is sent — what a refused send left is cleared first, so a later `--from` or a newer upstream is taken (`project-layout.md` § 1.6.3); K5's files are refilled from the run its prep took |
| L11 | the save before a first launch | a group's first launch saves the folder's state, at the send, before anything is written — once its plan stands, as prep's checkpoint 5 — and says which; the ledger records it (§ 5aa.5) |
| L9 | upstream failed / not converged | the door's answers, as prep's today: failed refused (name `--from`), not converged taken with a warning, the transport gates |
| L10 | the walk (group, bias sweep) | unchanged |

**Status and surfaces.**

| # | situation | the design |
|---|---|---|
| U1 | states | described → prepared (pending: *ready to launch* or *waits for X*) → launched → ended; a K5 stage not prepared: *waits for relax before it can be prepared* |
| U2 | `jobset status`, the Results ladder | the same door |
| U3 | Task setup — Stages card | a **Group** column: the number with *same as above* / *new group*; the building-on rule said beside the row (the folder route) |
| U4 | Task setup — Prep box | the groups, each: not prepared · prepared (never launched → **Prep again**) · launched; Preview and Prep for the picked |
| U5 | Task setup — tabs | **one tab per group**: its run card (shared values; a wall row per member, the job's wall their sum), and per member its *measure it* (benchmark, K13) and its lines; the launch line carries the Continue-from choice (L3a) |
| U6 | commands | the group's prep and launch; a member alone for a relaunch (D4) |

**Migration.**

| # | situation | the design |
|---|---|---|
| M1 | a `task.json` with no `group` | refused by the reader, naming `jobset migrate`: prepared stages take the groups `job-set.json` records; the rest repeat the number above only where the rule allows it (no stage it builds on in that group, `group.upstream_of`) **and** their run cards share the allocation, else add one. A prepared grouping the rule cannot state (non-adjacent stages) is refused by name, and the person decides |
| M2 | a `run-0` filled by a prep under today's rule, never launched | as L8: filled anew at the first launch |

### 5aa.4 V — settled by the trace: the force-constant stage keeps its geometry in its deck

Option (a) — write the force-constant deck from the structure and let SIESTA
read relax's `.XV` — is **withdrawn**: a force-constant run writes its last
displacement into `.XV` (`vibration_deck.py:119-127`, measured), so a retry
in the same folder (`continue_retries`) would start from a nudged geometry,
and the finish, whose reference is FC step 0 of its own output, could not
tell. The deck holding relax's coordinates is the design's reason, not an
accident. So a force-constant stage is the one value hand-over: preparable
once `relax` has finished, its source decided at its prep (`--from` a relax
run), as today.

### 5aa.5 R — redoing a launched stage: a save before a group's first launch

With every group prepared at once, *"redo via rollback to the state before its
prep"* restores the state before the one prep — and removes every run made
since, the earlier groups' included (`checkpointing.md` § 7, A5). Every printed
way back says it today (§ 5aa.2).

**Withdrawn, (i) — prepare a launched stage again in place.** Its runs keep
their own deck, but molbuilder reads an earlier run's facts from the stage's
one job row and today's description (the optimizer a `.CG` follows, a swept
rung's points): preparing again would make it misread its own runs.

**The answer: a group's first launch saves the folder's state first** (L11).
That state holds every run upstream of the group — they had to finish before
it could launch — and the group as prepared, never launched. **Redoing a
launched stage = restoring the state saved before its group's first launch,
then preparing it again** (P4: never launched there). Only the runs made since
— the group's own, and anything that built on them — are removed, and the
restore names them first (A5). Every printed way back
(`commands.rollback` and its callers, § 5aa.2) names that state; a stage never
launched needs no restore (P4). Today's rule, *a launched stage is not
prepared again*, stands.

### 5aa.6 The contract, then the code

**Contract (written first, shown for a yes):** `job-system.md` (*The task*; *What molbuilder does for you* rule 3;
the states; § 5.0 agreement 4–5, checkpoints 2, 2a, 4a, 6, the preview; § 5.4
the two ladders, *A stage launched again*; § 6.0 step 1, step 4) ·
`project-layout.md` § 1.6 lead, § 1.6.2, § 1.6.3 (the table: `.continued-from`
by launch; the prepared row), § 1.6.5, § 1.6.6, § 1.6.7 (`prep --cold` moves to
launch) · `stages.md` (the `group` field; § 6.6 the preflight's group check) ·
`architecture.md` § 3.2 · `job-contracts.md` (the copy at launch, :109) ·
`transport.md` § 1.1, § 2a.11, § 2a.14 (:2045), § 6.1 (:3759, :3813) and its restatements (:1510, :2197-2260, :4466)
· `vibration.md` § 5.2a (the force-constant stage named as the value
hand-over; its files at its first launch from the run its prep took) · `task-setup.md` § 5.3, § 6, § 11 · `web-api.md` (prep, commands,
folder) · `checkpointing.md` § 9 (a group's first launch saves, L11).

**Code:** `task.py` (the field and its shape) · `jobset/ready.py` (preparable;
launch readiness) · `jobset/prep.py` (prep groups; no file copy, no gather;
re-prep; the frame count from the plan; the group rule in the preflight;
`--group`) · `jobset/continuation.py` (the one door, asked by launch) ·
`jobset/submit.py` (first-launch take, L4's plan check, L8's refill, L11's save, `--from`
/ `--gather` on a relaunch, the upstream gate, G9; launch-time grouping retired) ·
`jobset/commands.py` and every printed way back (`rollback` callers; `prep --from/--cold` → launch) ·
`jobset/placement.py` (G8, the group's request admitted at prep) ·
`jobset/_cli.py` (`--group`; launch flags; L1's offer) · `jobset/runstatus.py`
· `jobset/migrate.py` (M1) · every description writer (G2) · Task setup page
and routes (U3–U6) · Results ladder.

### 5aa.7 Tests

Retired or rewritten (each asserts prep's copying or prep's waiting):
`test_the_road_on_real_runs_e2e.py:113`, `test_restart_group.py:135`,
`test_status_lists_the_ladder.py:42`, `test_hand_overs.py:33`,
`test_task_setup_prep_e2e.py:1265`, and the transport e2e cases reading
`.gathered-from` written at prep (`test_transport_on_a_real_junction_e2e.py`
:925). New rows through the road (`init → prep → launch --dry-run`): G1's
refused cases, P1, P4, G4 (membership, a removed stage), G8, G9, L2, L3, L4 (a template edit and a run-card edit after prep, each refused), L6, L8, L11 (the save, and a redo restoring it keeps the upstream runs).
Engine e2e on the user's word.

### 5aa.7a The contract — written 2026-10-11

Written into `stages.md` (§ 2, § 3, § 6, § 6.6), `job-system.md` (*The task*,
*What molbuilder does for you*, § 5.0, § 5.3, § 5.4, § 6.0), `project-layout.md`
(§ 1.6–1.6.7, § 2.3, the manifest), `architecture.md`, `job-contracts.md`,
`transport.md`, `vibration.md` (§ 5.2a), `task-setup.md` (§ 5, § 5.3, § 6.2b,
§ 11), `web-api.md`, `checkpointing.md` § 9, `results.md`, `running-a-job.md`,
`run-identity.md`, `script-preparation.md`, `tuning.md` and the worked
examples; one fresh review of the written text, its 15 findings fixed (the save
rule's restatements, the prep route's answer, the restore's reach — every run
since the save, `--from` / `--cold` speaking for one stage: `--stage` or a
one-stage `--group`, and on prep for every force-constant stage it prepares).
**Awaiting the user's yes before code.**

### 5aa.8 Before the work, and done when

2.6.0 (uncommitted) is committed first, on the user's word; this lands as its
own minor version (`task.json` gains a field; readers refuse one without it,
naming `jobset migrate`). Done when the contract is agreed, the code built
through the framework, two review rounds passed, and the user's
`AuBDTAu/transport/lowbias` prepared whole through the browser — this machine,
and a Sol preview — with group 1 launched by the user.

**Status 2026-10-12.** Built; the contract's one state machine and its two
flows are in `job-system.md`, *The task* (a stage, a group and a run; `prep
task` and `launch task`; three worked examples), each read off the code.
Review round 1 (two fresh agents, contract-first and code-first) and its
revision; round 2 the same, and its revision; the non-e2e suite 7204 passed;
every diagram of every doc touched parses with the vendored mermaid. Lowbias
prepared on this machine through Task setup, a Sol preview made (the folder
restored to its prepared state after), a dry run of group 1's launch clean.
Two simplifications, agreed 2026-10-12 and built: prep plans every member
of the groups picked into ONE plan and writes that plan after its one save
(the second planning pass gone, § 5.0 rule 3 kept); every run opened and
never launched is emptied whole by whatever fills it, a first launch giving
prep's own files back as they lie (`empty_taken` and the `whole` switch
gone). **Open:** group 1 launched by the user; the e2e files rewritten for
the groups and the gather at launch, run on the user's word; one member
planner for a single run and a swept rung — the next milestone, after 2.7.0.

---

### 5aa.9 The unification — one framework for prep and launch *(user, 2026-10-12: "a clean, logical, well unified structure and api system to support our flexibility, simplicity, explicity"; "do this now"; "make sure this is a framework level design and api level unification … diagram updated to include this and explain relation between layers/api/roles")*

Three fresh reviews (the contract as a set; the code's structure; the contract
cross-validated against the code) fed one revision. **Ruled 2026-10-12:** a
swept run whose points all ended reads *finished*, convergence in its own
column (never folded into the state); a job's member list is `members`
(`group` is only ever the number — `job-set.json` migrated); the ready door
does not re-plan a stage — *ready to launch* means its inputs are there, the
launch compares the files.

**The framework (code + `job-system.md` § 9, its layers diagram and API table):**
surfaces → one entry per verb → the work (which members) → ONE member planner
per verb (prep: `plan_stage`, quiet, no save/ledger; launch: `_plan_member`,
every layout) → the run filler (one folder, a run of points, flat) → the
submission builder (one job; a group's walk; a run's points' walk; a
benchmark's shelves) → save, write, send, record; the doors underneath answer
and never write (ready — `ready.task_state` read once; continuation; gather —
its own module; run; group — `header_path`, `may_share`; commands — one
`way_back`).

**Work list** (✓ when done):
- [x] F1 launch: one member planner, `_plan_member` (one run folder, a run
  of points, flat) → `_first_launch` (every kind through
  `ready.first_launch_takes`, the force-constant `--cold`/`--from` refusals
  moved in) / `_launch_again` (`_newest_launched`); `_points_walk` the
  points' submission builder (was `_plan_sweep`); `_plan_trial` for a
  benchmark trial; a group refuses a run of points by name; the job row's
  `members` excluded from launch's comparison (an F7 regression, caught by
  a new prep-protocol row)
- [x] F2 `jobset/gather.py` — the gather door (`gather_sources`,
  `transport_inputs`, `gather_into`, `composed_junction`, `rung_deck_now`);
  `jobset/transport_rung.py` — the transport rung builder prep and the
  gather both ask; `errors.user_error_as_prep` public
- [x] F3 prep: `read_description` (1), `checked` (3–4), `plan_stage` — the
  one member planner, no ledger/save, `quiet` for launch — and the verbs
  `prep_task` (groups: read and checked once, every member into one plan,
  the preflight notes recorded once) and `prep_bench`; launch's description
  check through `plan_stage(quiet=True)` (no deck warnings at launch);
  `prep_stage` retired, tests on the verbs (`conftest.prep_the_group_of`)
- [x] F4 `ready.task_state` → `TaskState` (task, template, plan, stages,
  groups; `prep_offer`, `launch_offer`, `group(n)`, `text()`), read once by
  prep's question, the launch offer, status, init's and describe's next
  line, the folder and commands routes; `plan_launch` with nothing named
  names the line a person types
- [x] F5 `commands.way_back` — one composer of a stage's way back (launch's
  `_redo`, ready's force-constant refusal, migrate's run-number line);
  `regrouped` names a launched group's way back instead of a prep that
  would be refused
- [x] F6 one job name, `group.names_for` (prep's header, launch's send;
  `header_of` names a job already written by its rows); the share rule
  applied by the page to the groups as edited, from each stage's
  `builds_on` in the folder answer — `check_groups` the one refusal
- [x] F7 `Job.members` (and the wire), migrated — `GroupAnswer.members`, the
  folder route's `members`, the status row's `group` (the number) and
  `members`; `jobset migrate` renames `group` (`.pre-2.7.0`); `lowbias`
  migrated on a copy, group 1 still ready — the real one needs `jobset migrate`
- [x] F8 one "prepared" door, `ready.job_of` (prep, the gather, the CLI);
  public names for what crosses modules (`continuation.source_stage`,
  `stage_before`, `cannot_be_named`, `unreadable`; `engines.pseudo_dir`,
  `screen_pseudos`, `siesta_provide_pseudos`); a walk's header is
  `runwrap.walk_header` (`submit.into_launch` retired)
- [x] D the reviews' defects: STAGE-PLAN.md written once per prep, every
  member's line, a column for each stage's job mates, the Order sentence
  rewritten; the commands route's "prepared" is the ready door's; a launched
  group's edited membership refused by name (a member launched alone named);
  names matched by `stage_key` in `_the_group`; a launched stage's row and
  its job's header kept when the description drops it (so one described
  again is launched); `Readiness.blocked` — waits / regrouped / unreadable /
  again — and `detail` says which; status's next step for an unreadable
  record; prep's next line the bare `launch task`; flags and mode checked
  before a question; a swept run *finished* when every point's run
  finished, convergence in its column, each point's `state`; `--from` on
  the alone line; the machine-limit refusal's retired rule; one
  force-constant takes line (`ready.written_at_takes`); stale wording and
  comments; dead fields (`Readiness.continuation/gather`, `readiness`'s
  `from_attempt/cold`, `Attempt.continued_from/cold`, the CLI's
  unreachable resolver, the folder's `prepared.task/placed`, `launchable`)
- [x] G the diagrams, read off the code as built and parsed with the app's
  own mermaid: `job-system.md` § 9.1 — the layers (surfaces, entries, the
  work, the member planners, the run fillers, the submission builders,
  save/write/send/record, the doors), each box its module and API; the
  roles table and the doors table; one `prep task` and one `launch task`
  as sequence diagrams, call by call. *The task* — the stage, group and run
  machines, every edge its act and module, each state its door; the two
  flows, their refusal nodes naming rows of the refusal table
- [x] C the contract: *The task* rewritten as the one contract (what a stage
  takes and when — the build-on rule's one home; the machines with the
  missing edges and states: ready ↔ waits, preparable → waits, the
  restores, removed, the unreadable record, queued → finished/failed, the
  flat note; one refusal table, R-P1–15 and R-L1–25; the examples
  corrected); § 5.0 in the code's order, its rules pointers to their homes;
  § 5.3's grammar (one `--group` at launch; `--from`/`--cold` scoped); § 5.4
  and § 6.0 mechanism, pointing to *The task*; the history notes and plan
  labels out of these sections. The other documents (project-layout,
  stages, transport, checkpointing, task-setup, web-api, job-contracts,
  architecture, results, changelog) — review A's D1–D8, I5, I9–I12, S1–S5,
  S11–S15, O1–O3 — each rule one clause and a link outside its home; the
  changelog's 2.7.0 row the file and command changes
- [~] T tests updated and added as rows; the full non-e2e suite; diagrams
  render; two fresh review rounds. **Round 1** (2026-10-12: the contract as
  a set; the code's structure; the contract traced against the code) —
  revised: a launch `sbatch` took with an id it did not print is recorded
  (id unknown, said); a stage put into a launched group answered by the ready
  door and prep (`group.launched_without`, `joined_launched`), and the page
  opens it a group of its own; warm, a stage's own restart files stand over
  the gather (a device's `.DM`), one carry for a run and its points; a
  relax-less force-constant stage's `--from`/`--cold` refused; the way back
  named by the members a group was launched with; a benchmark's placement
  refusal names its own way back; the Prep box's lines include its prep;
  prep's own command-line refusals written down; the description read and
  checked once per launch; dead names (`TaskState.group`, the status row's
  `members`, `prep._not_again`) out; the contract's state machines (prepare
  again, unreadable, removed, launched-members-changed), flows (each refusal
  at its step) and refusal table (R-P16–20, R-L26–36; conditions corrected)
  revised, the rules pointing to their homes across the documents. **Then**:
  the hand-built path/parse audit (user, 2026-10-12), an e2e in the browser
  on an H2 ladder reading every generated script whole — the continue and
  hand-over steps unified (user) — and round 2. **The e2e, 2026-10-12**
  (`projects/claude-e2e-27/optimization/H2`, headless — the Chrome extension
  not connected): Task setup — `tight` added with `restart: clean` into
  `medium`'s group (Save refused it first while it continued from `medium`),
  run cards, Preview, Prep; `launch task` (group 1), `--group 2` (the walk:
  `medium` from `01_coarse/run-0`, its `.XV` read; `tight` from its deck),
  `--stage coarse` again warm (run-1, the CG history read) and `--cold`
  (run-2, nothing read); status, Results and Task setup after; a Sol copy
  (`H2sol`) prepared for every header, R-L11 and R-L6 refused as the worked
  example says. Found in the generated text and fixed: a deck's header named
  `launch task --stage <it>`, refused for a member of a group of several
  (R-L6) — it names the group's first launch and the stage alone
  (`identity.deck_launch`); the SIESTA deck said *prep copies in* what its
  first launch copies; the preview's *at launch: takes* listed `.CG` the
  launch withholds (CG → Broyden) — it asks the pair's rule
  (`ready.carried_from` → `model.warm_carry`; a prep-protocol row,
  mutation-checked); the way back in a run script's banner and usage, the
  deck-agreement note and the vibration remedy said *the state saved before
  its prep* — one wording, `identity.way_back_words` / `LAUNCHED_WAY_BACK`
  (a benchmark's stays its prep's); the `.sbatch` header's *Submit with: cd
  <projdir>; sbatch X* — a line the run script's gate refuses, with no run
  number — names `jobset launch` and the hand form; the walk script's
  unused chain flags and its `run_trial` (`run_member`); a cold launch
  again's *to start it over, --cold*. **Then** round 2

### 5aa.10 One hand-over, the audit's items, the e2e's small findings *(user, 2026-10-12: "yes, do the hand-over design and all the audit items, address the minor issues that you did not change, make sure all are consistent, clarify how the state of each run and stage is marked/recorded? by a file name? is that file listed in the manifest?")*

No visible change to what a run's records say — the same text in the same
files; it is the unification approved. Contract first (`job-system.md` *The
states* and § 5.4, `project-layout.md` § 1.6, the catalogue's rows), then
the code; then the e2e's launches again on the H2 ladder, every generated
file read; then round 2.

**Work list** (✓ when done):
- [x] S how a state is marked: *The states* gains the one table — each
  state, the file(s) that mark it, its manifest row and its door; every such
  file checked to be in the manifest (`project-layout.md` § 5)
- [x] H one hand-over: `runrecord.RunRef` — a run as a record names it (its
  folder from the calculation's; where runs share the folder, its own name),
  one text, written and read one way (`.continued-from`, `.gathered-from`,
  `run.json`'s `continued_from`); `materialize.HandOver` — the run that
  takes, the files each from its run, the record (`continued` / `gathered`),
  copied with the plan or by the walk; `materialize.hand_over` — the one
  executor (the attempt's carry, the gather, the gather carried on a launch
  again, a point taken over, a point's start, a run of points continuing, a
  flat run's record at the send); the walk script's start copies read off
  the point's `HandOver`; `_Member.hands`
- [x] A2 the stage's newest run, one rule: `runrecord.newest_run` (runs that
  share a folder: the newest launched, by the launch records; a folder of the
  run's own: its newest try) — `run_of`, `run_state_of`, `run_files`,
  `continuation.read_run` (its patch out), the flat defaults
- [x] A5 a run's files through the run door: the displacement sweep's
  `.spectra.json` / `.FC`, the transport record's `.ORB_INDX` and decks
  (`Run.on_label`, `Run.deck`; no deck is an error, never a default), the
  force-constant finish's `.FC` composed
- [x] A9 `paths` reads a benchmark container's name; one `JobSet` reader, a
  planned plan's too (`JobSet.load(path, plan=)`); one `slot-provenance.json` reader; the
  unnumbered name composed once (`RunNames.unnumbered`); a walk's names from
  one composer; one deck reader for *does it warm-start*
- [x] C broad excepts narrowed (each says what it catches, or raises);
  dead code out (`validation/identity.py`, `runfiles.is_carried`,
  `materialize.trials_in`, `Run.carried` renamed and used); private helpers
  crossing modules made public (`parse.fdf`, `script_emit`'s record
  readers, `rundir`, the engine seam's hooks, `runwrap`'s header renderer,
  the bench check); `JobResult.status` one closed set, the sentence its own
  field; `trajectory.js`'s carried roles; the deck roles from the catalogue;
  the run script's placeholders filled
- [x] M the e2e's small findings: `STAGE-PLAN.md` lists the machine record
  the same prep writes; the run script's *Mode* line names the restart files
  that lie there; the effective-parameters block says what its columns are;
  Save's group refusal names `restart: clean` beside a new group; the
  *waits for … Launch it* line
- [x] R `run-identity.md` § 5 rewritten (user, 2026-10-12: *"yes, rewrite
  § 5"*): each case refused, closed by the layout, or said by `launch` and the
  banner; a changed cell said by nothing — `validation/identity.py`, never
  called, retired; ID-CHK closed
- [x] F the flat layout's record *(user, 2026-10-12: "your recommendation is
  ok" — option (b))*: a flat run's record names the run whose restart files
  it reads, the last run in the folder of any stage (`runrecord.last_run`),
  and `launch` says so beside the stage's own latest run
  (`continuation.flat_files`); a flat force-constant stage's launch reads
  the `relax` run its record names (`runfiles.parse_run_name`) — e2e on a
  flat chain, mutation-checked
- [x] T targeted tests, mutation-checked; the e2e's launches again; the full
  non-e2e suite once; round 2. **2026-10-12**: the full non-e2e suite, one
  row judged (the Mode line's words are `job-contracts.md` § 4.4's — kept,
  only `<files>` now the ones lying there); the e2e — road, flat, vibration,
  transport junction — each run's `.continued-from` and `run.json` read
  through their doors (road; a flat chain added for the record written at
  the send); found and fixed: a test still calling `Run.carried`, and the
  junction test's template comparison holding `psml_lib`, which § 3.1 makes
  differ (a cited run's pseudopotentials travel with it). **Round 2** (two
  fresh reviews: the contract against the code; the unification's code) —
  fixed: `fdf_true` lacked libfdf's `y` (read from SIESTA 5.4.2's
  `fdf.F90`; `1` is not one) and two more word lists folded in; an
  unreadable `.gathered-from` / `.continued-from` said on its status row,
  never the whole status, and a record that is not text read as not
  reading; the launch output's row label from its code; `gather` and
  `runs.speaking` ask the one answer (`latest_attempt`, `newest_run`); the
  run record reads one run for every part; a flat force-constant stage's
  launch reads the `relax` run its record names (`runfiles.run_of_name`);
  the status detail's reason from `commands.reason_of`, beside the block
  format; `BENCH_DIRNAME`; the provenance compared resolved on both sides;
  the states table's door column, who reads each record, a flat warm
  retry's record, where a point's inputs come from and what a point taken
  over keeps (transport § 2a.11), the flat layout's shared restart files
  said rather than claimed as one run's, and stale names in docs and
  comments. Left as they are, said: a cited transport record fails whole on
  an unreadable provenance record (it is a claim); `RunRef.within` beside
  `run_folder`; the sidecar's `fc_file` field, written, no longer read

### 5aa.11 Review rounds A and B — the contract, the code, the UI, the CSS *(user, 2026-10-12: "do another two rounds agent review ... check for inconsistency, duplicated code/design, redundance, detect over engineering, logical problems and UI design problems. check css system ... detect handcrafted magic number, detect bad smell of bad design")*

**Round A** (four fresh reviewers, each finding verified against the code
before it is acted on) — triage; ✓ fixed:

*Defects, fixed now:*
- [x] UI: a folder answer that fails, or a description its reader refuses,
  leaves the last folder's description for Save to write here — one
  refusal state (`refuseFolder`), Save held; the mode is the folder
  answer's own
- [x] UI: the queue card shown in a hand-over drops its values — shown with a
  description only; hand edits read back into the queue and notify cards;
  a card's write over text that does not parse refused out loud
- [x] UI: the bench grid's fit stale after a machine or folder change; a
  refused Save drops its findings; the group card's wall sentence (worked out
  a second time, and wrong); the remove-column question outlives its answer
- [x] Python: the transport record's unreadable `.gathered-from`; each swept
  point's seeded progress log at its own frame; `--mode ask` on a run of
  points in the point's folder; a named trial's way back; the queue-named
  refusal once, in `_sbatch_request` (a group had none); the per-node core
  cap named; the converged verdict structured (`RunStatus.verdict`), never
  parsed back from its words
- [x] Python: the notify file read's `UnicodeDecodeError`; four broad
  excepts (`contract.relaxation_of`, `registry` sniffer, `_cli`'s task
  read, `_folder_bench_refusal`); group numbering twice (`default_groups`,
  `migrate`); job matching by two rules (`stage_key` / exact, a bare
  `next()`); dead code (bench `parse_sacct_mem`, `Refusal.as_issue`,
  `Partition.has_gpu`, `Directives.queue`, `activity.CLASSES`,
  `STEP_INDEX`, `prep_inputs`' unreachable fallback, `_launch_unit`'s `js`);
  comments contradicting code; `runwrap`'s dead label checks; also
  `JobResult.to_dict`, `script_emit._extract_header_text` (its tests only),
  `gpu_share_notes`' node-fit branch, `record`'s private exports
- [x] Contract vs code: a queued warm launch of a run of points read
  *failed*; the gather's "launch it again" for a run still running; an
  unreadable record of another run escaping `status` / launch whole
  (`gather`, `flat_files`); launch's usage refusals not written down; a
  prep's save refusal unrecorded; `--cold` "at its first launch" in three
  texts and R-L12; `prep_stage` named in architecture.md; the floor table;
  the class diagram; the flat force-constant `--from` line naming a refused
  prep; the states table's *opened* door
- [x] UI: the hand-over's Send into a folder nobody checked (`task-handover.js`
  `safeSave` → `writeFile` into the checked `dest`, under the page fence);
  Transport's electrode rename kept across junctions (a pick starts without
  it; Describe waits for the panels drawn for the citation); the road text
  after Describe; Results' "open the report first" (held across the ladder's
  redraw), the deck's View path (`record_dir`), `_pickRun`'s failures and late
  answers; `removeChannel` not checking `ok`; the Spectrum Send wording; the
  640 px drawer breakpoint (the stylesheet's answer, one number); dead JS
  (the retired lock's subscriber, `CM_ASSETS`, the dialog test seams)
- [x] CSS: page sheets re-owning generic controls (`spectra/style.css`),
  Modify's second severity classes (its edit line a `.status` through
  `lib/status.js`), the banner restyled per tab (padding only), Results'
  sheet styling the inspector module's classes (`lib/inspectors/
  inspector-card.css`; the boundary test's known list empty), `!important`
  where specificity already wins, inline styles in templates, dead rules
  (`a.btn-like`, `.structure-viewer`, `.molviewer-label-predefined`, five
  unused tokens), every fallback mirroring its token, raw colours in the
  sidebar, the trajectory inspector's inline colour, the off-grid spacings
  with a token

*Design calls, for the user:*
- UI: Structure optimization's *discard unsaved changes?* on Load — the
  viewer is read-only and a load leaves the parameters as typed, while
  `runtime.md` puts the question on a dirty canvas; dropping it changes what
  the page offers (and `test_build_e2e`'s gate test)
- UI: the shape offered after it is fixed (disable it with its way back, as
  the machine card); the fence claimed by every card edit (§ 7a says so);
  prep's answer lines served rather than composed in the page; keyboard
  access (the tabs pattern once, the sidebar rows, glyph buttons named);
  the detection chip's atom-count advice, which no contract states; one
  element builder, one path helper, one download helper — *the shape, the
  fence and the chip done 2026-10-13 (Q19, small UI P1–P2); the rest are
  Q19's later phases*
- CSS: one modal component (`.mb-dialog`) — *done 2026-10-13, Q19's CSS P1* —
  and one button vocabulary; a flat
  `.card` variant for Results' six restatements; `.num-table` reused; one
  tab-strip component; tokens for durations, z-index layers, weights and
  line heights; the sealed modules' own palettes (`vibrationview`,
  `spectrumchart`, `docs-render`) — named in § 2, or brought onto the
  tokens; the stylesheet load order
- Python: the files a calculation holds read once (`JobSet.of`, the
  template text, `Shape.named(task.shape)`); `done` returning its
  `RunStatus` and `run_state_of` everywhere; one instruments reader; the
  long functions (`render_run_wrapper`, the engines' `spec_for`); the
  floors' upward imports and lazy cycles; `RunRef` for `written_at` and the
  continuation's fields — *Q19's Python phases 2–5*; `runwrap`'s f-strings
  without placeholders, gone (pyflakes finds none, 2026-10-13)

**Round B** (four fresh reviewers on the result, 2026-10-12) — triage; each
finding re-read in the code before it was acted on:

*Found by the suite, fixed first:* a generator in `_plan_calculation` read
`stage_key` imported only when a job set already existed — a first prep
raised `NameError` (282 failures); the stage's job is now `JobSet.job`'s.

*Defects and conformance, fixed:*
- [x] Launch: a transport rung launched warm, then `--cold`, lost the seed's
  density — a run's gather is carried one hop at a time through
  `.continued-from` (`submit._carried_gather`); a record that does not read
  refused once, in `plan_launch`, and written down (R-L21); `--from` naming a
  run never launched — the run the launch fills among them — refused (R-L37);
  a flat force-constant stage read at the `relax` run its prep took, said as
  *the run its deck is written at*; a cold relaunch's carried inputs in the
  ledger (`gathers`)
- [x] Prep and status: a swept rung's seeded progress log moved into the
  run's point (and the real-junction e2e asserts it); the way back says the
  change after the restore (`way_back_words(change=)`); the bench grid's GPU
  share judged at ranks per GPU, never a sentence's word; a point's row
  decided on its state and verdict (`opened`, `converged`), an unread start
  in its own key (`start_unread`), one *taken over* rule (`_taken_over`);
  `migrate`'s launched flat stage; ask mode's missing header through
  `_no_sbatch`; the cited run asked at its one run; a flat `written_at` that
  does not read refused; the cap notes keyed by domain
- [x] Python hygiene: the step errors one tuple (`errors.step_errors`); jobs
  looked up by `JobSet.job`; the dead `gpu_side` / `prefer_gpu` chain,
  `render_stage_refs`, `_register_engine_validator`, two one-caller wrappers,
  the `hasattr` guard, the validation package's unused private imports;
  broad excepts narrowed (`_cli`, `configured_mode`, `read_description`,
  `_folder_ladder`); stale comments and docstrings (ledger, runwrap's dual
  test, `numbered`, `materialize`, `migrate_cmd`, the header's *at prep*);
  pyflakes clean on every changed file but the autouse fixture imports and
  runwrap's f-strings
- [x] UI: the page fence holds the keyboard too and shows Cancel only with
  cancelers (`page-busy.js`, ui-contract § 10, its test mutation-checked);
  Describe opens only on panels that drew, the last citation's schemas
  cleared, a late citation answer dropped (`_citeSeq`); a transport
  Describe has no Cancel (its door writes in one act), a cancelled write
  says it may have landed; the road after Describe one group per launch;
  *one calculation per folder* (Rule 1 renamed); `safeSave` /
  `saveToWorkspace` deleted with their tests and every stale mention; the
  status lines through `lib/status.js` (Results' ladder, Structure
  optimization's hand-over, Modify's bootstrap, Spectrum's copy note, the
  markdown editor's bar); the structure pair named by the folder answer
  (`structure_pair`); This machine's Test held while it runs; the dead
  `molbuilderTab` hook
- [x] CSS: the chemistry card's id rules no longer re-pick `.status` and
  `.hint`; every line reservation `1lh`; MolView's per-control padding and
  family that its panel rule overrode; the run-history graph's lanes on
  tokens; the page copy of the disabled-input rule; `.rec-diff-name` in its
  owner's sheet; form-schema's `!important`; § 3's breakpoint list recounted

*Design calls, for the user (beside round A's):*
- Execution: `runrecord.last_run` orders a flat group's runs by
  `launched_at`, and a warm retry inside the group writes its record later
  than the member after it — the rule (§ 5.4) and the code share the flaw;
  the order wanted is the walk's
- UI: the closed drawer kept out of the Tab order (`inert`) and the Run
  panel's toggle keeping focus — with round A's keyboard-access call; one
  `pageBusy.run(reason, cancelers, fn)` for the three *claim unless held*
  copies; one debounce helper
- CSS: the local severity callouts (`.ps-preview-status`,
  `.inspector-inline-error`, `.selection-bootstrap-error`, `.transport-warn`,
  `.bench-mismatch`) drawn as finding rows or `.status`; one disclosure
  (`<summary>`) look — eight today, Modify's a copy of form-schema's; one
  failed-mount look (`.inspector-error` vs the partial factory's unstyled
  `error-card`); the busy cover twice (`.results-loading-*`); one inline
  `code` chip; one markdown typography; a chrome tier for the `--ps-*` tokens
  other modules borrow; global tokens for the accent rule (3/4 px), letter
  spacing, the page caps; Task setup's two viewport queries as container
  queries; the banner's padding per tab; `--warn` (vivid) beside
  `.status.warn` (soft)

*The user's rulings on the calls (2026-10-12):*
- [x] e2e: the whole batch run (278 of 301 at first), and the device
  launched warm, then `--cold`, added to the real-junction e2e (mutation-
  checked: the gather carried one hop only drops the seed's `.DM`).  The 24
  failures, each judged: two regressions of this round, fixed at their
  owner -- the tab strip's listeners outliving the spectrum inspector's
  dispose (the strip registers through its owner's scope, `listeners`,
  ui-contract § 4.1) and the page fence's focus scrolling a refusal out of
  view (`preventScroll`, § 10); stale tests brought to the contract -- the
  Task setup fixtures' stages without a group (stages.md § 2), Modify's
  `.is-active` panel (shown by `hidden`, § 4.1), the monitor bundle broken
  where prep wrote it (a first launch opens its run from the stage's files,
  job-system.md § 6.0 step 4), the PySCF probes started with their run's
  number (`runtime_info.run_given`, W57); one teardown was this session's
  own edit under the checkout guard
- [x] A flat calculation's LAST RUN is a record, not a comparison of
  timestamps *(user: "there should be a record manifest of the last run so
  that is the key information what is the last run")*: one file in the
  folder naming it (`<label>.last-run`), written by the run script the
  moment its engine starts, in the manifest, read by every reader of "the
  last run here"; § 5.4 and `runrecord.last_run` rewritten
- [x] Structure optimization's *discard unsaved changes?* on Load removed,
  its browser test retired
- [x] Keyboard access, one pass: the tabs pattern once (`tab-strip.js`) and
  used everywhere, the closed drawer out of the Tab order, the Run panel's
  toggle keeping focus, every glyph button named
- [x] runwrap's f-strings without placeholders (mechanical)
- [x] The run card is the job's container (user, 2026-10-12, on
  `AuBDTAu/transport/lowbias`: "different groups have different default
  group of parameters"): one row per setting in one order on every card,
  written into the members that read it, a value on a non-reading member
  shown with its ×, the add list what some member reads (`task-setup.md`
  § 6.2b; `runCard`)
- [x] `diag_algorithm` defaults to `ELPA-1STAGE` (user: "we know that works
  better for most cases"; `engines/siesta.md` § 7)
- [x] A transport ladder always keeps its seed (user, 2026-10-12: "Always
  keep the seed"): ruling Q4's skip withdrawn — `task.fixed_by_kind` the five
  rungs, the reader refusing any other list, no × or add-back on the page,
  the ladder's inputs without their seed switch
- [x] The run card, second pass (user, 2026-10-12, on `lowbias`: "the last
  group does not have all parameters listed and it's not possible now to add"
  the GPU switch; "framework level solution, not patching"): the same rows on
  every card (`runRowNames`), a row no member reads shown and not editable;
  `gpu_count` read where `use_gpu` is (the catalogue's `stages` — `tbtrans`
  has no GPU path); a run-card value on a rung that does not read it refused
  by Save and prep as an override there is (`validation.task`,
  `template.reads`); a blank row shows what the run uses, the empty stage
  cell's helper (`defaultText`) — the calculation's block, else the
  template's value (`task-setup.md` § 6.2b). `lowbias`'s template, written
  before the new default, records ScaLAPACK as *not chosen*: the template is
  the calculation's own, and a molbuilder default does not rewrite it
- [x] A prepared stage refused at launch (R-L15) because molbuilder itself
  changed between the prep and the launch, no setting changed: the rule
  stands, by its purpose -- prepared files are the product of the
  description, the machine's record and molbuilder's code, and files the
  current code would not write make a run its readers cannot read back (the
  old run script wrote no `.last-run`); a prep again costs no compute
- [x] The Transport tab's chemistry card is asked on the real junction's
  road (user: "rebuild it on the real relaxation"): the relaxation run cited
  through the tab's restored session -- the charge 0 by rule, the cited
  unrestricted spin floating (ES6, ES7), restricted stated on the panel --
  and `test_chemistry_card_e2e`'s case on a hand-written junction, which
  § 3.1 refuses since Q17-c, gone
- [x] *Every rung that runs an SCF* is one named set (user: "build the named
  set"): `paths.ROLE_SETS["scf"]` beside the roles, cited `stages = "scf"`
  by the fifteen rows that wrote it out, read as its roles
  (`template._stages_from`, a name no set has refused) -- every item reads
  as before; a calculation's own template writes the roles out
  (`template.md` § 5, § 6.4)
- [x] The run card's × only where it removes what the field cannot: a row
  added on the card and not filled, and a value on a member that does not
  read it (user, 2026-10-12: "i clicked "x" and it stayed"); a standing
  row is cleared by its blank (`task-setup.md` § 6.2b; `standingRows`)
- [x] **The browser walk, 2026-10-13** (user: "use the AuBDTAu
  optimization run as starting citation, create a project by yourself,
  vigorously check all UI components"): headless Chromium on a scratch
  copy of the tree -- New project, New folder, the junction picker by
  mouse and keyboard, the shared panel and two rungs' values set, the
  bias builder, Describe; Task setup's every card, the machine, three run
  cards, Save, Preview, Prep; `launch --dry-run` of group 1 and the
  refusal of group 2; Results on the cited run.  Every value landed where
  set (template sources, the rungs' overrides, the bias, the run cards).
  Fixed: Describe left disabled when a newer rung draw superseded the
  citation's (the newest draw settles it, a citation voids those in
  flight); the picker's root opened by hand (its state said closed) and
  its meta line moving the rows under a double-click (anchored, its line
  reserved); the lock note's doubled arrow; checkbox fields spread down a
  tall row (`align-content: start`); Describe a bare `full-btn` in an
  action row (`primary`, as the other tabs' send); the run card's value
  cut at 16ch (sized to what it says, `--ts-runval-min`); the prep answer
  box past the card (`min-width: 0`); "Pick a group" after every group was
  prepared; the run-count marker's two builders (one, `ranMark`); the
  sidebar's rows a click alone (a listbox, ui-contract § 4.1) and its
  create bar leaving Download alone (two by two); 52 raw font weights
  (their tokens); the energy charts' ticks all alike at a fixed count of
  figures (one shared axis rule, `plot-theme.energyAxis`)
- [x] A projects root not named `projects` refused (2026-10-13): the
  contract already says the tree is the folder named `projects` wherever
  it is (`job-contracts.md` § 2.5), so `paths.projects` and
  `$MOLBUILDER_PROJECTS` move it and never rename it --
  `projects.named_projects`, asked by the resolver and by `envs
  init-config` before it writes; the tests that pointed the root at
  `tmp_path`, `tree` or `fixtures` build in the conftest's tree
- [x] **The transport k-point grids, investigated and consolidated
  (2026-10-13, 2.8.0)** (user: "a thorough investigation ... with
  literature/manual/document review"; "yes to A, B and C"): read against the
  SIESTA 5.4.2 source and manual on this machine and the literature, every
  reference Crossref-checked. The grids molbuilder wrote were right; the
  reasons were not, in about ten places -- a lead's points along z do not
  smooth the self-energy (the endless lead is summed exactly, Lopez Sancho),
  they decide the Hamiltonian and Fermi level it is built from and aligned
  to (`m_ts_electrode.F90:1462`); the leads and the device share one in-plane
  mesh because the lead is the bulk of the device's electrode region, which
  TranSIESTA checks (`ts_electrode.F90:1235`), not because Σ "pairs with H at
  the same k". Built: A -- `electrode_kz` warned below 50, the manual's
  floor; B -- a blank `tbt_k_grid` is three times the shared mesh, worked
  out by `kmesh.mesh_for` so it follows `kgrid`, named in the deck's
  provenance (`resolve.BLANK_ANSWERED_BY`), never recorded as the person's or
  the citation's (closes § 5u.1 row 13 ⑫); C -- a transmission mesh the
  person set coarser than the shared mesh is warned (`kmesh.check`).
  `transport.md` § 0.2-0.3b rewritten with the per-stage table, the
  dependency diagram, the setup order and the citations; the Transport
  tab's k-points card as one section per stage, each stage's note with its
  k-points, the stage labels one spelling (`_rungLabel`, "1 · seed" -- the
  rung's panel said "1. seed", read as a card number), the rail's stage list
  numbered once. Four references added (`Brandbyge2002`, `LopezSancho1985`,
  `Thygesen2005b`, `SiestaManual542`). Row 13 ⑭ (the folders renumbering
  without the seed) is moot: the seed is always kept
- [x] **The k-point wording, one vocabulary (2026-10-13, 2.9.0)** (user:
  "the language are vague to state k-point,k-grid or mesh etc
  inconsistently"): *k-point grid* everywhere (never *mesh*, SIESTA's
  real-space grid), *in the electrode plane (x, y)* and *along z* (never
  *transverse*), *stage* on the Transport tab; each reason in full sentences
  with its background -- the contract § 0.2-0.3b and every k-point passage
  of it, `siesta.md` § 6.1, `science/overview.md`, the four items' labels and
  help, the stage notes, the k-points card, `kmesh`'s findings, the decks'
  k-point comments. Done by an agent, every edit re-read; the § 2a.7 and
  § 3.8 passages that still gave the old reasons were corrected with it.
  Left as they are: "transverse" where it names the cell's geometry (§ 4,
  § 5's I12), and the engine's own message, *"found incompatible k-grids"*
- Open, for the CSS pass: two modal vocabularies (`.mb-dialog`,
  `.molbuilder-projects-dialog`) and the sidebar's native `prompt()`
  (rename, a state's name); the projects dialog's raw `13px/1.45`, sizes
  and hex fallbacks; `.lock-reason`'s raw `2px` / `1.35`; the junction
  card's control under nine paragraphs of prose. *The two vocabularies, the
  native prompts and the projects dialog's raw values done 2026-10-13 (Q19,
  CSS P1: one dialog component, the sidebar's own deleted with its values, no
  native dialog left — rename was already in-page, a state's name was a
  `prompt()`); `.lock-reason`'s values go with CSS P2's tokens, the junction
  card's control with decision 3 (six paragraphs, not nine).*
- 2.7.0 is committed with these once the tests pass; the CSS group and the
  Python group of design calls follow as their own passes, each proposed
  before it is built

*Not done here, and why:* a test for the warm-then-cold device needs a real
engine run (an e2e, on the user's word); the cold sweep's docstring against
its case arms and three small dead guards (`next_step`'s `s is None`, the
`bench=` that is always false, `_with_notify`'s `None` tests) were not
re-read whole.

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

### 5w.1 Open, in Q9's order

| | what is left | where it is | done when |
|---|---|---|---|
| **K13** | **one atomic writer** — private writers remain at `checkpoint.py:664`, `sidecars/molstruct.py:549-581`, `sidecars/spectra.py:159-180` (V1.37), `workingcopy_structure.py:435`; 25 raw `write_text` calls (e.g. `compose.py:829`, `jobset/summarize.py:369`, `jobset/_cli.py:445`, `script_emit.py:1460`, `:1934`) | `persist.py:68,163` | every file molbuilder writes goes through `persist`'s atomic writer |
| **K14** | **the engine's parallel model as data** — today a suffix branch, `runwrap.py:4242` `ntasks = 1 if suffix == ".py"`, also `:3443` | `job-contracts.md` § 6.2 | the model (MPI or OpenMP) declared per engine and read by the scheduler header |
| **K16** | **the wrapper fence narrowed** — the SIESTA fence lists every SIESTA item (`runwrap.py:1746`), PySCF narrows by calculation only (`pyscf/layout.py:313-314`), against `parse.md` § 5d.3a's *engine, calculation and stage* (marked not built) | `model/parse.md` § 5d.3a | the fence writes the rung's own items |
| **K18** | **a PySCF run's level of theory** — `parse/contract.py:28-46` reads SIESTA decks only (`pyscf.md:135`) | `parse.contract.contract_of` | a PySCF deck's basis, method, functional, charge and spin read by the one reader; the Results export carries `info.calculation` for a PySCF run |
| **K17** | **physics needing a build or a refusal**: SO-C10/C11 — `ParallelOverK` automatic over k (`siesta/input.py:443-446`), ignoring ELPA, its help silent on the deviation, and counted before time reversal (`kmesh.py:111-114`); SO-C12 — `find_psml`'s case and prefix fallback (`siesta/input.py:222-235`) against `pseudopotentials.md` § 9.1's exact name; **PO-C4 — the geomeTRIC log**: § 5w.3 of 2026-09-29 dropped its promise, but `runfiles.py:994-1000`, `pyscf/input.py:49`, `:1518-1522`, `pyscf.md:137`, `project-layout.md:2683` and `vibration.md:2093` still promise `_geom.log`, and PySCF 2.14 substitutes its stream-only `log.ini` when `geometric/log.ini` is absent (`geometric_solver.py:147-150`) — confirmed on an H₂ PySCF run, then the promise dropped everywhere; PO-C10 — the ECP read back; PO-C12 — a `-V` functional with D3; PS-C13 — mode numbers bounded at prep (`validation/spectra.py:107-134`); T-F4/F34 — Task setup's add and rename on a transport ladder (`task-setup/viewer.js:1253-1266`); the vibration's level-of-theory check skipping the recorded k-mesh (`parse/contract.py:63-68`); PS-C3 and PS-C2's builds (their refusals are in: `validation/spectra.py:386-415`, `validation/chemistry.py:378-391`) | as listed | each built, refused, or its promise removed |
| **K22** | **per-atom starting spins** — **the user's design word first**: where the starting moments live (the structure's per-atom data, or a template item) and how SIESTA is told them (`DM.InitSpin`). Measured 2026-10-08: an unrestricted hydrogen junction cannot run — every rung's moment floats, the aligned start leaves no minority density, the empty channel's GGA potential puts its states near 3000 eV; the suite's transport run is restricted until this is built (`tests/test_transport_on_a_real_junction_e2e.py:26-32`) | no `DM.InitSpin` writer | designed, ruled, built, and a polarized junction run end to end |
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
remainder is § 2's F9, F11, F14, F19b, R2-15.*

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

- [`archive/2026-10-13-plan-rows.md`](?doc=archive/2026-10-13-plan-rows.md)
  — the rows of § 2 and the queue done, settled or closed by 2026-10-13
  (after 2.9.3), each verbatim under its verdict.
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
