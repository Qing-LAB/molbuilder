# The plan — one file

**Role:** plan — **the only one.** Every open item lives here or in the
ledger a release names; the nine plan documents that preceded it are archived
under `docs/archive/2026-09-01-*.md` as records of what was decided and built.
**Domain:** all
**Started:** 2026-09-01, by consolidation

> *(user, 2026-09-01: "We don't need ten plan files scattered. We want one
> plan folder or one plan file and stay with that file.")*

**Consolidations** — each moved what was done or settled, verbatim under its
verdict, to its archive (the list at the end): 2026-09-10, 2026-10-09,
2026-10-13, 2026-10-14, 2026-10-18, 2026-10-20, 2026-10-27, 2026-10-28, and
**2026-10-31** *(the user, 2026-10-30: "consolidate with previous review
results to unify a single plan to finish"; "go ahead, science after the
finish line")*: a fresh review of the documents, the pages and the backend at
2.56.1, every open item of this file and of the 2.54.1 and 2.55.1 reviews
settled against the code, every finding checked by a second reader — the
ledger is [`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md).
The queue, THE ORDER, § 2's rows, § 1am's and § 1ak's lists and every
section's *left* bullets were replaced by **The one plan** below; they are in
[`archive/2026-10-31-plan-consolidation.md`](?doc=archive/2026-10-31-plan-consolidation.md)
as they stood.

---

## ▶ HANDOVER — where we are, and where to start *(2026-11-02)*
**Where we are: 2.64.0** — **F3's engine side**: an engine's facts are its
record's — its settings check, its reading of the same calculation, the
longest label it keeps whole, its deck's keyword reader — and a run's files
are read by what they are, not by engine name (the changelog's 2.64.0 row);
2.64.1 says a SIESTA GPU run on ScaLAPACK rather than refusing it (G11).
**Next: 2.65.0**, F3's kind side and pages: a kind's and a pair's facts
asked of their records — the atom order a pair's decks list atoms in, a
transport run's scratch status line, the description's checks, the kind a
citation is, the regions a junction's labels name — the settings catalogue's
reasons for what a kind fixes or shares, and the pages' engine and kind words
and label palette served by the server (the ledger's F3 rows left open, and
*Found by 2.64.0's review rounds* F3-R-2 and F3-R-6). The
releases are the changelog's rows; the history this handover carried,
release by release, and the checks each release ran are in
[`archive/2026-10-31-plan-consolidation.md`](?doc=archive/2026-10-31-plan-consolidation.md),
*The handover's history*, and the finish-line review's ledger. The three
before it: the whole-jobset review's **Release B** — what a stage was prepared
with, a sequence's header, a run of points' continuation, the ways out
composed once (2.62.0) — **2.62.1**, the code's text brought to the
rewritten jobset chapters, and **2.63.0** — one run script for every
engine, each block decided by the engine's record
([`execution/architecture.md`](?doc=execution/architecture.md) § 3.3:
`environment`, `gpu_environment`, `programs`) or the job, or asked of the
parse layer by the output's role; a PySCF run's failure says how it ended;
who applies a limit, the run script or the deck, is the catalogue's
(`limit_applied_by`, renamed in 2.64.0).

**Last checked**: the fast batch on 2.64.0's code (6,460 passed); two
review rounds over the whole touched modules, each finding checked in the
code before it was fixed; each new or changed test row seen failing on
2.63.0's code. **No e2e**; **untested** is listed in 2.64.0's row. This machine's record was re-probed by the user
(2026-11-02) after a test-style record had replaced it on 2026-10-30.
**The test suite's guards** landed just before 2.63.0, after a test helper
run by hand outside pytest had written the suite's fake machine record into
the person's real config (2026-10-30): test code loads only under pytest,
every test process runs in a sandbox of its own, nothing leaves the machine,
and one write guard covers the whole run
([`process/testing.md`](?doc=process/testing.md) § 2b). The audit that
found why — every test file read whole, with a verdict per file — is
[`archive/audits/2026-11-02-test-audit`](?doc=archive/audits/2026-11-02-test-audit/README.md);
its cleanup is F9's. Nothing waits on the user.

**What runs next** is **2.65.0** — F3's kind side and pages — then the
releases in the order of *The one plan* — the user's to change. **The
user's decisions** are under *Decided*; **what only Sol can show** and
**new science** are under *After the finish line*.

**Off the plan** (the user, 2026-10-18): every feature the three roads do not
need; any check, warning or format the contracts do not state. The AuBDTAu
workflow is the user's, on Sol *(the user, 2026-10-10: "no, you should not
run it on this workstation")*.

---

## THE FINISH LINE *(the user, 2026-10-30)*

molbuilder is finished when:

1. **§ 1am's steps 2–8 are done** — the run script names no engine; every
   engine and kind fact is asked of its record; every name and record goes
   through its owner; one reader and one writer per format; one home for each
   vocabulary, default and rule; the floors and the objects whole, the
   results views built (§ 1ak S3); the tests made through the road (§ 1ak S4).
2. **The three roads are proven** — an optimization, a vibration (SIESTA and
   PySCF) and a transport calculation, start to finish in the browser on
   real runs.
3. **The documents are true and findable** (§ 7).

New science and what only the user's Sol runs can show come **after** it
*("science after the finish line")*.

---

## THE ONE PLAN — the releases to the finish line

**Every item is a symptom; a release is one redesign.** *(The user,
2026-10-31: "make sure after these checked validated results come, your fixing
should be framework level, holistically and unify api and data structure
design, such that you are not patching and hacking to generate handcrafted
redundant/duplicated codes".)* Each release names the door or data structure
it redesigns; the verified items it closes are its section of the ledger
([`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md)),
each with its evidence pointer and its verifier's verdict. A release reads
its owning contracts end to end and its code whole before it designs; an
item whose only fix would be a local patch means the root cause is not yet
found.

**How a release before F10 is checked** *(the user, 2026-10-31: "i feel ing
is e2e is only needed after f9. all others should focus on specific api and
validation tests")*: F0–F9 by the fast batch and each change's own API and
validation tests, seen failing on the old code — no e2e file; a change whose
only test is an e2e is named untested until after F9. The e2e runs again
from F10 on, once F9 has made the tests through the road. Each round of
review reads the involved scripts, modules and documents whole, and each
topic's test is confirmed present before the release ships *(the user,
2026-10-31)*.

| # | release | the redesign — what becomes true | items | needs |
|---|---|---|---|---|
| **F0** | **Wrong answers now** | **F0a released as 2.56.2** — 18 of its rows closed and 2 in part (the ledger marks each): status's next step from the run and ready doors, the run's clock read at the reader and from the run's record, the ending reader's crash an answer the wrapper says once, prep's writes leaving nothing it no longer writes, what 2.56.0–2.56.1 left, the documents that made a person act wrongly. **F0b released as 2.57.0** — B2-1, B2-2, U1-2, U2-2, U2-4, I2-134 closed, and what its reviews and checks found (the ledger's *Found by F0b*). **F0c released as 2.58.0** — SWEEP-RELAX and SPIN-EF, and what its review rounds found (the ledger's *Found by F0c*). The whole row: Each fixed at its owning door, not patched: **a species is a label and an element** — one shape through the structure codec's writer and reader and the file catalogue's pseudopotential field (a structure holding `Au1` is written and then refused, B2-1, B2-2); the composed junction keeps its view fields through `Structure.replace` (extending a junction whose order is unstated answers 500, U1-2); **one *IR asked* reading** of a result, older files included (a PySCF result's IR column hidden, U2-2, U2-4, U2-5); the build route returns the builder's findings (the peptide's aldehyde warning lost, I2-134); the run-ending reader's crash an answer, never *no* (I2-133), and the diagnostics door's (B3-28); prep's writes of a stage leave nothing it no longer writes (a stale Makov–Payne script, I2-199); status's next step from the run door (a missing member said finished, a sequence-sent group said not launched — D1-JSa-1, B3-31); the run's clock read once (U2-3); a SIESTA displacement sweep that relaxes reaches its comparison (SWEEP-RELAX — its design first) and a spin-polarized optical frame keeps its Fermi level (SPIN-EF); what 2.56.0–2.56.1 left — *not run* wherever an ended walk's never-run is said, `.continued-from` bytes from `runrecord` alone, migrate's write into a finished run stated as the rule's exception, the package pyflakes-clean; the documents that make a person act wrongly (installation's commands, the run script started by hand, the restart route's rule, the thread default); and the way out of a refused junction citation (decided 1). | 30, 18 closed 2.56.2 and 2 in part; + 2 (F0a's review) | — |
| **F1** | **The results views** (§ 1ak S3) | **F1 released as 2.59.0** — its rows closed (the ledger marks each), B3-11, I1-107 and I1-113 in part, and what its review rounds found (the ledger's *Found by F1*); B3-10 → F3, I2-087 → F9, the ready door's re-reads → F7, the transport report's and Task setup's path joins and Reload's dropped pick → F8. The whole row: `point_of` (floor 4), `points_view` and `views_of` (floor 6) built; the run door answers its own folder and nothing more; `/api/results/dir` returns the folder answer and the views; **every reader of a run of points' points asks the one point view** — the transport record, its digest and DOS-per-orbital reader, `summarize`, the report page, the ladder. Closes a run-once transmission whose DOS per orbital no road reads (U2-1), the I–V table's missing *not run* row (U2-21), a sweep's run folder that opens nothing (B3-34), the run door reaching floors 3–4 and walking a sweep's points 4N+1 times per answer (B3-1…B3-8, B3-33). Settles the user's open question *"build S3, or state the live road"* — the finish line builds it. | 34 | — |
| **F2** | **§ 1am step 2 — the run script names no engine** | **F2 released as 2.63.0** — its rows closed (the ledger marks each), B2-29 → F5, U2-15 → F4, I2-208 → F3, and what its review rounds found (the ledger's *Found by F2's review rounds*). The whole row: one skeleton over the engine record — its program, environment and GPU environment, one process or ranks, keyword deck, the step after the engine; the parse layer asked by meaning. Closes the PySCF banner promising a checkpoint and the rest of B1-1…B1-16. **Before it, 2.60.0 — a run is one start** *(the user, 2026-10-31)*: the in-place warm retry, the try counter, the run script's `--run`/`--cold`/`--force` and its folder sweep removed, which closed B1-2, B1-3, B1-4, B1-5, B1-10 and B1-16 by removal (the ledger's *Found by one start per run*). **Then the whole-jobset review** *(the user, 2026-10-31)*: Release A, 2.61.0 — a run's end known wherever it runs, older runs renamed, only `jobset launch` starts a run — and Release B, 2.62.0, the rest of its rows (the ledger's *Found by the whole-jobset review*). The design set aside on 2026-10-29 is its starting point, re-read against the code first; architecture § 3.3 and job-contracts § 2.6 are rewritten to it. | 34 + 2 (F0a's review) | — |
| **F3** | **§ 1am step 3 — the engine record and the kind/pair record whole** | **Two releases** *(the user, 2026-11-02: "go ahead with two releases, names ok")*: **2.64.0 released** — the engine side: the engine record's `validator`, `same_calculation`, `longest_label`, `keyword_reader`; the settings check asked once, in order; the output readers by role; the run script's record by its step, its program from its deck, one restart reader, and the bundle's members the records name — a run of points' words, an engine's keyword reader — taken from the records (the ledger marks each row; *Found by 2.64.0's review rounds*). **2.65.0 next** — the kind side and the pages. The whole row: An engine's validator, reading of *the same calculation*, label bound, words and environment are its record's parts; a kind's rung builders live in its own module, and its sharing role, scratch status line, description checks, extension and citation parts, sort key and report-field gate are its record's; § 1ai's two locks of the Structure-optimization tab move into the catalogue; the page's label palette served from the kind table (decided 6). Closes B1-17…B1-52, B3-10. Carries § 1ak's decisions: S2's reach (every branch on a kind's or engine's name judged into the tables, an engine's own modules choosing by kind staying), `describe` explicit, one record per kind and a part per engine. | 47 + 1 (F0a's review) | — |
| **F4** | **§ 1am step 4 — the file catalogue and the record doors** | A `ROLE_*` constant for every catalogue row and `compose` refusing a role no row has (131 literal roles in 41 modules today); each record's path formed by its owner — the machine record, the description, the job request, `.copied-from`, a launch group's files, the runtime record, the reports folder; nothing climbs to a root. Closes B2-3…B2-22. | 29 + 7 (F0a's review) | — |
| **F5** | **§ 1am step 5 — one codec per format** | Our JSON through `persist`; each of our formats one vocabulary module its writer and reader share — the trajectory log, the monitor's lines, deck blocks, the structure pair's sidecar envelope, a NetCDF file's JSON attributes, the run's index, the pidfile; each foreign format read in `parse/`. Closes B2-23…B2-35. | 25 | — |
| **F6** | **§ 1am step 6 — one home for each vocabulary, default and rule** | The run states named beside `RUN_STATES` and every set read from there; the count's tallies once in `runpoints`; the job kinds, a request's words, defaults, a kind's and an engine's words, and each sentence composed in one place. Closes B3-17…B3-24, B3-35. | 45 + 11 (F0a's review) | — |
| **F7** | **§ 1am step 7 — the floors and the objects** | The 15 upward imports moved down (`selection` reaching the `spectra` package among them); a module holding two floors' parts split; objects passed whole; private names made doors or left alone; rename-only wrappers and dead code deleted; the structure API's rebuilds keep every field (the A1 rows). Closes B3-11…B3-16, B3-25…B3-43, D6-7. | 26 + 10 (F0a's review) | — |
| **F8** | **The pages — words and residue** | Every word a person reads plain: no machine token or code id, a code span drawn as code by one shared helper, help no longer than the person needs, one count said one way, a frame numbered from 1 wherever a person reads it (decided 7); the residue gone. Closes U1's and U2's words, 2.55.1's leftovers. | 103 + 3 (F0a's review) | — |
| **F9** | **Tests through the road** (§ 1ak S4) | The support helpers make their inputs through the road; the tier-1 spectrum tests read a run's result, never a hand-filled one; no launch record or restart file planted; the tests owed written; the tests of older releases' files retired; the harness cannot report green when it is not; the sent-run pair kept with its reason stated (decided 10). Closes B4-1…B4-26 and the TEST-AUDIT rows. **And the 2026-11-02 test audit's cleanup** ([`archive/audits/2026-11-02-test-audit`](?doc=archive/audits/2026-11-02-test-audit/README.md); the ledger's *Found by the test audit*: TA-DOCS, TA-RETIRE, TA-REWRITE, TA-ISOLATE) — its 22 files retired and 50 rewritten through the road, with the single tests each group table names *(the user, 2026-11-02: "keep the plans order on the conditions that (1) this would not happen in current/new tests (2) clean up is scheduled")*. Condition (1) is the guards that keep every test off the person's machine, landed before F2 ([`process/testing.md`](?doc=process/testing.md) § 2b): a new test is inside them from its first run. | 62 + 8 (F0a's review) + 4 (the test audit) | — |
| **F10** | **The documents true** | Each chapter rewritten against the code once — the untrue and stale sentences corrected, every link resolved, history inside a contract removed and each reason it carried kept as a rule; a chapter a release F0–F9 touches is brought true in that release; README R5 says why `--bundle` keeps its name (decided 8). ~300 items over 70 chapters. | 305 + 1 (F0a's review) | — |
| **F11** | **The documents findable** (§ 7) | § 7's four indexes; `docs/README.md` the one index naming every live chapter (nine have no row today) and each domain overview listing its own; one place for each thing now stated in several. | 2 + § 7 | — |
| **F12** | **Proven** (§ 1am step 8) | Each road — an optimization, a vibration on SIESTA and on PySCF, a transport calculation — driven start to finish in the browser on real runs; the full batch; two review rounds over F0–F11. | 8 | — |

**How a release closes** *(the user, 2026-09-27: "a consolidated persistent
plan that get's updated at every milestone and reviewed with agent full code
text review to validate results")*, in this order:

1. its owning contracts read end to end and its code whole; the design
   written into the contract first (a design change waits for the user's
   yes); the code; a test that drives the road, broken on purpose once and
   seen to fail;
2. the fast batch and the targeted tests; the package pyflakes-clean;
3. **two review rounds** — every finding re-read before it is acted on:
   correctness, by readers of the contracts and the changed code; and
   **holistic** *(the user, 2026-10-31: "make sure your review on milestones
   will include holistic code evaluation on api and framework unification,
   and detect duplicated/redundant/handcrafted code")*, by readers who each
   take whole files of a coverage table of every touched module — one door
   and one API per fact, one data shape, and what is duplicated, redundant or
   handcrafted, in the code and in its tests; a finding this release does not
   own goes to the release that owns its door;
4. the reviews' fixes;
5. **one e2e run, last** *(the user, 2026-10-31: "when e2e tests should be
   called need careful judgement. we don't want repeated waiting")* — the e2e
   files whose roads the release's code touches, named with why; the whole
   e2e batch only for a change every road passes through, said so first;
   never while a review of the same code is open;
6. this plan brought current in the release's commit — its row's state, its
   ledger rows marked with the release that closed them, what it found placed
   under the release that owns it, nothing archived unchecked *(the user,
   2026-10-31: "your plan should be persistently updated, and obsolete items
   should be correctly validated, and archived so you have one unified clean
   plan")*; the release tagged `v<version>` and the tag pushed (conventions
   § 2; decided 9); then the next release.

---

## DECIDED *(the user, 2026-10-31: "go with your recommendations")*

| # | the question | decided | where it goes |
|---|---|---|---|
| 1 | A junction citation the describe door already marks refused (U1-1): Describe writes the folder, prep refuses it, and citing again into that folder is refused | the refusal stays at prep (*prep decides*); the way out is made to work | F0 |
| 2 | The half-built run-record pieces (§ 5t P2, P4; § 5s P5, decision 5): the engine-echo column, asked against ran, a vibration record's machine rows, the device contour, divergence signs, the read-back of charge and spin | after the finish line (the warm retry was removed at 2.60.0: a run is one start) | *After* |
| 3 | N11 release 2 — the page split (Transport tab the physics editor, Task setup the running) | after the finish line | *After* |
| 4 | Features the three roads do not need (11 rows: the per-trial SCF plot, the markdown editor, a notification's `ttl`, multi-node MPI, `detect()`'s reasons, a `checkpoint verify` verb, …) | dropped (the user's rule of 2026-10-18); their rows stay in the ledger as a record | — |
| 5 | N12 group 4 (9 rows, its text in no document) | dropped | — |
| 6 | The label palette's own copy of the lead names | served from the kind table | F3 |
| 7 | Frame numbers: server sentences say *frame index 0 (of 3)*, the page's rows *Frame 1 of 3* | 1-based wherever a person reads; 0-based only in files | F8 |
| 8 | `--bundle` on every `jobset` verb, a word README R5 retired for the calculation folder | the option keeps its name, and R5 says so (a rename would break the user's scripts) | F10 |
| 9 | Release tags — conventions § 2: every release gets a git tag; the last was `v2.6.0` | each release is tagged again, from the next one | *How a release closes* |
| 10 | The sent-run test pair (B4-2): two basic tests write a launch record by hand, because the basic suite never submits | kept, their reason stated in them (testing.md § 6) | F9 |
| 11 | How a builder hands back its findings (I2-134; `validation.md` § 4.1 R5: a finding is an `Issue`, never a Python warning) — (a) each builder returns one answer, structure, backend and findings, its Python API changed; or (b) the builders return the structure as now, and one check after the build says the findings: the structure's own (an aldehyde end, a duplex clash) read off what was built, a backend's limits (a form it cannot enforce, a terminal it ignores) a table of what each backend can do, shown on every builder panel through the one renderer | (b) *(the user, 2026-10-31: "go with b")* | F0b |

**Still the user's words** (no work waits on them): a result written before
2.49.0 keeps its stored Methods sentence beside the free atoms' sentence;
`PLACE_NOTE_UNKNOWN`'s words; `envs/recipes.py`'s comment calling ASE a monitor
reader.

Settled since it was asked: **build S3, or state the live road** — the finish
line builds it (F1). **The citation door's *states no cell* refusal**, said to
be reached by no road — a road may reach it (a saved relaxation whose cell is
cleared, then cited); F1 walks it on the dev server before anything is
decided.

---

## AFTER THE FINISH LINE

- **New science** — each brought to the user as a design with its science
  first: § 1ag (corrections to a junction's electronic structure), § 1ah
  (which modes get Raman), M10 (from a mode to the current), § 5e (a person's
  own engine text), § 9.2 (ATOM), S13 (transport's convergence sweep), K22
  (per-atom starting spins), LEAD1, the V1 rows, § 1ac's D-h, B and the
  measured intensity's factors — the ledger's *After — science* (28 rows);
  and F0c-R1-S, a fixed-spin optical frame's ε₂ filled against one Fermi
  level (SIESTA's, read in its source, not measured — the ledger's *Found
  by F0c*).
- **Designed, not built, and after it** *(decided 2, 3)*: § 5t's and § 5s's
  run-record pieces (the ledger's *Waits — run record* rows); N11 release 2,
  the page split.
- **Only the user's Sol runs can show** — E13, SCHED-HDR, BENCH-SHELF,
  SCHED-R, § 1ac's proofs 3a and 3b, the PySCF environment's `netcdf4`, the
  gcc 15 pin — the ledger's *After — Sol* (13 rows). Measured on the user's
  AuBDTAu Raman run of 2026-10-28: about two hours an optical call, nearly all
  in SIESTA's band-pair loop.

---
## Where the earlier sections went *(2026-10-31)*

Each section below was replaced by *The one plan*; its text as it stood is in
[`archive/2026-10-31-plan-consolidation.md`](?doc=archive/2026-10-31-plan-consolidation.md),
and every open item it held is a row of the ledger
([`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md))
under the release that closes it. A reference to one of these numbers reads
on here.

## 0a. The work order — now *The one plan*
The queue and THE ORDER's steps 2–6 are F0–F12; *How a release closes* keeps
§ 0a's milestone rule.

## 1am. One framework: one layout, then every fact through its one door — now F2–F7 and F12
Steps 1 and 1b built (2.54.0, 2.55.0); step 2 is F2, step 3 F3, step 4 F4,
step 5 F5, step 6 F6, step 7 F7 (with § 1ak S3 in F1), step 8 F12.

## 1ak. One system for a calculation's files — now F1, F3, F9 and F12
S0–S2.3 built (2.47.6–2.47.12); S2.4's rest is F3; S3 is F1; S4 is F9; S5 is
F12. Its decisions (S2's reach, `describe` explicit, p3 rebuilt on S3, the kind
table a data file with one reader, one record per kind and a part per engine,
`run_answer` on floor 6) stand, and are carried in F1 and F3.

## 2. OPEN — the one list — now the ledger's I2 rows
Each row settled against the code at 2.56.1 and placed under its release, or
closed with its reason.

## 1t, 1u, 1w, 1ae, 1af, 1al, 1ac, 1aa, 5q, 5s, 5t, 5w, 8 — now the ledger
§ 1t, § 1u and § 1w built, their parked items in the ledger. § 1ae and § 1al
built; § 1al's F3 built 2.52.0. § 1af's design lives in
[`execution/run-reports.md`](?doc=execution/run-reports.md) § 2, its items in
the ledger. § 1ac's proofs are *After — Sol*, its science *After — science*.
§ 1aa's release 2 is after the finish line (*Decided* 3). § 5q and § 5r.1 are
F7's; § 5s and § 5t are after the finish line (*Decided* 2); § 5w's K rows are F3, F5 and F6's; § 8's
rows 10, 11 and 13 are F5's and F10's.

## 1ad, 5ab, 5u, 5aa, 5z, 5v, 5x, 5y — archived earlier
Their pointers are under *Archived sections* below.

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
intention; a plan kept current describes the work.**

**And R3 now means what it says: there is ONE table** *(2026-09-10)*. Until
that day the open items sat in six tables across §§ 2, 3, 4, 4a, 5, 5b, 5f, 5l
and 5m, which is how three different rows came to be numbered `P3`, how `W8`
could cite line numbers for three endpoints that are not registered, and how
`W9` could carry one sentence naming no behaviour for weeks. **The one plan
is the list.** A design section keeps the *why* and sends its rows there; a row that
is done, withdrawn or measured untrue leaves for
[`archive/2026-09-10-plan-consolidation.md`](?doc=archive/2026-09-10-plan-consolidation.md)
the same day, with what it turned out to be.

---

## 1ag. Corrections to a junction's electronic structure — a framework before any one correction *(2026-10-27; planned, not designed)*

*(The user, 2026-10-27: "let's put this deltaH, IP/EA design into a plan
instead of directly build it now because we need a better framework to
integrate such consideration rather than one arbitrary parameter".)*

**The question it answers.** PBE places a molecule's frontier levels too
close to the electrodes' Fermi level, so a junction's conductance comes out
high: AuBDTAu (thiolate on flat Au(111), three-fold sites) reads
T(E_F) = 0.225 G₀, its HOMO-derived resonance at −0.68 eV, against Kaneko
2016's 0.02 G₀ (bridge-tip state; JACS 138, 1294). The corrections in the
literature all act on a converged device Hamiltonian, and TBtrans takes any
of them as a correction to it — `TBT.dH`, a NetCDF file sisl writes
(`deltancSileTBtrans`, eV in, Ry stored; SIESTA 5.4.2
`Util/TS/TBtrans/m_tbt_delta.F90`, manual `tbtrans.tex` § δH) — so only the
transmission re-runs, never the device SCF.

**Why a framework, not a parameter.** Several corrections share one road and
differ in what they move and why:

| correction | moves | its inputs |
|---|---|---|
| DFT+Σ (scissor; Neaton 2006, Quek 2011, Mowbray 2008) | occupied down, empty up — the gap opens — less the image charge | IP, EA and the same functional's gas-phase HOMO/LUMO of a reference molecule; the image plane z₀ |
| a rigid shift (a non-self-consistent gate scan) | every molecular level by one Δ | Δ, or a range of them |
| an optimally tuned hybrid's levels (Egger 2015) | each level by its own gas-phase difference plus the image term | a gas-phase calculation of another functional |

A framework names, for every correction: the subspace it acts on (a label
the person names), the projection (δH_MM = Σₙ Δₙ S_MM ψₙψₙ† S_MM with
ψ†S_MMψ = 1, exact in a non-orthogonal basis), the levels it moves and by
what, each input's unit and source, the terms molbuilder derives from the
geometry (the image term, P = 3.600 eV·Å / (z − z_surface − z₀) per
electrode), and where the record keeps it — the deck, the data file's
settings, the report's curve label, the Methods sentence.

**What to design, in order** (contract first, `engines/transport.md`):
1. The correction as a record: its kind, inputs with units and sources,
   derived terms, and the δH's sha256 — one writer in the job's python
   (sisl, in the SIESTA env), one reader.
2. Where it sits: the transmission rung, and an extension of a finished
   calculation changing only that tab (§ 2a.16 road, built 2.33.0).
3. Inputs from another calculation — a gas-phase reference's IP/EA and
   levels — cited as a calculation of molbuilder's or typed with a source.
4. The comparison: corrected and uncorrected T(E) side by side.
5. The help each correction carries: when it applies (weakly coupled, levels
   away from E_F) and when the papers call it unjustified — a thiolate on
   gold (Mowbray 2008, Liu 2017, Souza 2014) — stated, never refused.

**Facts already verified** (2026-10-27, the agent's report, DOIs
Crossref-checked): z₀ = 1.0 Å (Quek 2011; Souza 2014: 0.79–1.13 Å), 0.9 Å for
Au(111) (Egger 2015); no measured IP for 1,4-benzenedithiol (computed: PBE
ΔSCF 7.5 eV, EOM-CCSD 7.8–7.9 eV; PBE HOMO −5.1 eV — Strange 2011, Bâldea
2014); PBE ΔSCF IP error 0.16–0.24 eV (Ernzerhof & Scuseria 1999, Rostgaard
2010), EAs unreliable with semi-local functionals; Kaneko 2019 shifted BDT's
HOMO by 1.08–1.17 eV and gets 0.029 G₀ (bridge) against 0.024 measured, its
split into the two terms unreported. On AuBDTAu the image term is 1.4–1.8 eV
at the ring centre for z₀ = 0–1 Å, and its 1/d diverges at the sulfurs —
the classical picture's limit for a bonded atom. The step with the best
support for the 0.02-vs-0.22 question is the structure: the thiol hydrogens
kept, or Au adatoms (Strange 2011: GW 1.0×10⁻² G₀ with H kept, 0.83 for the
thiolate; Inkpen 2019: H retained under common conditions) — ordinary
transport calculations molbuilder runs today.

## 1ah. Which modes get Raman — after the run, and by symmetry *(2026-10-27; planned, not designed)*

*(The user, 2026-10-27, on *Raman for*: "because this is done before any
calculation how could there be anything to select from?"; "is there a way
to exclude obvious ones that is insensitive? or there is no general
simplification?")*

Before a run there are no modes, so 2.45.0's *Raman for* offers what can be
decided then — all modes, or a frequency range a chemist knows
(`engines/vibration.md` § 3.1). Two ways to spend less, each its own design:

1. **Pick modes from a finished spectrum.** A vibration whose force
   constants are done, extended with the optical calls of the modes a person
   ticks on the Spectra page — only those run, on the run's own modes; the
   extension road of a finished calculation (`engines/transport.md`
   § 2a.16) for the vibration kind, the run's index already planning optical
   calls after `fc` (§ 5.11 § 3). **Asked on a live run** (the user,
   2026-10-28, of the AuBDTAu Raman run — 121 optical calls at about two
   hours each): can a continuation shrink the modes to save time? Not today —
   *Raman for* is fixed once its stage has run, and a warm relaunch refuses a
   change to what the run computes; this design is the answer, offered and
   waiting for the user's go.
2. **Skip the modes symmetry forbids.** A mode's Raman is exactly zero when
   its symmetry forbids it — CO₂'s antisymmetric stretch and two bends, 3 of
   its 4 modes (our run: 0.23 against 45.1 Å⁴/amu, zero to numerical noise),
   9 optical calls cut to 3. To design: the point group found, each mode's
   symmetry species with a tolerance for numerical noise, a forbidden mode
   recorded *not computed: forbidden by symmetry* — never a silent zero —
   and which library in the SIESTA environment does it. It saves nothing on
   a junction, which has little or no symmetry; the frequency range stays
   the general cost control there (above about 300 cm⁻¹ leaves out the slow
   gold modes).

## 1ai. The Structure-optimization tab's two locks into the catalogue *(2026-10-27; planned)*

2.45.0's `active_when` greys a field out from one catalogue declaration,
and the settings gate says the same (`web/form-schema.md` § 1.3). The
Structure-optimization tab still keeps two locks of its own in its page
script (`structure-optimization/viewer.js`): a relaxation's step settings
under *Relaxation type* `none`, and the solvent model with no solvent —
which the gate never mentions. To design: `active_when` naming a value that is
*any one given* (the solvent's blank is its gas phase, not a member to list).
A vibration's relaxation settings stay live under *Structure is already
relaxed* — the person may relax again with other settings; the choice is
theirs, never greyed *(the user, 2026-10-27: "structure that is already
relaxed should not stop user from relax with a different set of parameter,
the user's choice for this, no greying")*.


## 5e. Engine additions — a person's own engine text, as an INPUT

*(Your ruling, 2026-09-05: option 2 — a distinct input to the writer, not a
catalogue extension; **engine-specific**; and **the person is told the
consequence**. Contract-first: this section is the design. **Not started.**
No code has moved.)*

*Re-checked 2026-10-09:* the measurements hold in shape, at new lines —
`render_deck`'s concatenation is `script_emit.py:1429-1431`. Two of the known
defects named below have moved: a stray marker no longer drops text silently —
`check_deck` refuses a deck holding more than one pair of USER-CUSTOM markers
(`script_emit.py:1567-1588`); and transport has the zone — its decks are
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

`render_deck` assembles the deck like this (`script_emit.py:1429-1431`):

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
   (`siesta/layout.py:451`, its refusal `:512-530`) splits the whole file and knows nothing of the
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
path. They do not, so those mechanisms are untouched; the three defects this
draft named against them are closed since (the 2026-10-09 note above).)*

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
(`script_emit.py:1429-1431`'s concatenation, the 12/520/980-vs-1222 positions, the
reproduced duplicate-keyword refusal, `Parameter.writes` — `script_emit.py:806`) are re-checkable, and
should be re-checked rather than trusted if this is picked up later.

## 5r. The structure API — the confirmed non-findings *(its order, 5r.1, is F7's ledger section now)*

### 5r.2 Do NOT "fix" these — confirmed non-findings

Recorded so nobody fixes them by analogy. Re-read against the code 2026-10-08.

- **`frozen_atoms`'s shape is benign.** A data descriptor sends every read and
  write through `regions[FROZEN_LABEL]`, so the two cannot disagree in any
  order; `replace()` handles it by not re-passing it. **Not the defect `pbc`
  was** — the init-field spelling exists so `Structure(…, frozen_atoms=[…])`
  reaches the one place that spells the reserved label.
- **`resolve_cell` is not the one resolver**:
  `transiesta._compute_cell_from_extents` (`transiesta.py:184`, called
  `:346` and `wizard.py:400`) computes one for a transport lead — X4 ③.
- **The companion-lookup merge stays withdrawn.** What forbids it is a test
  pinning a deliberately more permissive guard, not `parse.md` § 5.3; a shared
  helper with the guard as a parameter would keep both.
- **The second multi-frame XYZ reader is forced**, not duplicated: it
  tolerates a torn final frame (`parse/engines/pyscf.py`, ~`:300-355`), which every
  live geomeTRIC run has and ase refuses. Unifying it would break live-run
  viewing.
- **Registry overlap: none** — 14 parsers against a 29-file synthetic run
  directory, no file claimed twice.
- **Bare atom-index arithmetic in `parse/`: none that carries data**, and
  the transport half is done: since 2.32.0 `transport/transiesta.py`
  numbers atoms through `engine_atom_index.siesta_numbering`
  (`:35, 579-582, 674-675`).
- **The CLI's seam verdict** was raised and dropped on the user's ruling
  (2026-09-22: *"people working with CLI would know what they're doing… leave
  that out"*). X3 is the browser half only.


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
routes do not exist and **9** live ones are undocumented, against **86** unique `/api` routes live on 2026-10-09 (about 93 by a decorator count on 2026-10-28; 86 on 2026-10-18 —
74 `@bp.route`, 10 `@bp.get` / `@bp.post`, 6 on the app itself, `app.py:658-718`, `rate_limit.py:565-571`; a naive path diff of `web-api.md` against the live routes on 2026-10-18 gives 19 documented-not-live and 26 live-not-documented, the checkpoint, files, modify and workspace-storage families among the latter — the exact counts depend on how the document names a family, re-derived when step 2 is taken)
(`app.py:641-701`, `rate_limit.py:565-571`); the 71 counted only the first kind. And coverage was never truthfulness:
the 2026-09-29 check read every live document against the code and found
statements the code contradicts — § 5v holds that list, and the survey starts
from it rather than repeating it.

**Findability.** **83** live documents, **83,077 lines** (2026-10-28; 79 and 73,853 on 2026-10-18), nine directories
(`docs/` and eight below it; `archive/` and `plans/` excluded). The
information is there and a person cannot reach it.

**The existing index has the problem it is meant to solve.** `README.md`
§ Index (`docs/README.md:199`) is one row per DOCUMENT, and the longest row
runs about 828 words (`:240`, 2026-10-28). To find out which document owns a fact you read
essays about documents. It is a good map of the tree and a poor answer to
"where is X".

**Two documents contradicted themselves in one day** (2026-09-07), both found
by accident while doing something else — `science/pseudopotentials.md` § 3
listed a check as not-done that its own § 2a.1 documented as done (since
corrected; § 3's note at `:468-479` records the removal), and
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
genuinely scientific. The worked example: `constants.py:106` defines
`HARTREE_BOHR_EV_ANGSTROM_ASE = 51.42208619`, the ASE/NIST value, and says why
the name carries the convention — `HARTREE_EV / BOHR_ANGSTROM` gives
51.422067476, 0.36 ppm apart, and a force and the threshold line drawn over it
must use the SAME one. The emitter imports it (`trajectory_log/emitter.py:44,
210`) so emitted forces line up with what a person reads in ASE, VASP and QE
logs. That is a real convention choice with a real justification, and it lives
in a code comment nobody will find.

Citations are split the same way: `science/references.bib` holds **80**
entries (2026-10-28; 34 on 2026-10-09), **11** live documents cite it, and the
rest cite literature in prose, if at all. Stokbro 2003 joined it in 2.46.3,
and no live document cites Reed 2006 now (E14, closed 2026-10-28).

So the reference index carries three things per row: the value, where it is
defined, and the sentence saying why that value and not the neighbouring one.

### 7.4 Order, and the one risk

1. **Survey first, and record findings without fixing them** — **done as
   § 5v** (2026-09-29): every live doc read against the code, findings in one
   list, an entry leaving it when its document is corrected. The route sweep
   (§ 0a *Unscheduled*) is the same kind of finding for `web-api.md`.
2. **The two generated indexes**, which are cheap and cannot go stale. Inputs:
   the live routes (about 93 on 2026-10-28); `architecture.md` § 3's module list.
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

## 9. ATOM as a pseudopotential validator — § 9.1 DONE, § 9.2 PENDING, not started

*(User, 2026-09-19.)* **§ 9.1 — done, archived 2026-10-08**: the user's two
rules — the pseudopotentials are explicitly provided, and the file for element
`E` is named `E.psml` — are refusals when the calculation is configured
(`validation/siesta.py:20`, `:65`; `pseudos.py:75`; contract
`science/pseudopotentials.md` § C1a, `:140`).

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
| `x3dna*/` in `.gitignore` | `atom*/` likewise, added (`.gitignore:88`): the ignore has to exist *before* the folder does, or a redistribution-prohibited package lands in `git status` |
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

- [`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md)
  — the finish-line review of 2026-10-30: every verified finding and every open item
  settled against the code at 2.56.1, by the release that closes it.
- [`archive/2026-10-31-plan-consolidation.md`](?doc=archive/2026-10-31-plan-consolidation.md)
  — the handover's queue, decisions and Sol paragraphs; §§ 1t, 1u, 1w, 1ae, 1af, 1am, 1ak,
  1al, 1ac, 1aa, 0a, 2, 5q, 5r.1, 5s, 5t, 5w and 8 and the archived stubs, as they stood
  on 2026-10-31 — replaced by *The one plan*.
- [`archive/2026-10-28-plan-consolidation.md`](?doc=archive/2026-10-28-plan-consolidation.md)
  — §§ 1ad and 5u whole, done; §§ 1ae, 1ac, 1ak, 5q, 5r, 5s, 5t and 5w as
  they stood, partly done; the status paragraphs, the handover of 2026-10-27,
  § 0a's NOW queue and its settled decisions as they stood; § 1af's
  environment paragraph; § 2's RL42, GATHER-WARM, E14, V1.9, V1.14 and GKN
  rows — each under its verdict, read against the code on 2026-10-28.
- [`archive/2026-10-27-plan-consolidation.md`](?doc=archive/2026-10-27-plan-consolidation.md)
  — §§ 1s, 1v, 1x, 1y, 1ab, 1z and 1aj whole, done; §§ 1t, 1u, 1w, 1aa and
  § 1ad.1 as they stood, partly done; the handover of 2026-10-20 and § 0a's
  NOW queue as they stood; § 2's TPL-L1, DOC-TOC, LEAD-TOL, RL15, V1.25 and
  A1.9 rows — each under its verdict, read against the code on 2026-10-27.
- [`archive/2026-10-21-1s-job-request.md`](?doc=archive/2026-10-21-1s-job-request.md)
  — § 1s's job-request design record.
- [`archive/2026-10-20-plan-handover.md`](?doc=archive/2026-10-20-plan-handover.md)
  — the handover of 2026-10-18, verbatim.
- [`archive/2026-10-14-plan-rows.md`](?doc=archive/2026-10-14-plan-rows.md)
  — the rows done in 2.9.5–2.9.6.
- [`archive/2026-10-14-rmp-design-rounds.md`](?doc=archive/2026-10-14-rmp-design-rounds.md)
  — the remapped state's design rounds.
- [`archive/2026-10-18-review-findings.md`](?doc=archive/2026-10-18-review-findings.md)
  — the seven-lane review of 2026-10-18: every finding with its verdict
  (fixed, rewritten, a decision, judged).
- [`archive/2026-10-18-5ab-remapped-state.md`](?doc=archive/2026-10-18-5ab-remapped-state.md)
  — § 5ab (the remapped state, RMP) whole, as it stood when the plan was
  cleaned on 2026-10-18.
- [`archive/2026-10-18-plan-rows.md`](?doc=archive/2026-10-18-plan-rows.md)
  — the rows of § 2 done, built, closed, retired or answered between
  2026-10-14 and 2026-10-18 (2.10.0–2.15.4), each verbatim under its verdict.
- [`archive/2026-10-18-plan-consolidation.md`](?doc=archive/2026-10-18-plan-consolidation.md)
  — the handover's push-history paragraphs, 2.0.0 to 2.13.2, as they stood
  when the plan was cleaned on 2026-10-18.
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

