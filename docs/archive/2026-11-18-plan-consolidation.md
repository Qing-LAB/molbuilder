# Plan consolidation — 2026-11-18

The sections of `plans/plan.md` whose work was done, superseded or kept only as a record, moved here on 2026-11-18 *(the user: "clean up the task list/job list, consolidate plan, archive appropriately")* — each verbatim under its heading, its verdict in front. What stays open was carried back into the plan.

---

## The plan's header, as it stood

*Verdict:* **Rewritten.** The new header names the 2026-10-31 consolidation and this one; the earlier consolidations are the plan's *Archived sections* list.

# The plan — one file

**Role:** plan — **the only one.** Every open item lives here or in the
ledger a release names; the nine plan documents that preceded it are archived
under `docs/archive/2026-09-01-*.md` as records of what was decided and built.
**Domain:** all
**Started:** 2026-09-01, by consolidation

> *(user, 2026-09-01: "We don't need ten plan files scattered. We want one
> plan folder or one plan file and stay with that file.")*

**Consolidations** — each moved what was done or settled, verbatim under its
verdict, to its archive (*Archived sections*, at the end). The one this plan
stands on is **2026-10-31** *(the user, 2026-10-30: "consolidate with previous review
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

## The handover of 2026-11-18, at 2.77.4, as it stood

*Verdict:* **Done.** Its three bodies of work — the UI review (2.68.0–2.69.9, 2.70.27), a transport result's export (2.70.32–2.76.1) and the overlap audits (2.69.10–2.70.14) — closed by the releases its changelog rows name. Its open items — the vibration data files' copy loops, F9's low-harm last items, the page doors' rest and the test framework's TF3–TF10 — were carried back into the plan's *Open beside the finish line*, each read against the code on 2026-11-18 (`spectra/datafile.py`, `spectra/raman.py` and `spectra/optical_data.py` still copy with `key_of`; `jobset.runstatus._copied` still says its note in the copy's own field).

## ▶ HANDOVER — where we are, and where to start *(2026-11-18)*

**Where we are: 2.77.4.** F0–F11 and F7a are closed; F9's low-harm last
items wait (*Left, in order*, below). F12, *Proven*, has walked its three
roads in the browser on real runs, run the full batch and its two review
rounds; its fix releases run next (*What runs next*). Since 2.67.11 three bodies of work the user put first have run;
each release is its changelog row and its ledger section
([`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md),
*Found by …*):

- **The UI review** *(the user, 2026-11-08: "Right after 2.67.11")* —
  2.68.0–2.69.9, 2.70.27: done — the last two items, the tabs following
  the sidebar's folder and the ladder's layout, in 2.70.27.
- **A transport result's export** *(the user, 2026-11-16, after F10)* —
  three pieces, each designed in its chapter before code: (A) the
  calculation's Export moved into the Results tab's top line (2.70.32,
  done); (B) the data
  file in groups, `@9` — the junction whole (cell, regions, labels), each
  stage, each point with its device run, a summary across the points from
  the same pass, each run's engine version, how each DOS is made
  (`engines/transport.md` § 2a.12, designed; the post-processing
  document's examples move with it, each run on a real calculation) —
  built in 2.71.0; the PDOS of chosen atoms from the files built in 2.72.0,
  after two review rounds that also fixed 2.71.0 (the energies' names);
  its internals one door each in 2.72.1 (the user's discipline review);
  the framework across the kinds' data files in 2.72.2 -- one structure
  layout, one engine version, declared JSON records, axis coordinates from
  their rows, one schema door and one group walk (`model/datafiles.md`
  § 3a), with Python 3.12 declared as the floor; every data-file name in one
  catalogue and made by one naming rule in 2.72.3 (`model/datafiles.md`
  § 9, § 2a rule 2, § 2c; the user, 2026-11-16); one export door per
  kind, its asks declared in the kind table, one /api/export route and
  `jobset export` from the rows in 2.72.4 (`model/datafiles.md` § 5a);
  an older file's names read through one door (`ncdata.named_now`) and
  the transport summary opened through the one door in
  2.72.5 (`model/datafiles.md` § 2a rule 5, `model/parse.md` § 5e);
  one Export bar for every kind, the Results page's, drawn from the
  calculation's offer in 2.73.0 (`web/results.md` § 1.1; the user,
  2026-11-16: "Every kind"); `pdos`, the re-average, a reading's order
  record and a mode's lookup moved to their owners in 2.73.1, the
  post-processing layer adding no rule of its own (`model/parse.md` § 3);
  an optimization's and a vibration's Export holding every run's steps
  and each step's SCF cycles, one layout in groups, and every file's SCF
  history one layout, in 2.74.0 (`model/datafiles.md` § 9.5, § 9.0.6;
  the user, 2026-11-17: "Into the calculation Export", "Groups, as
  transport"); each chart's own Export — a picture, its drawn data as
  CSV and NetCDF — for the trajectory and transport charts in 2.75.0
  (`web/results.md` § 1.2; the user, 2026-11-17: "Go ahead"), and the
  vibration's charts and the bench chart in 2.76.0;
  the PDOS of a mixed selection from the data file and each point's digest
  (`postprocess.pdos`, the user's option 3, designed with B). The full-text
  review of the result summary and data export code *(the user,
  2026-11-16)*: two reviewers, 2026-11-17 -- its defects fixed in 2.76.1.
  Left from it, next: the vibration's data files (`spectra.datafile`,
  `spectra.raman`, `spectra.optical_data`) still copy a record's values
  onto their grid by their own `key_of` loops and encode and decode
  their record attributes by hand, where the transport file and a
  chart's use `ncdata.copy_stated` and `ncdata.json_attrs` /
  `read_file(records=)`; the status door says a copy note that does not
  read in the copy's own field (`runstatus._copied`), where the transport
  record says it as `unreadable`; the record's and the data file's
  builders are long functions.
- **The overlap audit** *(the user, 2026-11-09)* — 2.69.10–2.69.12: a
  structure file through the one writer, every answer through the one reader,
  every failure through the one builder.
- **The codebase overlap review** *(the user, 2026-11-09: "Accept all,
  defects first"; about 170 findings by owning door, the ledger's *Found by
  the codebase overlap review*)* — 2.69.13–2.69.68, closed: every science and
  correctness defect (V-*), the path door (X-PATH), record reading (V-READ,
  V-PANEL), X-STRUCT, X-PERSIST, X-PARSE, X-TMPL, X-EXEC and X-ENG; then the
  page doors' first stretch, 2.69.69–2.70.1 (O-STATE, O-FIND, O-STATUS,
  O-KEYS, X-UI), and the tests stretch, 2.70.2–2.70.14 (O-TESTS T0–T1, the
  test framework's TF1 and TF2) — each release's line in the changelog.
  X-ENG closed with 2.69.68 — its two items that change a generated
  program placed by the user with F10's end-to-end pass.
  **Left, in order** *(the user, 2026-11-15: "focus on real issues",
  "focusing on lib code"; asked which first, "Toward the finish line")*: the
  rest of F9 as its row orders it -- its page-side test doors done by
  O-TESTS (2.70.2–2.70.10), the dead-code release (2.70.20) and the tests
  owed on real runs (2.70.21) done; F9's closing review rounds -- the first
  done (2.70.22), its findings fixed or judged; its carried items done
  (2.70.23 -- the exported picture's resolution, a product defect they
  found, fixed); the second round's readers over the fast tier done, its
  two product defects and the bias window's floor fixed (2.70.24); its
  most harmful test gaps closed (2.70.25), the rest (2.70.26, which found
  the SIESTA vibration record's place in the same calculation); left of it:
  four page stand-ins without a named need (the Task setup save bar's
  refusal, the export door's answers, the save door's page, the checkpoint
  Cancel's signal) and the installer's repair audit -- low harm; then F10
  with its first full end-to-end pass, on the user's word. The page doors' rest -- O-STATE's
  one save door, O-WORDS, O-KEYS's rest with X-UI and X-PLOT -- is internal
  tidying no page shows (measured against `tabs.md` § 6 for O-STATE) and
  waits; so do the test framework's TF3–TF10 and the page tests on jsdom.

**Last checked** (2.77.2, F12's full batch, 2026-11-18): the fast tests
6106 of 6106, the e2e 805 of 805.

**What runs next** -- **F12's fix releases** *(the user, 2026-11-18:
"Harm first, by module")*. F12's two review rounds
([`archive/audits/2026-11-18-f12-review/`](?doc=archive/audits/2026-11-18-f12-review/TRIAGE-round1.md))
found, verified, 27 wrong results and 7 crashes (round 1, by module in
`TRIAGE-round1.md`) and about 395 one-home findings (round 2,
`TRIAGE-round2.md`), many of them the cause of a round-1 defect. Each
release fixes its module's round-1 defects through the one-home redesign
round 2 found for that module, every finding re-read when the module is
read whole: (1) the builders build what was asked — done, 2.77.4; (2) the
deck writes the value set; (3) the person's files safe, with the test
suite's refusing scheduler covering `squeue` and `scancel`; (4) a failure
said in words, never a traceback; (5) the pages show the folder and run
they say; (6) MolView; (7) the viewers; (8) the run script and the machine.
**No check or warning stops a build or a calculation** *(the user,
2026-11-18)*: the person decides; a warning is said and the run goes as
set. Then the rest
of round 2 module by module, then the 85 chapter sentences. Two answers
for them *(the user, 2026-11-18)*: a SIESTA vibration's result keeps the
structure's own notes, as PySCF's does ("Keep them, as PySCF does"); a
report-cards copy that does not read stops only the monitor of a run that
sends reports ("Only runs that send reports"). **Waiting on the user**:
nothing now.

**The test suite's guards** keep every test off the person's machine
([`process/testing.md`](?doc=process/testing.md) § 2b): test code loads only
under pytest, each test process runs in a sandbox of its own, and one write
guard covers the whole run.

**Off the plan** (the user, 2026-10-18): every feature the three roads do not
need; any check, warning or format the contracts do not state. The AuBDTAu
workflow is the user's, on Sol *(the user, 2026-10-10: "no, you should not
run it on this workstation")*.

---

## THE ONE PLAN — its introduction, the rule for a release before F10, and the release table F0–F12, as it stood

*Verdict:* **F0–F11 closed** (2.56.2–2.77.2), each by the releases its row names and the ledger's *Found by …* sections; F12 walked, run and reviewed at 2.77.2, its fix releases running. The rule for a release before F10 is superseded: F10 ran the first full end-to-end pass, and the e2e runs again in every release since. The plan keeps one line per release and F12's row.

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
| **F0** | **Wrong answers now** | **Closed** — F0a 2.56.2, F0b 2.57.0, F0c 2.58.0: a species a label and an element; the composed junction keeps its order; one *IR asked* reading of a result; a builder's findings returned; the run's clock, the ending reader and prep's stale writes; status's next step from the run door; a relaxing displacement sweep and a spin-polarized optical frame (SWEEP-RELAX, SPIN-EF); the documents that made a person act wrongly; the refused citation's way out (decided 1). The ledger marks each row (*Found by F0b*, *Found by F0c*). | 30, 18 closed 2.56.2 and 2 in part; + 2 (F0a's review) | — |
| **F1** | **The results views** (§ 1ak S3) | **Closed** — 2.59.0: `point_of`, `points_view` and `views_of` built, the run door answering its own folder only, and every reader of a run of points' points asking the one point view (the ledger's *Found by F1*). | 34 | — |
| **F2** | **§ 1am step 2 — the run script names no engine** | **Closed** — 2.60.0, a run is one start *(the user, 2026-10-31)*; 2.61.0–2.62.0, the whole-jobset review; 2.63.0, one run script over the engine record (the ledger's *Found by one start per run*, *Found by the whole-jobset review*, *Found by F2's review rounds*). | 34 + 2 (F0a's review) | — |
| **F3** | **§ 1am step 3 — the engine record and the kind/pair record whole** | **Closed** — 2.64.0–2.65.4: an engine's facts asked of its record, a kind's and a pair's of theirs; the rung builders in each kind's package; the Structure-optimization tab's two locks in the catalogue; an engine's own stop said, never refused; one describe for the transport kind, an extension's junction its first prep's (the ledger's *Found by 2.64.0…2.65.4*). B1-31 waits in F9. | 47 + 1 (F0a's review) | — |
| **F4** | **§ 1am step 4 — the file catalogue and the record doors** | **Closed** — 2.65.5–2.65.7: a `ROLE_*` constant for every catalogue row; the engines' files in one table (`runfiles.ENGINE_FILES`); a run's own records in one table; each record's path formed by its owner (the ledger's *Found by 2.65.5…2.65.7*). | 29 + 7 (F0a's review) | — |
| **F5** | **§ 1am step 5 — one codec per format** | **Closed** — 2.65.8–2.65.10: our JSON through `persist`; each of our formats one vocabulary its writer and reader share; each foreign format read in `parse/` (the ledger's *Found by 2.65.8…2.65.10*). | 25 | — |
| **F6** | **§ 1am step 6 — one home for each vocabulary, default and rule** | **Closed** — 2.65.11–2.65.13: what a person was told wrongly; a benchmark states its grid; one home only where a copy could silently go wrong, the rest left and said why (the ledger's *Found by 2.65.11…2.65.13*). | 45 + 11 (F0a's review) + 4 (from F4) + 1 (from F4) | — |
| **F7** | **§ 1am step 7 — the floors and the objects** | **Closed** — 2.65.14–2.65.17: what gave a wrong answer; floor 1 loading nothing above it; one read per answer, objects taken whole; dead code by role and history (`archive/audits/2026-11-05-2.65.17-removals.md`; the ledger's *Found by 2.65.14…2.65.17*). | 26 + 10 (F0a's review) + 6 (from F4) | — |
| **F7a** | **A person reads a run's atoms by their own index** *(the user, 2026-11-05)* | **Closed** — 2.66.0–2.67.0: the atom map stated on every answer holding atoms; the steps a run finished in the person's order; an optimization's data file written by the export alone (`archive/audits/2026-11-05-postprocessing-api.md`; K2660-1…14). |
| **F8** | **The pages — words and residue** | **Closed** — 2.67.1–2.67.3: what the pages got wrong; every word a person reads plain, no document cited in a message, frames from 1 (decided 7); one prep answer, the terminal's lines drawn on the page (the ledger's *Found by 2.67.1…2.67.3*). | 103 + 3 (F0a's review) | — |
| **F9** | **Tests through the road** (§ 1ak S4) | **2.67.4–2.67.11** (the ledger's *Found by 2.67.4…2.67.11*): the audit's retire list judged test by test; a test validates what is correct *(the user, 2026-11-06; [`process/testing.md`](?doc=process/testing.md) § 3b)*; the tools' real output; one door per test helper; the prep-side tests through the real commands; runs that never happened made on the road; the page tests fed what the server answers; the four product defects those tests found, fixed (2.67.10); the server-side test doors (2.67.11) — one way to point a test at a projects tree (`conftest.point_the_one_door_at`), one maker of a described calculation, the support helpers public and none borrowed from a test file, one maker per real run, the audit's retire list finished (K2679-9's server half, TA-REWRITE, TA-ISOLATE, TA-RETIRE's rest). **Left, in order** *(the user, 2026-11-07: "continue", on the inventory of F9-6 split three ways)*: **the page-side test doors**, done as 2.70.2–2.70.10 (O-TESTS) — the page tests under node on the one runner and fed the server's answers, no browser in the fast tier, an order held where a time was waited, one collector of a page's errors, the pages asked of the app, the page's clock for its timers (K2679-9's page half), the hand-made workspace stand-ins and their typed answers (the 2026-11-11 test audit's TA-8's store, TA-9, TA-16); **the tests owed**, each on a real run (E2E-OWED, I2-087, B4-14's walk half, B4-15, B4-16), done as 2.70.21, I2-085 with them; **the dead-code release**, done as 2.70.20 — K2677-4, K2678-5, K2679-5, K27610-6, and the three states no road makes (E2E-Q, and E2E-OWED's monitor that dies loading and viewer that throws), its removal list shown first; **F9's closing review rounds**, 2.70.22–2.70.26, their low-harm last items left (the handover's *Left, in order*). Left as they are, said why in the ledger: each module's own server fixture (`testing.md` § 8 lets a module choose its scope), the e2e marker a file repeats (dropped when the file is touched), B1-31 (the form id prefix stated by the server and the page, the browser tests catching a drift). The ledger's rows still marked F9: TA-REWRITE's page files, K2679-9's page half; B4-2, the sent-run pair, kept with its reason stated (decided 10). | 62 + 8 (F0a's review) + 4 (the test audit) | — |
| **F10** | **The documents true** | Each chapter rewritten against the code once — the untrue and stale sentences corrected, every link resolved, history inside a contract removed and each reason it carried kept as a rule; a chapter a release F0–F9 touches is brought true in that release; README R5 says why `--bundle` keeps its name (decided 8). ~300 items over 70 chapters. | 305 + 1 (F0a's review) | — |
| **F11** | **The documents findable** (§ 7) | § 7's four indexes — **done in 2.77.0**, built from the code each time they are read, the Documents tab's Indexes folder (`web/tabs.md` § 5.6; the user, 2026-11-17: "Live from the code"); `docs/README.md` the one index naming every live chapter, each row one line, and every domain overview listing all its own — **done 2026-11-17** (docs only: every row's facts checked against its chapter first, the five found nowhere else written into their chapters); the chapter mistakes the row check found — each verified against the code — corrected 2026-11-17 (docs only); and one place for each thing stated in several — 43 contradictions corrected to the code and 30 restatements made pointers, the archive's index whole — **done in 2.77.2**: F11 is closed. | 2 + § 7 | — |
| **F12** | **Proven** (§ 1am step 8) | Each road — an optimization, a vibration on SIESTA and on PySCF, a transport calculation — driven start to finish in the browser on real runs; the full batch; two review rounds over F0–F11. **Walked, run and reviewed 2026-11-18** at 2.77.2: H₂ relaxed on PySCF; water's modes on PySCF and on SIESTA; a hydrogen-chain junction built and labelled on the Molbuilder tab, relaxed on SIESTA, its transport described on the Transport tab and run — G(E_F) = 0.9999 G₀; the batch 6106 + 805, none failed; the two rounds in `archive/audits/2026-11-18-f12-review/`. The walk's one defect (Task setup's *What gets written* keeping the last folder's structure names) fixed in 2.77.3; the rounds' fixes run as *What runs next* orders them. | 8 | — |

---

## DECIDED, as it stood

*Verdict:* **Rows 1, 6, 7 and 11 done** (F0, F3, F8, F0b), and *Settled since it was asked* settled; the rows still in force — 2, 3, 4, 5, 5a, 8, 9, 10 and 12 — and *Still the user's words* stay in the plan.

## DECIDED *(the user, 2026-10-31: "go with your recommendations")*

| # | the question | decided | where it goes |
|---|---|---|---|
| 1 | A junction citation the describe door already marks refused (U1-1): Describe writes the folder, prep refuses it, and citing again into that folder is refused | the refusal stays at prep (*prep decides*); the way out is made to work | F0 |
| 2 | The half-built run-record pieces (§ 5t P2, P4; § 5s P5, decision 5): the engine-echo column, asked against ran, a vibration record's machine rows, the device contour, divergence signs, the read-back of charge and spin | after the finish line (the warm retry was removed at 2.60.0: a run is one start) | *After* |
| 3 | N11 release 2 — the page split (Transport tab the physics editor, Task setup the running) | after the finish line | *After* |
| 4 | Features the three roads do not need (11 rows: the per-trial SCF plot, the markdown editor, a notification's `ttl`, multi-node MPI, `detect()`'s reasons, a `checkpoint verify` verb, …) | dropped (the user's rule of 2026-10-18); their rows stay in the ledger as a record | — |
| 5 | N12 group 4 (9 rows, its text in no document) | dropped | — |
| 5a | A deck names the description that produced it: a hash in its PROVENANCE block, so a deck copied out of its folder can still be traced. Never built; the argument `emit_provenance` kept for it removed 2.65.17 *(the user, 2026-11-05: "i agree to your ... #3")*. Inside its folder a hand edit is already told (`prep.as_written`) | the idea kept, not built | *After* |
| 6 | The label palette's own copy of the lead names | served from the kind table | F3 |
| 7 | Frame numbers: server sentences say *frame index 0 (of 3)*, the page's rows *Frame 1 of 3* | 1-based wherever a person reads; 0-based only in files *(the user, 2026-11-05, over an October rule of theirs: "From 1 in messages")* | F8, **done 2.67.2** |
| 8 | `--bundle` on every `jobset` verb, a word README R5 retired for the calculation folder | the option keeps its name, and R5 says so (a rename would break the user's scripts) | F10 |
| 9 | Release tags — conventions § 2: every release gets a git tag; 2.7.0–2.56.1 have none | each release is tagged again, from the next one | *How a release closes* |
| 10 | The sent-run test pair (B4-2): two basic tests write a launch record by hand, because the basic suite never submits | kept, their reason stated in them (testing.md § 6) | F9 |
| 11 | How a builder hands back its findings (I2-134; `validation.md` § 4.1 R5: a finding is an `Issue`, never a Python warning) — (a) each builder returns one answer, structure, backend and findings, its Python API changed; or (b) the builders return the structure as now, and one check after the build says the findings: the structure's own (an aldehyde end, a duplex clash) read off what was built, a backend's limits (a form it cannot enforce, a terminal it ignores) a table of what each backend can do, shown on every builder panel through the one renderer | (b) *(the user, 2026-10-31: "go with b")* | F0b |
| 12 | PDB files, and maybe later `.cif`: what more molbuilder reads of a structure file *(the user, 2026-11-04: "pdb (and maybe later .cif) file support should be a separate plan - these are not urgent for now")* | a plan of its own, not urgent; no more of it in the finish-line releases | *After* (the ledger's U-PDB) |

**Still the user's words** (no work waits on them): a result written before
2.49.0 keeps its stored Methods sentence beside the free atoms' sentence;
`PLACE_NOTE_UNKNOWN`'s words; `envs/recipes.py`'s comment calling ASE a monitor
reader.

Settled since it was asked: **build S3, or state the live road** — the finish
line builds it (F1). **The citation door's *states no cell* refusal**, said to
be reached by no road — a road reaches it (a saved relaxation whose cell is
cleared, then cited), so F1 kept it (2.59.0), its words naming the way out,
the Cell page.

---

## Where the earlier sections went, and their stubs (§§ 0a, 1am, 1ai, 1ak, 2, the ledger's rows, the archived numbers)

*Verdict:* **Pointers only.** Each section they name was replaced by *The one plan* on 2026-10-31 and is in `archive/2026-10-31-plan-consolidation.md`; a reference to one of these numbers reads on here.

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

## 1ai. The Structure-optimization tab's two locks — built in 2.65.0 (F3)
The relaxation's step settings under *Relaxation type* `none` and the solvent
model without a solvent grey from the catalogue (`active_when`, `"set"`).

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

---

## § 5r — the structure API's confirmed non-findings

*Verdict:* **Still policy — moved back into the plan** (`plans/plan.md`,
*Confirmed non-findings*), where a reviewer reads it.

---

## § 7 — the document survey and the four indexes, as it stood

*Verdict:* **Steps 1–3 done** (the survey, archived in `archive/2026-10-09-plan-consolidation.md` § 5v; the four indexes built from the code in 2.77.0, `web/tabs.md` § 5.6; F11 closed in 2.77.2). Step 4 — deciding what merges, with its one risk — stays in the plan.

## 7. The document survey and the four indexes — OPEN; steps 1–3 done (§ 5v, archived; the four indexes built from the code in 2.77.0, `web/tabs.md` § 5.6), step 4 not started

*(User, 2026-09-07: "the documents are very fragmented and a lot of key
information cannot be easily found. we also should have an index that
summarizes key design aspects, such as api, module, data structure etc. we
also should have a reference list that gives the foundation of all the
constants, scientific validation foundation and paper citations summarized in
one place.")*

### 7.1 What is actually wrong — measured 2026-09-07, re-measured 2026-10-08 and 2026-10-09

**Coverage.** `web-api.md` and the live routes disagree: a path diff on
2026-10-18 gave 19 documented routes not live and 26 live ones undocumented
(the checkpoint, files, modify and workspace-storage families among them),
against about 92 unique `/api` routes (the blueprints', the app's own in
`app.py`, `rate_limit.py`'s). Re-derived with the Routes index (2.77.0,
2026-11-17): every live route is named by a chapter.
And coverage was never truthfulness:
the 2026-09-29 check read every live document against the code and found
statements the code contradicts — § 5v holds that list, and the survey starts
from it rather than repeating it.

**Findability.** **83** live documents, **83,077 lines** (2026-10-28; 79 and 73,853 on 2026-10-18), nine directories
(`docs/` and eight below it; `archive/` and `plans/` excluded). The
information is there and a person cannot reach it.

**The existing index has the problem it is meant to solve.** `README.md`
§ Index is one row per DOCUMENT, and its longest row runs about 828 words
(2026-10-28). To find out which document owns a fact you read
essays about documents. It is a good map of the tree and a poor answer to
"where is X".

A document can contradict itself, and nothing systematic finds it — which is
the survey's real justification.

### 7.2 The four indexes — organized by THING, not by document — DONE in 2.77.0

Each is a lookup table, each row pointing at the documents that own the
detail. **None of them may become a second home**: a row says where a fact
lives, never what the fact is. (`README.md` R-W1, and the reason `docs/`
survived its 2026-06-02 over-compression.) **All four are built from the code
each time one is read** *(the user, 2026-11-17: "Live from the code", over
two written by hand as first sketched here)* — the Documents tab's Indexes
folder, `web/tabs.md` § 5.6, which holds what each row says and how a chapter
is found to name a thing: **Routes** from the running server, **Modules**
from the package's files and `architecture.md` § 3's rule, **Files** from the
run-file catalogue in its one rendering (`catalogue_tables`, the rows
`execution/project-layout.md` § 5 shows too), **References** from
`constants.py` and the bibliography's reader.

### 7.3 The reference index, which is the one with real content in it

Constants have one home already (`molbuilder/constants.py`), and the dialects
a quantity is written in have theirs (`molbuilder/units.py`); the rule for both
is `architecture.md` § 3.  Review holds that rule; no test does. What has no
home is the **reasoning**, and it is
genuinely scientific. The worked example: `constants.py` defines
`HARTREE_BOHR_EV_ANGSTROM_ASE = 51.42208619`, the ASE/NIST value, and says why
the name carries the convention — `HARTREE_EV / BOHR_ANGSTROM` gives
51.422067476, 0.36 ppm apart, and a force and the threshold line drawn over it
must use the SAME one. The emitter imports it (`trajectory_log/emitter.py`)
so emitted forces line up with what a person reads in ASE, VASP and QE
logs. That is a real convention choice with a real justification, and it lives
in the comment above the constant — which the References index now shows
beside its value (2.77.0, `web/tabs.md` § 5.6), the comment staying its one
home.

Citations are split the same way: `science/references.bib` holds **94**
entries (2026-11-16), about a dozen live documents cite it, and the rest cite
literature in prose, if at all.

So the reference index carries three things per row: the value, where it is
defined, and the sentence saying why that value and not the neighbouring one.

### 7.4 Order, and the one risk

1. **Survey first, and record findings without fixing them** — **done** (§ 5v,
   in [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md)):
   every live doc read against the code, findings in one list, an entry
   leaving it when its document is corrected. The route sweep, kept there
   too, is the same kind of finding for `web-api.md`.
2. **The four indexes** — **done**, all built from the code (§ 7.2). Measured
   with them, 2026-11-17: every live route is named by a chapter.
3. *(folded into 2: no index is written by hand.)*
4. **Only then** decide what merges. Fragmentation may be the symptom; the
   fix might be four indexes and no merges at all.

**The risk:** a survey that reorganizes without a per-item carry-over check
loses items. The rule for step 4: **nothing moves until the thing it says is
written where it is going** — the archive's substance-first rule.

---


---

---

## § 9.1 — the pseudopotentials explicitly provided, and named for their label

*Verdict:* **Done**, archived 2026-10-08: refusals when the calculation is configured (`validation/siesta.py`, `pseudos.check_coverage`; `science/pseudopotentials.md` § C1a). § 9.2 stays in the plan.

## 9. ATOM as a pseudopotential validator — § 9.1 DONE, § 9.2 PENDING, not started

*(User, 2026-09-19.)* **§ 9.1 — done, archived 2026-10-08**: the user's two
rules — the pseudopotentials are explicitly provided, and the file for element
`E` is named `E.psml` — are refusals when the calculation is configured
(`validation/siesta.py::_check_siesta_pseudo_coverage`; `pseudos.check_coverage`;
contract `science/pseudopotentials.md` § C1a).

