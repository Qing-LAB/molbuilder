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

## ▶ HANDOVER — where we are, and where to start *(2026-11-10)*

**Where we are: 2.70.28.** F0–F8 and F7a are closed; F9, the tests through
the road, has the low-harm last items of its closing review left (*Left, in
order*, below); F10 is under way — wave 1 shipped as 2.70.28, waves 2 and 3
being applied (*What runs next*). Since 2.67.11 three bodies of work the user put first have run;
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

**Last checked** (2.70.27, F10's first full e2e pass): every e2e test, 782
of 783 passed; the one, a notebook test whose port another parallel test's
server had taken, passed alone and is fixed in 2.70.28 (test servers take
even ports below the system's own range).

**What runs next**: the stretch above (*Left, in order*) -- F9's last
step, its closing review rounds (the page-side test doors, the dead-code
release and the tests owed done: 2.70.2–2.70.10, 2.70.20, 2.70.21) and
the UI review's two left items (2.70.27) done; **F10 under way** (the user,
2026-11-15: "start F10") -- its first full e2e pass done on 2.70.27; the
chapters read whole against the code: wave 1 (2.70.28) the top-level
chapters, ops and execution; waves 2 and 3 (2.70.29) the engines, science,
model, web and process chapters and this plan -- every live chapter read
once; the product defects those readings found in the pages measured and
fixed (2.70.30, 2.70.31). Next F11, F12. **Waiting on the user**: nothing now; TF5
brings one decision when it is reached (a test's own projects tree by
default). F12's walk of the three roads in the browser starts only on the
user's word, said to them first.

**The test suite's guards** keep every test off the person's machine
([`process/testing.md`](?doc=process/testing.md) § 2b): test code loads only
under pytest, each test process runs in a sandbox of its own, and one write
guard covers the whole run.

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
| **F11** | **The documents findable** (§ 7) | § 7's four indexes — **done in 2.77.0**, built from the code each time they are read, the Documents tab's Indexes folder (`web/tabs.md` § 5.6; the user, 2026-11-17: "Live from the code"); `docs/README.md` the one index naming every live chapter (it does, 2026-11-17) — its rows shortened to one line each, and each domain overview listing its own; one place for each thing now stated in several. | 2 + § 7 | — |
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
  gcc 15 pin — the ledger's *After — Sol* (13 rows); and the monitor's cgroup
  v1 readings, `tests/field/test_the_monitor_reads_its_job.py` (K2675-5), run in
  one whole pass on Sol *(the user, 2026-11-06: "i'll do sol run holistically next
  time")*. Measured on the user's
  AuBDTAu Raman run of 2026-10-28: about two hours an optical call, nearly all
  in SIESTA's band-pair loop.
- **When needed** — a real fully relativistic pseudopotential *(the user,
  2026-11-06: "leave the relativistic file for now, just document it as next
  plan when needed")*: every file on this machine is scalar relativistic, so
  reading `relativity="dirac"` as spin-orbit is checked on a real file with that
  one word changed. When a calculation needs spin-orbit, take a file from
  PseudoDojo's `nc-fr` set into `tests/fixtures/psml/` and read it there; that
  also confirms the word.

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

## 0. Rule R3, restated

R3 ([`docs/README.md`](?doc=README.md)) *(user, 2026-09-01: "the old roadmap
and audit should be archived")*:

> **`plans/plan.md` is the one plan. Every open item lives there, and nowhere
> else. A document that finds work records the evidence and sends the item
> here.**

The old roadmap and the two audit reports are archived under
`archive/2026-09-01-*`.

**R3a — a plan section is kept current while its work runs** *(user,
2026-09-16)*:

> **A programme with an implementation section updates that section when each
> step lands, and records a problem found mid-step before working around it.**

Not afterwards, and not in a commit message alone. A step that changed shape, a
step that turned out to depend on something unlisted, a defect found while doing
it: each goes in, with what was measured. **A plan written once describes an
intention; a plan kept current describes the work.**

**The one plan is the list.** A design section keeps the *why* and sends its
rows there; a row done, withdrawn or measured untrue is marked with the
release that closed it (*How a release closes*, step 6).

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
transmission re-runs, never the device SCF. **It is not `TS.dQ fermi`** *(2026-11-08)*: that moves the
device's reference energy inside its own SCF — the manual's correction for
*"an incorrect reference energy level"* — where a δH changes the converged
Hamiltonian afterwards — the comparison is
[`engines/transport.md`](?doc=engines/transport.md) § 2a.12, *The device's
charge, its reference energy, and `TS.dQ`*. A δH's level targets are taken
against the E_F the device converged with, corrected when `TS.dQ fermi` ran.

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

## 5e. Engine additions — a person's own engine text, as an INPUT

*(Your ruling, 2026-09-05: option 2 — a distinct input to the writer, not a
catalogue extension; **engine-specific**; and **the person is told the
consequence**. Contract-first: this section is the design. **Not started.**
No code has moved.)*

`render_deck` concatenates the zone after the walk; `check_deck` refuses a
deck holding more than one pair of USER-CUSTOM markers, so a stray marker
drops no text silently; and transport decks are rendered by `render_deck`
like every other. `engines/template.md` § 9.2 says what this ruling holds: a
person's own text is no catalogue item.

### The need

A person wants to run a SPECIFIC task on a structure that is already
optimised, using engine content molbuilder does not model: a Lua setup driving
SIESTA, a `%block` for a feature with no form field, a PySCF call.

Today the only place for that is the USER-CUSTOM zone of a relaxation deck,
which is the wrong shape twice over — it is the wrong kind of task, and (below)
the zone cannot carry the content anyway.

### Why the zone cannot carry it — measured, not argued

`render_deck` assembles the deck like this (`script_emit.render_deck`):

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
   (`siesta/layout.py`) splits the whole file and knows nothing of the
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

**What it buys instead** is that the new need lands in its own layer:

- nothing in the relaxation path changes to accommodate it;
- the mechanism is defined by what it IS (an input, engine-placed, switchable,
  unvalidated) rather than by which existing function it borrows;
- when a task kind does need it, that task brings its own writer and this
  mechanism plugs into it — no structural change to make room.

### THIS IS NOT A STAGED RUN, and must not be fitted into one

*(Your correction, 2026-09-05.)*

**Stages exist for one reason: a calculation that needs several steps to fit
the computational resources and constraints** — coarse before tight, a ladder
that accommodates a machine. That is a different problem from this one.

A customised block is for **a specific task, on a structure that is already
optimised**. It is the mechanism molbuilder EXPOSES for a future kind of task
that needs it — not an extension of the relaxation path.

**An addition is not a staged run**: per-stage override is a stage concept,
and an addition has no stages to override across.

**What holds, because it does not depend on staging:**

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

### Before any code

This section is the contract. The measurements it rests on
(`render_deck`'s concatenation, the 12/520/980-vs-1222 positions, the
reproduced duplicate-keyword refusal, `script_emit.Parameter.writes`) are re-checkable, and
should be re-checked rather than trusted if this is picked up later.

## 5r. The structure API — the confirmed non-findings *(its order, 5r.1, is F7's ledger section now)*

### 5r.2 Do NOT "fix" these — confirmed non-findings

Recorded so nobody fixes them by analogy. Re-read against the code 2026-10-08.

- **`frozen_atoms`'s shape is benign.** A data descriptor sends every read and
  write through `regions[FROZEN_LABEL]`, so the two cannot disagree in any
  order; `replace()` handles it by not re-passing it. **Not the defect `pbc`
  was** — the init-field spelling exists so `Structure(…, frozen_atoms=[…])`
  reaches the one place that spells the reserved label.
- **The companion-lookup merge stays withdrawn.** What forbids it is a test
  pinning a deliberately more permissive guard, not `parse.md` § 5.3; a shared
  helper with the guard as a parameter would keep both.
- **The multi-frame XYZ reader is ours, not ase's**: it tolerates a torn
  final frame (`parse/engines/xyz.read_frames`, which the geomeTRIC reader
  `parse/engines/pyscf.py::_parse_pyscf_xyz` uses), which every live
  geomeTRIC run has and ase refuses. Reading it with ase would break live-run
  viewing.
- **Registry overlap: none** — 14 parsers against a 29-file synthetic run
  directory, no file claimed twice.
- **Bare atom-index arithmetic in `parse/`: none that carries data**, and
  the transport half is done: since 2.32.0 `transport/transiesta.py`
  numbers atoms through `engine_atom_index.siesta_numbering`.
- **The CLI's seam verdict** was raised and dropped on the user's ruling
  (2026-09-22: *"people working with CLI would know what they're doing… leave
  that out"*). X3 is the browser half only.


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

## 9. ATOM as a pseudopotential validator — § 9.1 DONE, § 9.2 PENDING, not started

*(User, 2026-09-19.)* **§ 9.1 — done, archived 2026-10-08**: the user's two
rules — the pseudopotentials are explicitly provided, and the file for element
`E` is named `E.psml` — are refusals when the calculation is configured
(`validation/siesta.py::_check_siesta_pseudo_coverage`; `pseudos.check_coverage`;
contract `science/pseudopotentials.md` § C1a).

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
| `x3dna*/` in `.gitignore` | `atom*/` likewise, beside `x3dna*/` in `.gitignore`: the ignore has to exist *before* the folder does, or a redistribution-prohibited package lands in `git status` |
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
out of scope today (it points back here), and the ATOM
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
(nothing is installed into an env without asking),
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

