# The plan — one file

**Role:** plan — **the only one** *(user, 2026-09-01: "We don't need ten plan
files scattered")*. Every open item lives here, said once; a document that
finds work records the evidence and sends the item here (R3,
[`README.md`](?doc=README.md)). Finished work is archived, every archived file
listed in [`archive/README.md`](?doc=archive/README.md); the plan as it stood
at 2.77.9 is [`archive/2026-11-19-plan-consolidation.md`](?doc=archive/2026-11-19-plan-consolidation.md).
**Domain:** all · **Last consolidated:** 2026-11-19.

---

## ▶ Where we are *(2026-11-21)*

**2.77.19.** F0–F11 and F7a are closed (2.56.2–2.77.2). 2.77.10 brought Plotly's toolbar back on every chart, each chart's Export in a strip above it, and 2.77.12 a vibration mode's animation into a column beside the spectrum and its table, held in view *(the user, 2026-11-19)*. F12, *Proven*, walked
its three roads in the browser on real runs, ran the full batch (6106 fast
and 805 e2e tests, none failed, at 2.77.2) and its two review rounds
([`archive/audits/2026-11-18-f12-review/`](?doc=archive/audits/2026-11-18-f12-review/TRIAGE-round1.md));
its fix releases are running.

**What runs next — five releases to the finish line** *(the user,
2026-11-19: "a consolidated plan to get things finished, not to drag things
along"; five releases, each with its two review rounds)*. F12's fix releases
1–6 are done (2.77.4–2.77.11: the builders, the deck's value set, every
scientific refusal a warning, the person's files, a failure in words, the
pages' folder and run, MolView); 7, the viewers — the animation 2.77.13,
the spectrum chart 2.77.14, the spectra page 2.77.15 (R1); 8, the run script
and the machine 2.77.16 (R2); 9, the jobs, run files and records 2.77.17 (R3a);
10, the run script, envs and server process 2.77.18 (R3b); 11, the job
bundles gathered from their members' own word 2.77.19 (R3b2).

1. **R1 — the viewers' last part, the spectra page** — done, 2.77.15.
2. **R2 — the run script and the machine** — done, 2.77.16 (§ H, the user's
   two answers of 2026-11-18, and "Two zones" for your own lines, 2026-11-19;
   [`fix8-review.md`](?doc=archive/audits/2026-11-18-f12-review/fix8-review.md)).
   Round 2's three safety items had shipped in 2.77.7.
3. **R3 — one home per fact, the server side**, in six parts: R3a the jobs,
   run files and records — done, 2.77.17
   ([`fix9-review.md`](?doc=archive/audits/2026-11-18-f12-review/fix9-review.md));
   R3b the run script, envs and server process — done, 2.77.18
   ([`fix10-review.md`](?doc=archive/audits/2026-11-18-f12-review/fix10-review.md));
   R3b2 the job bundles gathered from their members' own word (R2-3 F1) —
   done, 2.77.19
   ([`fix11-review.md`](?doc=archive/audits/2026-11-18-f12-review/fix11-review.md));
   R3c
   templates and engines;
   R3d transport and spectra, with the SIESTA vibration's data files on the
   shared writers; R3e parse and builders; R3f the web server, with the side
   rail's height measured on the real pages and brought to the user first
   *(the user, 2026-11-20)*. Round 2's items for template and engines
   (with `r2-3` F3, the pipeline log's order beside `template.SAID`, its
   owner), transport, spectra, parse, structure and builders, and the web
   server
   (`TRIAGE-round2.md` § Biggest one-home items, and each `r2-*.md`), with
   the chapter sentences that need code (H-U4, H-U5); and one R2 left
   here: a structure title pasted with a line separator (U+2028) cannot be
   saved — the title is one line, at its door. Left to its parts by R3a's
   review (`fix9-review.md`): **R3c** — `pyscf/script_input` and `script_emit` find block fences by
   exact text (`deck_record.MARKER_RE` is the grammar), the generated
   PySCF scripts write the standard atmosphere and the atomic mass unit as
   numbers (`constants.ATM_PA`, `AMU_KG`), `template._refuse` copies the
   record refusal kit (`persist.RecordRefusals`); **R3e** — `builders`
   resolves `module:NAME` itself.
4. **R4 — one home per fact, the pages and the tests**: round 2's items for
   the pages, MolView, the viewers and the test harness — the Results
   tab's file-menu pick copied in ten test files among them, its door now
   `support.sidebar.result_picked` — the warning that a file renamed to a
   name starting with a dot leaves the sidebar (round 2's file-name item,
   the rest shipped in 2.77.7), and the deletions the readers proposed,
   each shown to the user before it goes. The jobs' and run files' test
   items: `r2-1-jobset.md` T2–T7, `r2-2-runfiles-records.md` T1–T3, T6, T7.
5. **R5 — the documents and the proof**: the 80 chapter sentences that need
   no code (`TRIAGE-round1.md` § I, one docs pass, no version) with
   `r2-1-jobset.md` F2's sentence, F3 and F14, then the
   finish line's proof: the full batch, and the three roads walked again in
   the browser on real runs.

**Waiting on the user:** nothing.

## Rules in force

- **Nothing stops a calculation the person set** *(the user, 2026-11-18)*: a
  scientific finding is said and the run goes as set; an engine's own stop is
  said as information and the deck written as asked *(2026-11-02)*. What may
  still be an error is [`science/overview.md`](?doc=science/overview.md) § 4's.
- **Every warning in plain words** *(2026-11-18)*: what happens to the
  calculation, then what to do, in the pages' words.
- **A fix is framework-level** *(2026-10-31, again 2026-11-19)*: one door, one
  API, one data shape per fact; a defect seen at several places is fixed in
  the door they share; a needed mechanism is extended where it lives, never
  patched beside it.
- **Off the plan** *(2026-10-18)*: every feature the three roads do not need;
  any check, warning or format the contracts do not state. The AuBDTAu
  workflow is the user's, on Sol.
- **The test suite never reaches the person's machine**
  ([`process/testing.md`](?doc=process/testing.md) § 2b).

## Open beside the finish line — low harm, waiting *(read against the code 2026-11-18)*

- The vibration's data files still copy a record's values onto their grid by
  their own `key_of` loops and encode their attributes by hand, where the
  transport file and a chart's use `ncdata.copy_stated` / `ncdata.json_attrs`;
  the status door says an unreadable copy note in the copy's own field, where
  the transport record says `unreadable`; the record's and the data file's
  builders are long functions.
- A held side rail's height cap is the window less a margin (`form-components.css` `.page-aside`, Task setup and Transport), while the pane under the header is shorter: a rail taller than the pane holds its top, its end reached with the page's end (measured 2026-11-19, the spectra layout's review). A cap true to the pane needs the shell's content pane as a size container — a change to every page's shell, the user's to decide.
- F9's last items: four page stand-ins without a named need (the Task setup
  save bar's refusal, the export door's answers, the save door's page, the
  checkpoint Cancel's signal) and the installer's repair audit.
- The page doors' rest — O-STATE's one save door, O-WORDS, O-KEYS with X-UI
  and X-PLOT — internal tidying no page shows; the test framework's TF3–TF10
  and the page tests on jsdom.
- § 7 step 4 of the document survey: decide what merges — nothing moves until
  what it says is written where it goes (the survey and indexes are done).

---

## The finish line *(the user, 2026-10-30)*

molbuilder is finished when (1) every engine and kind fact is asked of its
record, every name and record goes through its owner, one reader and one
writer per format, one home for each vocabulary, default and rule (§ 1am
steps 2–8, F2–F12); (2) the three roads — an optimization, a vibration (SIESTA
and PySCF), a transport calculation — are proven start to finish in the
browser on real runs; (3) the documents are true and findable. New science,
and what only the user's Sol runs can show, come after it.

## How a release closes *(the user, 2026-09-27, 2026-10-31)*

1. Its owning contracts read end to end and its code whole; the design
   written into the contract first (a design change waits for the user's
   yes); the code; a test that drives the road, seen failing on the old code.
2. The fast batch and the targeted tests; the package pyflakes-clean.
3. **Two review rounds**, every finding re-read before it is acted on:
   correctness, and holistic — one door and one API per fact, one data shape,
   nothing duplicated, redundant or handcrafted, in the code and its tests.
4. The reviews' fixes.
5. One e2e run, last: the e2e files whose roads the code touches, named with
   why; never while a review of the same code is open.
6. This plan brought current in the release's commit, nothing archived
   unchecked; the release tagged `v<version>` and the tag pushed.

---

## After the finish line

The design notes named here are in
[`archive/2026-11-19-plan-consolidation.md`](?doc=archive/2026-11-19-plan-consolidation.md),
each under its section; each new piece of science comes to the user as a
design with its science first.

- **New science**: § 1ag corrections to a junction's electronic structure (a
  framework before any one correction); § 1ah which modes get Raman (after
  the run, and by symmetry); M10 from a mode to the current; § 5e a person's
  own engine text as an input; § 9.2 ATOM as an optional extra layer; S13
  transport's convergence sweep; K22 per-atom starting spins; LEAD1; the V1
  rows; § 1ac's D-h, B and the measured intensity's factors — the
  finish-line ledger's *After — science* (28 rows,
  [`archive/2026-10-31-finish-line-review.md`](?doc=archive/2026-10-31-finish-line-review.md));
  F0c-R1-S, a fixed-spin optical frame's ε₂ against one Fermi level.
- **Designed, not built**: § 5t's and § 5s's run-record pieces (the ledger's
  *Waits — run record*); N11 release 2, the page split; a deck naming the
  description that produced it (a hash in its PROVENANCE block); PDB and
  later `.cif` reading, a plan of its own (U-PDB).
- **Only the user's Sol runs can show** — E13, SCHED-HDR, BENCH-SHELF,
  SCHED-R, § 1ac's proofs 3a and 3b, the PySCF environment's `netcdf4`, the
  gcc 15 pin (the ledger's *After — Sol*, 13 rows), and the monitor's cgroup
  v1 readings (`tests/field/test_the_monitor_reads_its_job.py`), in one pass
  on Sol; and **each rank's cores** (`fix10-review.md`, round 2 #14): under
  `mpirun` with no binding flags — a GPU job under a scheduler, and every CPU
  launch — OpenMPI binds a rank to one core, so a rank's `-c`/`-omp` threads
  may share it; a Sol GPU run's session log says each rank's cores
  (`molbuilder[rank i/n] … cpus=`), and a hybrid CPU build has no such line
  yet. The binding is decided on that evidence (`running-a-job.md` § 3.3).
- **When needed**: a real fully relativistic pseudopotential — take one from
  PseudoDojo's `nc-fr` set into `tests/fixtures/psml/` and read
  `relativity="dirac"` as spin-orbit on it.

**Still the user's words** (no work waits on them): a result written before
2.49.0 keeps its stored Methods sentence beside the free atoms' sentence;
`PLACE_NOTE_UNKNOWN`'s words; `envs/recipes.py`'s comment calling ASE a
monitor reader.

**Do not "fix"** the structure API's confirmed non-findings (re-read
2026-10-08): they are listed in the archived plan, *Confirmed non-findings*.
