# The plan — one file

**Role:** plan — **the only one** *(user, 2026-09-01: "We don't need ten plan
files scattered")*. Every open item lives here, said once; a document that
finds work records the evidence and sends the item here (R3,
[`README.md`](?doc=README.md)). Finished work is archived, every archived file
listed in [`archive/README.md`](?doc=archive/README.md); the plan as it stood
at 2.77.9 is [`archive/2026-11-19-plan-consolidation.md`](?doc=archive/2026-11-19-plan-consolidation.md).
**Domain:** all · **Last consolidated:** 2026-11-19.

---

## ▶ Where we are *(2026-11-19)*

**2.77.12.** F0–F11 and F7a are closed (2.56.2–2.77.2). 2.77.10 brought Plotly's toolbar back on every chart, each chart's Export in a strip above it, and 2.77.12 a vibration mode's animation into a column beside the spectrum and its table, held in view *(the user, 2026-11-19)*. F12, *Proven*, walked
its three roads in the browser on real runs, ran the full batch (6106 fast
and 805 e2e tests, none failed, at 2.77.2) and its two review rounds
([`archive/audits/2026-11-18-f12-review/`](?doc=archive/audits/2026-11-18-f12-review/TRIAGE-round1.md));
its fix releases are running.

**What runs next — F12's fix releases**, harm first, by module *(the user,
2026-11-18)*. Each fixes its module's round-1 defects (`TRIAGE-round1.md`)
through the one-home redesign round 2 found for it (`TRIAGE-round2.md`):

1. the builders build what was asked — done, 2.77.4;
2. the deck writes the value set — done, 2.77.5 (D7 and D11 left: a warning,
   a cosmetic); 2b. every scientific refusal at prep a warning — done, 2.77.6;
3. the person's files safe — done, 2.77.7;
4. a failure said in words, never a traceback — done, 2.77.8;
5. the pages show the folder and run they say — done, 2.77.9;
6. MolView — done, 2.77.11;
7. **the viewers** — next — with the spectrum's pointer readout given a place clear of
   Plotly's toolbar, which covers it on a tall window *(the user, 2026-11-19)*;
8. the run script and the machine.

Then the rest of round 2 module by module, then the 85 chapter sentences
(`TRIAGE-round1.md` § I; five need code: A-U6, B-U3, C-U3, H-U4, H-U5). Two
answers for them *(the user, 2026-11-18)*: a SIESTA vibration's result keeps
the structure's own notes, as PySCF's does; a report-cards copy that does
not read stops only the monitor of a run that sends reports.

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
  on Sol.
- **When needed**: a real fully relativistic pseudopotential — take one from
  PseudoDojo's `nc-fr` set into `tests/fixtures/psml/` and read
  `relativity="dirac"` as spin-orbit on it.

**Still the user's words** (no work waits on them): a result written before
2.49.0 keeps its stored Methods sentence beside the free atoms' sentence;
`PLACE_NOTE_UNKNOWN`'s words; `envs/recipes.py`'s comment calling ASE a
monitor reader.

**Do not "fix"** the structure API's confirmed non-findings (re-read
2026-10-08): they are listed in the archived plan, *Confirmed non-findings*.
