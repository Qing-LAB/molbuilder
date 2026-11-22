# F12 fix 13 (a run keeps its files, 2.77.21, R3b3b) — the review

**Verdict: closed with 2.77.21.** The person's answers to the framework, API and deployment
review of 2.77.19 (2026-11-21): *a run keeps its files* — a SIESTA vibration's optical calls
took their deck, run script and bundles from the stage folder while the run went, so a prep
while it was queued gave its later calls another release's files (deployment D1); *changed
shipped code is said at prep, prep goes on* (G3); `-I` for the monitor and the finish (G4);
the re-prep sweep of every bundle. And the person's two rules this release: *"make sure
existing old files can be parsed correctly in the result"* — every run on disk read through
the Results doors by 2.77.20 and 2.77.21 alike; and *"synchronized review and revision of
parser of result … document is key to make sure api and framework/diagrame is synced"* — the
result parser had its own reader in round 1, and its doors in round 2's correctness brief.
Round 1: four readers (correctness, result parser, one-home and discipline, notes and tests);
round 2: two new readers; then two new, independent readers of the whole text; then a reader of
the fixes.

**What the review taught.** Two readers in round 1 found the transition the release itself
opened: a run launched by an older release, still queued, whose stage is prepared again under
this one, would have run this release's record steps against a copy it never had — every
optical call failing after its SIESTA, every ε₂ lost. A run without its own copy now fills its
calls from the stage, as it always did. And a fix of round 1 — a local renamed for its meaning —
collided with a name bound further down the same function, so every optical call was opened
empty; no fast test runs the planner (it needs a real `.FC`), and only the real-run e2e caught
it. Round 2 then pinned that planner on a real run.

**Old runs on disk.** The 210 folders of the 20 calculations in `projects/` (folder answer,
views, the file each opens, parsed), the transport calculations' reports, and 68 loose files a
parser claims: the same answers from 2.77.20 and 2.77.21; nothing in `projects/` written (a
before/after listing). Two refusals there predate this release and are the same on both:
six calculations with a `job-set.json` from before 2.60.0 (`continue_retries`;
`AuBDTAu/transport/lowbias2` the person's) and two structure pairs with a v6 sidecar
(`hemeC`, `hemeC-dithiol`), whose refusal's way out cannot be followed — both moved to R3e as
migrations *(the user, 2026-11-21: "do the run dir and sidecar migration in R3e")*.

## Round 1 — four readers, judged against the chapters

| # | finding (readers) | verdict |
|---|---|---|
| 1 | a run opened before 2.77.21, its stage prepared again under 2.77.21 while it waits: its later optical calls run this release's steps, which read a copy it does not hold — every call fails after a clean SIESTA (one-home, result parser) | fixed — `_optical_source`: the run's copy where it holds one, else the stage's `optical/`; the chapter says so |
| 2 | the changed-code note read the copies of points the newest run took over (another release's), and the newest run even when a refused send opened it (correctness, one-home) | fixed — the newest LAUNCHED run (`newest_launched`), a point it took over aside (`taken_from`) |
| 3 | `-I` missing from two chapters' command lines; § 3.2's row without the flags' door (correctness, notes-tests, one-home) | fixed |
| 4 | the operations diagram drew the copy from operation 2, which writes nothing (all but one) | fixed — from the opener, at prep and at each launch |
| 5 | a sentence about runs opened before 2.77.21; the pseudopotentials named among the stage's `optical/` files; the § 2.4 tree without the copy (correctness, one-home) | fixed |
| 6 | notes: `bring_files`, `beside_the_deck`, `_points_walk`, `Run.prepared`, parse.md § 5d.4, the `coordinates.fdf` row (all) | fixed |
| 7 | "held" names the held atoms in this module (one-home) | fixed — `_optical_source`, `copies`, `container` |
| 8 | the main rule pinned only at prep; the run script's own `-I` untested; the deck test's bundles taken from the code's output; a docstring untrue of 2.77.20 (notes-tests) | fixed — a real-run pin of the data file's deck, a fast test of the run script's finish check past a decoy `numpy`, both bundles named |

## Round 2 — two new readers

| # | finding (reader) | verdict |
|---|---|---|
| 1 | round 1's rename put the copy under `source`, the name the seed's run reference takes further down: every optical call opened empty (both; caught first by the real-run e2e) | fixed — `files_from`; each name bound once |
| 2 | the planner's filling path pinned by no test (one-home-notes-2) | fixed — a real-run pin: the planned calls taken out of a copy of the shared run, the stage written again as a later prep writes it, the record step run in place: every call holds the run's copy |
| 3 | the e2e input was no input a prep of a launched stage writes (one-home-notes-2) | fixed — another release's stamp, the same calculation |
| 4 | two sentences absolute where an old run falls back (one-home-notes-2) | fixed |
| 5 | in a run 2.77.21 opened from a stage an earlier release prepared, the calls are planned by that release's steps from the stage, while the data file cites the copy — the same calculation, so only lines that compute nothing differ (correctness-2) | the chapter says which release's steps decide |
| 6 | the deck and its files could fall back to the calculation's root (correctness-2) | fixed — two copy steps: the deck and its files from the copy alone, the pseudopotentials its else the root's |

## The two independent full-text readers

Neither found a defect that breaks a run; both read a new run and an old one through the
Results doors and found them read as the chapters say. Their notes:

| # | finding (reader) | verdict |
|---|---|---|
| 1 | prep's note silent for a run opened before 2.77.21 whose prepared points were taken over — no container copy to compare (fulltext-b) | fixed — a bundle is compared with any copy the newest launched run holds of its own, but a point it took over; the first answers, and its run script names the release |
| 2 | "a run opened before 2.77.21" is the wrong boundary in the diagram, project-layout § 1.6.2 and job-request § 5 — a run 2.77.21 opened from a stage an earlier release prepared plans its calls with that release's steps (both) | fixed — the criterion is the run's own steps' release; each sentence links vibration § 5.11 § 2 |
| 3 | the catalogue's writer column leaves out launch, which copies into a planned point it opens again (fulltext-b) | fixed; the manifest regenerated |
| 4 | an index that does not read stops prep's note with an error (fulltext-a) | dropped — only a hand-damaged index reaches it |

## The reader of the fixes

No finding. It ran the planner on a prepared run both ways (the copy; a run opened before
2.77.21) with the stage and the calculation's `pseudos/` rewritten after the run opened, and
prep's note through a real `jobset prep` of six launched layouts — the taken-over case the
full-text reader found among them — each as the chapter says.

## Evidence

- The whole fast suite: 6296 passed (round 1), 6297 (round 2), and again after the version
  (below). Real-run e2e: 63 passed before round 1; after round 2, the vibration and
  stage-that-has-run files 31 of 31, and the stage-that-has-run file 7 of 7 after the
  full-text readers' fix to the note — the failure between them, round 1's rename, found by
  them.
- Each new test seen failing on 2.77.20 at its assertion: the container's copy, the stray
  bundle, a kind step and the run script's finish check past a decoy `numpy`, the two real-run
  pins (the call's deck, the data file's deck); the prep-notice row cannot pass there (no such
  sentence).
