# F12 fix 12 (the bundle framework, 2.77.20, R3b3a) — the review

**Verdict: closed with 2.77.20.** From the framework, API and deployment review of 2.77.19
*(the user, 2026-11-21: "a review at framework and api level and deployment level")*: the three
bundles beside a job had no single shape (six places each knew a part of a bundle, and a
travelling row with no builder was skipped in silence), two naming rules, undeclared verbs, the
chat card's layout declared by the wrong module, three modules that tried an installed
molbuilder first, and checks weaker than the rules (the monitor's load test blocked numpy
alone; the import test saw only travelling names). Now each bundle is one row of
`runwrap.BUNDLES` and its doors (`bundle`, `members`, `build`, `verbs`) answer for all three;
one name rule; a kind's step its finish does not run is refused at prep; the checks hold each
bundle to the packages `configuration.md` § 2.3 states for it. Round 1: three readers
(correctness, one-home and discipline, notes and tests); round 2: two new readers; then two new,
independent full-text readers, kept from the earlier findings; then a reader of the fixes.

**What the review taught.** A rule stated in a new table is only as strong as the test that
reads its expectation from the chapter, not from the table: round 2 caught the load test
holding each bundle to its own row (so widening the monitor's row would have passed) and the
card test no longer run on the standard library alone. And a word used for three things
("entry": the zip's `__main__`, the module it imports, a function) misled a reader within one
section; the chapters now keep "entry" for the zip's `__main__` and say "starting module" for
the module it imports.

## Round 1 — three readers, judged against the chapters

| # | finding (reader) | verdict |
|---|---|---|
| 1 | the monitor's name beside a job spelled in its entry as well as `monitor.TRAVELS_AS` (one-home) | fixed — the entry imports the name the module states |
| 2 | the "a job's finish is a finish row" refusal written twice (one-home) | fixed — once, in `render_run_wrapper` |
| 3 | verb names typed by hand beside their constants: the vibration's usage, the run script's comment, the monitor's deck usage (one-home) | fixed — each from its constant |
| 4 | three `#:` notes pushed onto the new declarations (all three) | fixed — each on its own |
| 5 | `runwrap.bundle` missing from the door rows (one-home) | fixed |
| 6 | the plan's row fields; the vibration's `VERBS` note (one-home) | fixed |
| 7 | a finish row with no packages would stop molbuilder importing (correctness) | fixed — one phrase, "the standard library, …", for the entry and the run script |
| 8 | the standard-library refusal's way out (`TRAVELS_AS`) refused in turn for a part or a finish's module (correctness) | fixed — it names both ways out |
| 9 | `loads` accepted as a kind's step (correctness) | fixed — `verbs` is the module's own; `loads` is the wrapper's question |
| 10 | the load test took its expectation from the row it tests (notes-tests) | fixed — `CHAPTER_PACKAGES`, § 2.3's words, and the row asserted equal |
| 11 | four chapter sentences naming the renamed rule or retired names (notes-tests) | fixed |
| 12 | the step refusal and the finish entry untested (notes-tests) | fixed — a prep test (`plann`, `loads`) and a `loads` test, with numpy refused; `bundles_for`'s `.pyz` check has no real input and stays untested |
| 13 | a third "a, b and c" joiner; `Bundle.starts`; `_zip_bundle`'s two `members` (one-home, below triage) | names fixed (`starting_modules`, `entries`); the joiner kept — no shared one at runwrap's floor |

## Round 2 — two new readers

| # | finding (reader) | verdict |
|---|---|---|
| 1 | the step refusal says the finish "does not answer" `loads`, which it answers (correctness-2) | fixed — "no step … runs; the steps it runs are …" |
| 2 | the step check is by name, not by moment (correctness-2) | left as is — § 2.1 states membership only |
| 3 | the card test no longer ran on the standard library alone (one-home-notes-2) | fixed — one helper makes a python allowing only a row's packages; the card test uses the monitor's |
| 4 | parse.md rule 11 and `scheduler/record.py` said "`runwrap.members`" where they meant the monitor's bundle (one-home-notes-2) | fixed |
| 5 | "entry" for three things (one-home-notes-2) | fixed — "entry" is the zip's `__main__`; `verbs_part`; `member_source` |
| 6 | the monitor's usage lines and the ending verb spelled by hand (one-home-notes-2) | fixed — `NOTIFY_VERB`; `ENDING_VERB` at its reader, `_run_ending` |
| 7 | the finish's import phrase composed twice (one-home-notes-2) | fixed — `_what_it_imports` |
| 8 | stale notes: constants' rule name, an orphan `#:` block, runfiles' "map", the profile's Check (one-home-notes-2) | fixed |
| 9 | the one-name refusal untested (one-home-notes-2) | fixed — a record part in a module travelling under another name, refused naming the record |

## The two independent full-text readers

Both found every bundle holding what 2.77.19's held, the decks unchanged, and the checks biting
(each seeded an import and saw only the bundle whose row forbids it fail). Their notes:

| # | finding (reader) | verdict |
|---|---|---|
| 1 | `VERBS` held `finish`, the analysis by hand, so prep accepted it as a step that stops on its usage line after every call has run (fulltext-b) | fixed — `VERBS` is the three steps; `finish` apart; the prep test covers it |
| 2 | "each row's rule is its `packages`" — the PySCF row's rule says more than its load (fulltext-b) | fixed — "at load" |
| 3 | "entry" left in configuration § 2.3's joining sentence and two runwrap notes (fulltext-a) | fixed — "starting module" |
| 4 | the finish's sentence says the env "carries" the packages it just failed to load (fulltext-a) | fixed — "must carry"; prep's same-calculation check reads decks only, so the run script's sentence changes nothing for a launched stage |

## The reader of the fixes

Every accepted fix is in the tree and breaks nothing it ran (the bundle files and 18 more that
use runwrap, the monitor or the ending reader; two mutants, each failing its test). Its notes:

| # | finding | verdict |
|---|---|---|
| 1 | "a step its finish does not answer" left beside the fixed refusal and in the definitions of a bundle's verbs, untrue for `loads` and `finish` | fixed — a bundle's verbs are those a kind's steps and the run script may name; its entry answers others, never as steps (runwrap's comment, `verbs_part`, `verbs`; architecture § 3.3; runs-of-points § 2.1; configuration § 2.3) |
| 2 | the step test's docstring put `finish`'s failure at the run's end; in the `after` slot it stops at the first call | fixed — "as any step it would stop on its usage line" |

## Evidence

- Every bundle's member list identical to 2.77.19's (compare_head over `members`); the PySCF
  decks identical (2142 lines); run scripts on SIESTA and PySCF optimization and vibration
  identical but for the finish-check sentence.
- Whole fast suite: 6288 passed (before round 1), 6292 (after round 2); the bundle e2e files
  (live monitor, PySCF vibration, SIESTA vibration with its finish): 55 passed before the
  fixes; with the monitor's start-log file, 56 passed after them.
- Each new test seen failing on 2.77.19 at its assertion, or, for a check 2.77.19 already met,
  seen biting a seeded import (`scipy` in `runfiles`; numpy in `report_cards.load`).
