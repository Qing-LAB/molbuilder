# F12 fix 22 (a frame set's words its maker's; transport reads each frame's weight, 2.81.0, R3d2b2b1) — the review

**Verdict: closed with 2.81.0.** The motion toolkit's second release (the plan's R3d2b2b split in two: transport made
generic first, so a set the toolkit's writer writes is never laid out by transport's own table of rows). The user's
rule of 2026-10-18 built: a transport calculation over frames reads each frame's `weight` and carries every other
row as written; what a row means and how a page says it is its maker's — the host module `vibration_frames`.

The lessons: a page's words must be of the frames it draws, never of a file a person may move after the run (round
2's L1, a regression round 1 made); "a state's own rule" judged on the frames of weight other than zero hides the
large error of frames that happen to fall on a shorter rule — count every frame, or know the written weights (K2, L3);
a fix's words are checked where every other copy of the rule is written (three rounds found the "says nothing" and
"m − 1" sentences in four places); an e2e row carries the refusal's words too — the last-fixes reader found the one a
rewording left behind before the final run.

## What was built

`vibration_frames` (new, L2 host): the vocabulary `ROWS` (§ 5.1, with its prose rows), `said_of(pair_path,
recorded_files=None, *, structure=None)` (the page words of the frames drawn, what the set is, its provenance said
once, an older set's assumptions, a changed or moved pair with its way on), `weights_for_set` (the state named, the
cold-swing trap, a foreign unit said, the refusals), `assumptions_of_set`, `thermal_state_words`. Transport generic:
`frame_average` (the weights' warnings, five assumptions said of frames, `answer` the rows as written,
`citation_block` `{citation, files, rows}`, `summary_lines` printing rows as written, a family told apart,
`averaged_at`), data file `transport-data@14`, record `transport-result@19`. `vibration.State.weights_at` (the degree
reproduced, `written=`, frames counted from 1). `frameset.FrameRows.meaning`, `workingcopy_structure.
pair_files_changed`. `postprocess.frame_rows`, `weights_for_set`, `reaveraged` for any state. The pages: the
frame-set card from `frame_set` and `frame_set_said` on the Transport tab, Task setup and — new — the Results tab;
MolView's Info page without the dimensionless `1`. Today's generator writes the toolkit's rows.

## The early e2e run

Before round 1 (transport and vibration e2e): 240 passed; one teardown error, the suite's guard against the working
tree changing — the documents fork edited `docs/` during the batch (see Process).

## Round 1 — six readers (correctness 3, one home 8, notes and tests 7, science 2 + 3 notes, API 5, pages 5): 30

Decisions K1–K21 (`r3d2b2b1-round1-decisions.md`). **K1, a defect two readers found**: the Results tab's report of a
calculation described but not yet prepared answered 500 (2.80.0: 200) — `recorded_junction` returns none before the
first prep. Also: the degree a set's own state is exact to and what is said of it (K2, refined while building —
counted on every frame); the state taken at the set's temperature named (K3); an older set's harmonic assumptions on
the pages (K4); the provenance said once (K5); one changed-pair comparison (K6); one rule for what a set is (K7); a
row's meaning one home (K8); prose by the vocabulary (K10); the weights' warnings once (K11); `postprocess.
weights_for_set` (K13); numeric weights (K14); a family's card (K15); frame 0's words (K17); a family's reason (K18);
tests and documents. The science reader derived every weight, moment and table digit independently.

## Round 2 — six new readers (correctness 4, one home 8, notes and tests 7, science 3 + 4, API 8 + 6, pages 5 + 2)

Decisions L1–L26 (`r3d2b2b1-round2-decisions.md`); every round-1 decision found built. **L1, a regression round 1
made**: the card's words read from the cited pair were lost when a person moved or deleted it after the run (2.80.0
showed them) — the words are now of the frames the page draws. Also: an older set told by its own marks (L2); a set
at its own state by its written weights, an even rule's included (L3); refusals naming a kind (L4); the degree's words
(L6); the 0 K quiet (L7); assumptions beside `averaged_at` and `reaveraged` (L8); a foreign unit said (L9); the
cold-swing words neutral (L10); the warning tone (L11); frames counted from 1 (L12); MolView's `1` (L13); the plan's
R3d2b2b2 notes (L17). The science reader worked the degree rule in exact arithmetic: the 1e-9 bound sits 10⁵ from both
sides.

## The full-text readers and the discipline reader (fulltext-A 10, fulltext-B 8, discipline 13): 31

Decisions M1–M23 (`r3d2b2b1-round3-decisions.md`). **M1, for the user**: reading sets older releases wrote (the
spread's older name, `info.vibration`, an older set's assumptions) is the user-approved contract's (vibration-frames
§ 5.1) and conflicts with the standing rule (profile; testing.md § 6) — kept as the contract says, put to the user at
the commit. Also: pyflakes (M2); an off-road test case retired (M3); tests through the road's doors (M4, M5); the
pages' words (M7); notes and thresholds (M8, M9); one home for `said_of`'s words (M10); `reaveraged`'s assumptions
named for the state they were written for (M11); a family one meaning (M13); the record's citation in the
calculation's own tree (M16); small lines (M12, M14, M15, M17–M23).

## The reader of the last fixes: 8

All taken (`r3d2b2b1-lastfixes-decisions.md`): **N1, an e2e row still matching the refusal's old words** — it would
have failed the final run; the other copies of "says nothing" (N2); M7 finished (N3); a NaN weight's refusal (N4); a
docstring's history (N5); an older set's words named molbuilder's (N6); the changelog (N7); two notes (N8).

## The e2e run

The last, after every fix: `tests/test_transport_on_a_real_junction_e2e.py`, `tests/test_vibration_e2e.py`, `tests/test_inspector_registry_e2e.py`, `tests/test_spectrum_results_e2e.py`, `tests/test_vibration_mechanics_e2e.py` and `tests/test_pages_no_js_errors.py` — 392 passed, 0 failed (28 min). The whole fast suite before it: 6624 passed, 0 failed.

## Process

- The documents fork edited `docs/` while the early e2e batch read the checkout; the suite's guard caught it (one
  teardown error, the 240 tests passed). Later forks ran while no batch ran.
- A round-2 notes reader's mutation run started the whole suite in its scratch copy; stopped, contained to scratch
  (HOME, TMPDIR, XDG, MOLBUILDER_* pointed there), checked.
- A round-2 pages reader ran a probe with raw pytest first, then again through `tools/testrun.py`.

## Topic → test

| topic | test |
|---|---|
| a page's words by the vocabulary's unit rule | `tests/test_vibration_frames.py::test_a_page_says_a_row_in_the_vocabularys_words_where_its_unit_is_the_vocabularys` |
| a row's meaning, one without said once | `::test_a_rows_meaning_is_the_sets_and_one_written_without_is_said_once` |
| provenance said once | `::test_a_set_says_where_it_came_from` |
| a changed pair, a moved pair, a gone pair | `::test_a_pair_that_changed_since_its_citation_is_said`, `::test_the_words_are_of_the_frames_drawn_whatever_became_of_the_pair`, `::test_a_pair_that_does_not_read_is_said_and_gives_no_words` |
| what a set is | `::test_a_set_says_what_it_is` |
| an older set's assumptions | `::test_a_mode_sets_assumptions_are_said_beside_it` |
| a set's own state, odd and even rules | `::test_a_sets_rows_at_its_own_state_give_its_own_weights`, `::test_an_even_rules_set_at_its_own_state_says_nothing_of_another` |
| the state named, the cold-swing trap, a foreign unit, the refusals | `::test_a_swing_named_alone_on_a_warm_set_says_its_temperature_is_not_used`, `::test_a_coordinate_in_another_unit_is_read_as_written_and_said`, `::test_a_set_the_state_cannot_weight_is_refused_by_name` |
| a state's own rule, the degree, frames from 1 | `tests/test_vibration_mechanics.py::test_a_states_own_frames_weighted_for_it_are_its_own_rule`, `::test_frames_re_weighted_to_another_state_follow_the_sciences_table` |
| transport's rows as written, weights alone, five assumptions, a family, `summarize`, the citation, the data file | `tests/test_frame_set_meanings.py` |
| the Results tab before a first prep | `tests/test_transport_task.py::test_a_described_calculations_report_opens_before_its_first_prep` |
| end to end: the average, the record, @14, `frame_rows`, `reaveraged`, `averaged_at`, the describe/folder/record/reuse answers, the card on two pages | `tests/test_transport_on_a_real_junction_e2e.py` |
