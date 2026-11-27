# F12 fix 29 (the frame weights, required, 2.86.0) — the review

**Verdict: closed with 2.86.0.** The second of the three releases the baseline review of 2.85.0 and the user's
decisions of 2026-11-26 set (2.85.1 the clean-up, 2.86.0 the weights, 2.87.0 the older files dropped). Every frame of
a frame set states its weight and they sum to 1 — refused by name otherwise, wherever frames are averaged or cited;
the weights assigned by a call a person makes; every analysis on the weights the set stores; a finished calculation's
per-frame results averaged at a person's own weights.

The lessons: **a number a calculation computes with is stated, never assumed** (the user: *"you should not assume
equal weight for all frames, weight is explicitly calculated and saved with multiframe data, and is required, and will
be checked for normalization"*; *"there must be a weight for calculation otherwise how would transport get the final
average? if there is no weight then the user need to call a function to assign equal weight"*) — transport averaged
at weights summing to 1.3 with a warning, and an analysis counted 1/N for itself; both gone. **A rule changed changes
its test roads too**: two end-to-end tests outside the release's list built frame sets without weights — the test
road's helper and the real junction's "family" — which the citation now refuses (round 1, the tests reader).

## The user's decisions (2026-11-26)

- **Weights**: required and checked for normalization; a sum off 1 *"Refuse"*; *"Yes, use the saved weights, and api
  should allow user to take results from individual frames and assign its own weight as a post-processing and
  analysis - this is not transport's own calculation but talking about api support for data extraction and
  processing"*.
- **Mine, said to the user before the build**: equal weights are $1/F$ on every frame, frame 0 included (frame 0 out of
  the average is a weight of 0 the person gives); a calculation composed before 2.86.0 on frames that break the rule
  opens with every frame's points and no average, its reason the rule's and the way on.

## What was built

**The rule's home** (`frameset`): `_weights` (every frame a finite number, the sum by `sum_broken`), `weights_of`
(asked of a structure: the citation, the toolkit's reader), `read` answering the weights only where they meet the rule
and `warnings` how they break it, `weights_said`, `FrameWeightsError`; **assigned** by `with_weights` and
`equal_weights`, handed on by `postprocess`. **Transport**: the citation refuses a frame set whose weights break the
rule (`compose._classify_pair`, `CITATION_CONDITION`); the record's average only at weights that meet it, else no
average, its reason `NO_WEIGHTS_WHY` (the way on, `postprocess.averaged_at`) and its `warnings` the rule's words;
`averaged_at` checks the sum (`sum_broken`) and is handed on as `postprocess.averaged_at`; the "family" gone (the
card shows *N frames* alone for frames stating no rows). **The toolkit**: `write_frame_set` refuses a person's weights
off the rule; `read_frame_set` a set that breaks it; `weights_for_set` rows that break it, and one structure's;
`Snapshots.written_weights`, and `standard_error` and `parts` at the weights the set stores, a value missing dropping
its frame or pair and the rest scaled back to 1, said. **The statistics** (`sampling`): `mean_with_error(units,
weights)`, `pair_means(values, weights)` → `PairMeans`, `even_odd_parts(values, weights)` — the reliability-weighted
mean and error (science § 5.5), a value or pair at weight 0 adding nothing, the weights refused unless one finite
number per value, 0 or more, summing to 1; `WEIGHT_SUM_TOLERANCE` restated (it travels in a job's bundle), a test
holding it one with the container's. **Documents**: `model/structure.md` § 2.2f, the rule's home; `engines/transport.md`
§ 2a.9, § 2a.12, § 3.1, § 3.8.1 and its code map; `engines/vibration-frames.md` § 3.3, § 3.4, § 5, § 7, § 8, § 9.1
module 4; `science/vibrational-averaging.md` § 2c.3, § 2c.4, § 4.1, § 5.5 (the weighted error, derived);
`engines/transport-postprocessing.md` § 3.2 (*Your own weights*); `model/datafiles.md`, `model/parse.md`,
`web/web-api.md`, `web/results.md`, `execution/run-reports.md`, `architecture.md`.

## Round 1 — three readers (documents 8, code 5, tests 4): 17

All applied, the user's *"Yes, apply them all"*. **The documents**: transport § 2a.9 still said a sum off 1 is
averaged, § 3.1 that nothing of a set's rows is checked; structure § 2.2f listed `averaged_at` among the doors that
refuse a set and called the rule one check of every door; the post-processing page's *More frames* led to a refusal
(two sets' weights summing to 2); one structure needing no weight unsaid; the manual's version; two pages' and the
code map's old condition for an average. **The code**: a calculation whose frames' grids differ said its weights were
none, and its Export failed on it (older); re-averaging one structure's rows stopped with a Python error; a value or a
pair at weight 0 turned the statistics into NaN; leftover notes; the writer's second, silent weights pass; the weights'
sum written in two places (now `frameset.sum_broken`). **The tests**: the test road's frame sets and the real
junction's "family" citation stated no weights (three deck tests and one tab test would have failed); a test claiming
what no set the toolkit writes can show (narrowed); the spread at unequal pair weights untested.

## The reader of the fixes: 5 findings

Every fix did what its finding asked; none broke a caller, a page, a refusal's words a test matches, or a doc
pointer; the new tests failed on the tree as round 1 read it. Applied, the user's *"Yes, apply them"*: the data file's
own definitions of `weight_sum` and `weight_tolerance` (and `frame_warnings`, `average_assumptions` on its page) said
their blank meant *the frames state no weights* — for a calculation composed before 2.86.0 whose weights sum to 1.2 it
means *the weights break the rule*; a pair at weight 0's mean (its two values' plain one) checked and stated in the
manual; the even part at unequal pair weights checked beside the odd; notes calling `weights_of` every door's one
check (the rule is one, asked three ways: of a structure, of its rows, of weights given); two pages restating a fact
where they now point to its home.

## The e2e run

One run, last, after the reader of the fixes: `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py`,
`tests/test_vibration_mechanics_e2e.py`, `tests/test_spectrum_results_e2e.py` and `tests/test_transport_decks_e2e.py`
— 164 passed in 18 minutes, the deck tests citing frame sets the test road now weighs equally; and
`tests/test_transport_on_a_real_junction_e2e.py` alone — 232 passed in 19 minutes: the frame set's weights taken out,
refused as a citation by name with the way on, its weights off 1 refused, given equal weights and cited; the Transport
tab's describe answering the refusal and its shared panel none; a person's own weights averaging each frame's results
from the data file, the file unchanged; the frames re-averaged at another state; none failed.

## Topic → test

| topic | test |
|---|---|
| the weights' rule, its one check; weights assigned, equal or a person's own; one structure not weighed | `tests/test_frame_set_meanings.py::test_weights_are_required_and_assigned_by_their_rule`, `::test_weights_that_break_the_rule_are_not_assigned`, `::test_one_structure_has_no_frames_to_weigh` |
| the record's average only at weights that meet the rule, the reason and the way on otherwise; a grid mismatch keeping the weights | `tests/test_frame_set_meanings.py::test_the_record_reads_the_weights_alone_and_says_its_own_assumptions`, `::test_frames_with_no_average_say_why`, `::test_frames_on_another_grid_keep_their_weights_beside_the_refusal` |
| the statistics at the weights given; a value at weight 0 adding nothing; the weights refused by name; the container's tolerance | `tests/test_oscillator_states.py::test_a_means_error_is_the_spread_at_the_weights`, `::test_weights_the_statistics_cannot_take_are_refused_by_name`, `::test_the_statistics_weigh_by_the_containers_rule`, `::test_a_paired_sets_even_and_odd_parts_are_their_spreads`, `::test_fewer_than_two_units_leave_no_spread` |
| the toolkit's writer, reader and re-weighting refusing weights that break the rule; a snapshot set's analyses on its stored weights | `tests/test_vibration_frames.py::test_the_writer_refuses_by_name`, `::test_the_reader_and_the_run_door_refuse_by_name`, `::test_a_set_the_state_cannot_weight_is_refused_by_name`, `::test_a_snapshot_sets_analyses_take_the_weights_it_stores`, `tests/test_vibration_mechanics.py::test_an_analysis_says_what_it_dropped_and_few_frames_say_their_degree` |
| a frame set cited without its weights refused, then assigned and cited; the tab's refusal; a person's own weights on a finished calculation | `tests/test_transport_on_a_real_junction_e2e.py::test_a_frame_set_cited_without_its_weights_is_refused_and_assigned`, `::test_the_tab_cites_a_frame_set_and_answers_its_rows_as_stored`, `::test_a_persons_own_weights_average_each_frames_results`, `::test_the_frames_are_reaveraged_at_another_state` |
| end to end | `tests/test_transport_on_a_real_junction_e2e.py`, `tests/test_transport_decks_e2e.py` (frame sets on the test road), `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py`, `tests/test_vibration_mechanics_e2e.py`, `tests/test_spectrum_results_e2e.py` |
