# F12 fix 27 (one layer: every mode's state a chain or a distribution bound to the mode, 2.85.0, R3d2b2b4b) — the review

**Verdict: closed with 2.85.0.** The second of the two releases that build the user's design of 2026-11-25
(`engines/vibration-frames.md` § 2, science § 2e): the switch-over. The generic tools of 2.84.0 — `oscillator`,
`oscillator_state`, `distribution`, `sampling` — do every piece of mathematics; the molecule adds only what needs
atoms or a mode's number: `m.bind(chain_or_distribution)` → a `ModeDistribution`, `v.motion(chain_or_distribution)`,
`motion.with_mode(k, chain_or_distribution)` or a mode already bound. A frame set records what it was bound from and
is read back by binding again. Every old state type, keyword form, the doors' state dictionary, the second weights
type and the separate state words removed, nothing kept beside the new calls. The example notebook (R3d2b2c) follows.

The lessons: **one layer, built from the tools** (the user: *"no obsolete design, no obsolete api, no twist new
design to fit old needs, make sure we have one consistent api layer/framework design to support generic needs, and
our special cases are built on the combination of these tools clear logic and solid foundation"*) — the generic tools
gained only what the molecule needed and any person may use (`d.scaled(s)`, a family's and a chain's words and record,
one weights result carrying its words); the molecule's own code is the binding — a chain placed on the mode's
oscillator, its distribution put in the mode's unit, a temperature from where the person left it open — and nothing a
tool already does. **A value given is kept as given**: a distribution of a person's own in another unit, flattened
into its base by the binding, was recorded and said without its scale (round 1). **An older file is the user's
choice, each kind once**: a mode's set of 2.81–2.84 read without its state, said once; a snapshot set of 2.82–2.84
refused by name (round 1).

## What was built

`distribution`: `Distribution.scaled(s)` and `Scaled(base, scale)` (a Scaled of a Scaled one, the scales
multiplied); each family's `words` and `to_record()`, `from_record(record)` (a family made again by its own
constructor, so a record is checked as the family checks; a person's own class drawn from and placed, not read back);
`PointWeights.said` — the one weights result. `oscillator_state`: `Chain.words`, `Chain.to_record()`,
`Chain.from_record(record)` (by the starts and the events themselves). `vibration`: `Mode.oscillator`, `Mode.bind`,
`ModeDistribution` (its distribution of $Q$, the chain placed, its words, draws on the mode's own stream, frames,
`weights_at`, each atom's move, the outermost move, its record); `weights_of_frames`; `Vibration.motion(chain_or_
distribution, modes=…)`; `Motion.with_mode(k, chain_or_distribution | bound)`, `Motion.distributions`,
`Motion.background`; one temperature rule — given, else the motion's, the record's or the set's, said right after the
start's temperature (`_with_source`). `vibration_frames`: `doors_chain` and `read_state_ask` — the page's and the
command line's choices made chains; `info.state` on a mode's set, `info.sampling.background` and each mode's record on
a snapshot set; `_bound_again` reads them back by binding; `weights_for_set(rows, chain_or_distribution)`;
`outermost_move_of`. `postprocess`: `weights_for_set`, `reaveraged`, `outermost_move` take a chain or a distribution.
The command line and the spectrum tab: the same three choices, each a chain, their readout the chain's words.
Removed: `vibration.State`, `state()`, `STATES`, `checked_state`, `ModeState`, `Mode.state`, the kinds, `Weights`,
`outermost_move_ang`; `vibration_frames.doors_state`, `state_words`, `thermal_state_words`, `outermost_move`; every
keyword form of `motion`, `with_mode`, `weights_for_set`, `reaveraged`, `outermost_move`.

## The design put to the user, before the code (2026-11-26)

Two questions. **The layer**: the generic tools do the mathematics plus two additions; the molecule binds; a set
records what it was bound from; the doors' choices chains; everything old removed — *"Yes, build it"*. **A mode's set
written by 2.81–2.84**, which records no state: *"Read without state"* — read whole, said once, no reader of its old
rows.

## Round 1 — three readers (design 6, documents 13, tests 7): 26

Summaries: `review-r3d2b2b4b-r1/{design,docs}.md`; applied: `r3d2b2b4b-round1-applied.md`. **The user's**: a snapshot
set written by 2.82–2.84 and read with its vibration crashed on a missing key — *"Refuse by name"*: the reader says
what it is and to draw the snapshots again; the rest *"Yes, apply them"*. Applied: a distribution given kept as given
(`ModeDistribution.given`), its scale recorded and said (D2, X9, T1); one wording of a taken temperature, after the
start's temperature (D6); an older mode set read back refused for writing with the ways out (X10); the mode's quantum
asked of its oscillator (D5); notes and docstrings (D3, X12); documents teaching removed forms (D4, X2–X4, X6), § 9.2–
§ 9.4 archived as the decided design with the user's words (X5), the notebook's cases (X13), the seed rows (X11), the
manual's version (X8); test rows seen failing on five mutants (T1–T7).

## The reader of the last fixes (after round 1): 8 findings

None broke a road; every § 9.1 example (105) printed what the contract shows; the older files behaved as the user
chose. Applied, the user's *"Yes, apply all eight"*: § 2's `given` field documented and what `distribution.base`
holds for a distribution given already scaled; the `sampling_rule` note the writer stores made § 5.1's; three module
notes; a test comment that claimed more than its row; one leftover sentence of the keyword forms; and **the words
*bound state*** — a state held in a potential, in physics — replaced in 38 places by *the mode's state* (the user's
choice).

## Transport does the transport calculation, and that's it (the user, 2026-11-26)

Reading the second of those points — the Transport card telling a person that a set's frames follow *a thermal or
classical state's* rule — the user asked why a transport page says anything about how frames were made: *"why does
tranport card claim anything at all? how the fuck does transport knows anything how these multiframes come from?"*,
*"we have clarified that transport does transport calculation and that's fucking it"*. I had offered to make the
sentence more accurate instead of asking whether transport should say it. Since 2.81.0 the one frame-set card —
the Transport tab's, Task setup's and the Results tab's — had asked the vibration toolkit (`said_of`) for each row's
words, what the set is and where it came from. The user's choice, folded into 2.85.0: the frames and their weights
plus the set's other stored values as plain name, value and unit, no explanation added and no *where it came from*
line — *"the frame's own information is its own, transport does not interpret that at all"*. Built: the card draws
*N frames*, the set's rows as stored and a table of each frame's rows under their stored names; the rows' notes not
drawn; transport's own words kept — the weights' warnings and a calculation's cited pair no longer as pinned
(`compose.pair_changed_said`). Removed: `said_of`, the vocabulary's page words, `_shared.frame_set_said` and the
answers' `frame_set_said`, § 5.1's *on a page* column.

## The reader of the transport card and the last fixes: 8 findings

None broke a road; the card drew what § 3.8.1 says and nothing more, every caller migrated, no transport module asking
the toolkit. **The user's, on the one choice**: transport's citation refusal and the Transport tab's picker hint
defined a frame set as *the junction moved along one vibration mode* and told how to make one (*open the vibration's
result, select the mode, press Write frame set*) — the user's own way on of 2.49.3, and untrue of snapshots or a
person's frames: *"Only what it accepts"* — *a frame set: several geometries of the junction in one pair, its first
frame the relaxed one*, a vibration's own folder not cited, no how-to (the contract's three sentences the same). The
rest, *"Yes, apply them"*: the structure model's § 2.2f, the module map and the Results tab's diagram still describing
the old card; seven *bound state*s my replacement missed (capitals and line breaks); a module note; the contract's
example of the card's rows marking every row; the card's per-frame headings checked in the browser; a cited pair edited
after its calculation ran said on the Results tab's and the reuse answer's `frame_set` until put back (e2e, the real
junction); the unchanged file checked absent from the words; one test renamed.

## The e2e run

One run, last, after the reader of the transport card's fixes: `tests/test_vibration_e2e.py`,
`tests/test_siesta_vibration_e2e.py`, `tests/test_vibration_mechanics_e2e.py` and `tests/test_spectrum_results_e2e.py`
— 134 passed in 17 minutes; `tests/test_transport_on_a_real_junction_e2e.py` alone — 230 passed in 18 minutes; none
failed. Each file named for what only a real run or the browser shows: the four tools loading beside a PySCF and a
SIESTA job in their bundles; `molbuilder frameset from-mode` binding the chain the command line makes; the spectrum
tab's *Write frame set…* and its readout, the page loading; a real junction's frame set cited, re-averaged and drawn
by the frame-set card on the Transport tab and the Results tab as the set stores it; the cited pair edited after the
calculation ran, said, then put back.

## Topic → test

| topic | test |
|---|---|
| a chain or a distribution bound to a mode, its words, its distribution in $Q$; a binding of nothing refused | `tests/test_vibration_mechanics.py::test_a_chain_or_a_distribution_bound_to_a_mode`, `::test_the_bound_distribution_is_the_generic_one_in_q`, `::test_a_binding_that_is_none_is_refused_by_name` |
| the motion's background bound to every real mode; a mode set apart takes the motion's temperature or the record's | `::test_a_motions_background_is_bound_to_every_real_mode`, `::test_a_mode_set_apart_takes_the_motions_temperature_or_the_records`, `::test_a_mode_set_apart_in_a_motion_of_no_temperature_needs_the_records`, `::test_a_motion_says_what_it_takes_and_refuses_by_name` |
| frames, draws, re-weighting and the outermost move of a bound state | `::test_a_frames_geometries_are_each_atom_at_r_plus_q_l`, `::test_each_states_draws_follow_its_own_distribution`, `::test_frames_re_weighted_to_another_state_reproduce_its_averages`, `::test_a_states_own_frames_weighted_for_it_are_its_own_rule`, `::test_the_mode_moving_a_frames_atom_most_is_judged_by_its_move` |
| a distribution in another unit; a family's and a chain's record and words | `tests/test_oscillator_states.py::test_a_distribution_in_another_unit_is_its_change_of_variable`, `::test_a_bell_curve_in_another_unit_is_the_bell_curve_of_that_spread`, `::test_a_scale_that_is_none_is_refused_by_name`, `::test_a_distributions_record_makes_it_again`, `::test_a_record_that_makes_no_distribution_is_refused_by_name`, `::test_a_persons_own_class_records_nothing_to_make_again`, `::test_a_chains_record_makes_it_again_as_given`, `::test_a_record_that_makes_no_chain_is_refused_by_name`, `::test_a_chains_words_say_an_open_start_is_open` |
| a set records what it was bound from and is read back by binding again; a recipe's rows | `tests/test_vibration_frames.py::test_a_mode_set_records_the_state_it_was_bound_from`, `::test_a_set_read_back_is_the_set_written`, `::test_a_recipes_set_is_written_with_the_rows_of_its_kind` |
| a mode's set of 2.81–2.84 read without its state; the writer's refusals | `::test_a_mode_set_that_records_no_state_is_read_without_it`, `::test_the_writer_refuses_by_name`, `::test_the_reader_and_the_run_door_refuse_by_name` |
| re-weighting a set to a chain or a distribution, an open start taking the set's temperature | `::test_a_sets_rows_at_its_own_state_give_its_own_weights`, `::test_a_state_beyond_the_frames_is_said_not_refused`, `::test_an_open_start_takes_the_sets_temperature_and_says_so_first`, `::test_a_set_the_state_cannot_weight_is_refused_by_name` |
| the doors' choice one chain, asked every time, its words | `::test_the_doors_choice_is_one_chain`, `::test_the_doors_ask_the_state_every_time`, `::test_the_pages_ask_of_another_shape_is_refused_by_name`, `::test_what_a_state_means_for_a_mode_before_its_set_is_written`, `tests/spectra/test_what_a_temperature_means.py` |
| transport's answer of a cited set, its rows as stored and nothing interpreted; a calculation's pair changed since it was cited | `tests/test_frame_set_meanings.py::test_the_page_is_answered_the_rows_as_the_set_stores_them`, `::test_a_page_opens_whatever_rows_a_set_writes`, `::test_a_calculations_cited_pair_changed_since_is_said` |
| the card as a person reads it, on the Transport tab and the Results tab; the answers carrying no maker's words | `tests/test_transport_on_a_real_junction_e2e.py::test_the_shared_panel_says_which_kind_of_citation_its_values_came_from`, `::test_the_report_draws_the_junction_in_the_user_order`, `::test_the_tab_cites_a_frame_set_and_says_what_each_frame_is` |
| an older mode set's assumptions said for it | `tests/test_vibration_frames.py::test_a_mode_sets_assumptions_are_its_own_or_said_for_an_older_one` |
| end to end | `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py` (the bundles), `tests/test_vibration_mechanics_e2e.py` (the command line), `tests/test_spectrum_results_e2e.py` (the dialog and the page), `tests/test_transport_on_a_real_junction_e2e.py` (a junction's re-average) |
