# F12 fix 26 (a mode's state built in steps, the generic tools, 2.84.0, R3d2b2b4a) — the review

**Verdict: closed with 2.84.0.** The first of the two releases that build the user's design of 2026-11-25
(`engines/vibration-frames.md` § 9, science § 2e): the generic tools — `oscillator`, `oscillator_state`,
`distribution`, `sampling` — handed on by `postprocess` as `Oscillator`, `states`, `dist`, `oscillator`, `sampling`,
every call written out in the contract's § 9.1 (what it stands for, assumes, computes, refuses); the modes that move
chosen in steps (`v.mode_selection()`); every number the tools fix documented (science § 2e.6), the precision a
parameter, one equality epsilon. Removed by the user's word: joining snapshot sets (`pooled`), the drive groups —
degenerate modes set up by hand (science § 2c.2, contract § 3.1) — and, after round 4, the push and the steady drive:
a state is its list. The switch-over of the vibration and frame-set code follows in R3d2b2b4b, then the example
notebook (R3d2b2c).

The lessons: **molbuilder writes frames and offers generic tools; it does not answer science** (the user: *"when the
fuck is it molbuilder's job to answer any scientific question?"*) — joining snapshot sets was mine, never asked for,
and went; mirrored pairs are the user's agreed option and stayed. **A snapshot is one geometry** (*"a snapshot is not a
fucking average by its fucking definition"*): each mode's coordinate is one random draw, its moment in its own motion
random too, so a state is its list $p_n$ — the push, its phase, states kept at one moment, the time average, the
density-matrix route and the steady drive went, with the classes, tests and documents built for them through four
rounds. A feature needs a process on molbuilder's road that makes it, on that process's time scale (*"it better make
meaningful time scale relevant"*): a coherent swing lasts picoseconds against a measurement's seconds. **Check every
parameter against its physics before it reaches the user** (*"who the fuck would push something 400 quanta??"*): beyond
about 1 eV in a mode is a bond breaking, not a case. **Every call explained** (*"explain the assumption of the
calculation and the actual operation for each api"*), names that say what they compute (`level(n)`, `thermal(...)`,
`bose_einstein_mean_quanta`, the user's own suggestion). **No compatibility with a design never finished**, nor with
the toolkit's own older snapshot files (*"i don't give a fuck to old snapshots"*). **A search checks its start first**
(*"isn't the bug in the search function that you should first check if you already have the right value?"*), tested
from deliberately wrong starts. **A threshold is a documented number or a parameter, an equality one epsilon** (*"why
would you not limit comparison between floating numbers to a reasonable epsilon?"*). **No code while the user is still
asking** (*"who the fuck allowed you start coding?"*).

## What was built

`oscillator`: `Oscillator(frequency_cm1)` (`zero_point_amplitude_amu12_ang`, `quantum_ev`,
`bose_einstein_mean_quanta`, `bose_einstein_temperature`, `classical_spread`), the mechanics moved from `vibration`
(`zero_point_amplitude_amu12_ang`, `bose_einstein_mean_quanta`, `bose_einstein_temperature`, `spread_of_quanta`,
`thermal_spread_amu12_ang`, `classical_spread_amu12_ang`, `checked_temperature_k`), the levels (`level_wavefunction`,
`level_density`, `level_moment`), the ladder and position on a window (`lowering_matrix`, `raising_matrix`,
`position_matrix`), one recurrence, one table of a level, one search (`_bracketed_newton`, check-first), the one
epsilon. `oscillator_state`: the starts (`ground`, `level`, `thermal`, `classical_thermal`), the events
(`add_quantum`, `remove_quantum`), `Step`, `Chain.reduced`, `Chain.on(oscillator, temperature_k=None,
precision=1e-16)` → `OscillatorState` (`populations`, `populations_for`, `mean_quanta`, `temperature_k`, `words`); a
thermal start and its events by one formula, $p_n \propto c(n)\,t^n$. `distribution`: `Distribution`, `Gaussian`,
`QuantumAdded`, `Level`, `LevelList`, `of_state`, `points_weights`; every distribution symmetric, its draw from the
lower tail. `sampling`: `normal_stream` (one stream per mode, `(seed, mode, 0)`), `mean_with_error`, `pair_means`,
`even_odd_parts`. `vibration`: `ModeSelection` (`window`, `only`, `add`, `drop`, `final`),
`Vibration.mode_selection()`, `v.motion(..., modes=selection)`; `Snapshots.standard_error(values)` and
`parts(values)` on one array; the driven states gone from the states' table, the doors, the command line and the page.

## Round 1 — five readers (correctness 7, one home 12, notes and tests 12, API 9, science 4): 44

Decisions A1–A21 (`r3d2b2b4a-round1-decisions.md`): a quantum removed from a state that holds none refused by name
(A1); one bracketed Newton for the level table and the distributions (A2); the density matrix cut by § 2e.4's rule
(A3); two pushes cancelling at the push's own scale (A4); a time average's list clipped at 0 (A5); a state's arrays
read-only (A6); a thermal start keeps its temperature, 0 K included (A7); a classical chain no route builds refused
(A8); the quanta word rule (A9); the inverse temperature, the quantum, the ladder (A10, A11); `postprocess` hands on
the modules (A12); a chain's numbers once placed (A13); § 9.2's by-hand oscillator (A14); what travels said true
everywhere (A15); one recurrence, one table shape, one Lanczos (A16); one door each for whole numbers, quanta, spreads
(A17); the standard library's normal quantile (A18); the science's heading (A19); old pointers (A20); `sampling`'s
doors (A21). A3–A5 went with the push.

## Round 2 — five new readers (correctness 5, one home 14, notes and tests 10, API 5, science 3)

Decisions B1–B22 (`r3d2b2b4a-round2-decisions.md`). **The user's, after round 2**: a thermal start typed at exactly
0 K refused for a removal, above it one formula needing no division; the search check-first, tested from wrong
starts; one epsilon (10⁻¹²); every threshold in science § 2e.6, the precision a parameter; joining removed, drive
groups removed (by hand, two paths documented). Also: a conflicting temperature refused (B5); soft warm modes in
seconds (B6, the position as its tridiagonal); a stream's key whole numbers (B7); one family finder, one "holds none"
message (B16–B18); the science and notes (B9–B15, B19–B20); tests (B21). The typed density matrix (B2) and the push's
window went with the push.

## Round 3 — five new readers (one home 19, API 6, notes and tests 8, science 9, correctness 4)

Decisions D1–D3 and F1–F7 (`r3d2b2b4a-round3-decisions.md`). **The user's**: `only` a filter (D1); `pair_means` and
`even_odd_parts` take one value per snapshot and group them two by two themselves, an odd count refused, written out
in the contract § 7, science § 5.5 and transport-postprocessing § 3.2 with example code (D2). D3 (an older set's
refusal) withdrawn in round 4 by the user's word. Also: a thermal start and every event by one formula, c(n) (F2);
a typed list's rounding speck (F4); the selection's refusals (F5); tests (F6); documents and notes (F7).

## Round 4 — two full-text readers and a discipline reader (11, 8, 10)

Decisions G1–G3 and fixes (`r3d2b2b4a-round4-decisions.md`): the frames' degree check on the one epsilon (G2); one
door per distribution, a level's table on first use (G3); every handling of files made with the removed features gone
(the user's); the cut placement's example the code shows (a 5 cm⁻¹ mode at 3000 K, 15 330 for 15 364); the epsilon's
uses complete; the floors; the contract's refusals; the position's elements written once; dead code out; tests through
`postprocess`, split one question each. G1 (a state kept at one phase) went with the push.

## The user's decisions after round 4 (2026-11-25)

The user asked what the push stood for — its time scale, its phase, why two pulses for one snapshot — and found no
process on molbuilder's road that makes it: **the toolkit cut to lists**. Gone: the push and its phase, states kept at
one moment, `time_averaged`, two pushes adding, the density-matrix route (`DensityMatrix`, Cholesky, pure states),
the swing (`Sinusoid`, the classical swing, Chebyshev's rule), `uniform_stream`, `phases_of`, the angular frequency,
`classical_rest` (a classical thermal state at 0 K is the point), and the steady drive — the *driven* and *warm
driven* states of 2.78.0–2.83.0 — from `vibration`, `vibration_frames`, the command line, `postprocess` and the page.
Science § 2a.2 now says what a coherent state is, the experiments that make one and why molbuilder does not model
it. **Names**, the user's yes on a holistic review: `states.level`, `states.thermal`, `states.classical_thermal`;
`bose_einstein_mean_quanta` and `bose_einstein_temperature` (the user's suggestion); `zero_point_amplitude_amu12_ang`;
`dist.QuantumAdded`, `dist.LevelList`; `gauss_rule(nodes)`; the kind word `"level"`; `State.thermal_quanta`;
`ModeSelection.window(low_cm1, high_cm1)`. **Every call documented** — the contract's § 9.1 rewritten as a per-call
reference.

## Round 5 — a discipline reader and two full-text readers (11, 6, 10)

What was applied: `r3d2b2b4a-round5-applied.md`. **The user's**: the snapshot sets 2.82.0 and 2.83.0 wrote get no
handling (*"i don't give a fuck to old snapshots"*). Also: the assumptions row's science items; one stream per mode
in the `rng` and seed rows; dead code out (`_rule_from_vector`, the bath-spread helpers, a family's second value);
notes and refusal words after the renames; odd averages, quantiles and a distribution's own frames tested over every
family (four mutants seen caught); the documents' drive leftovers; references (one uncited entry removed, three notes);
§ 9.1 completed — the seven mechanics functions, `Step`, `Distribution`, a refusal column; § 2e.6's epsilon margin
re-measured (1.8×10⁻¹⁴ up to seven frames, about 55 times below; 10⁻¹³ at nine).

## The reader of the last fixes (after round 5): 9 findings

The kinds counted three everywhere; transport § 3.2 offering only a state the toolkit sets today; the stream keys as a
set records them (the `seed` and `start` rows); two refusals placed where they happen; the docs index; the bell
curve's own draw; **the cut's margin tested on a real case** (30 cm⁻¹ at 300 K, one quantum added, to 10⁻⁸: 244
levels, 243 without it, leaving out 1.04 of the precision) — the reader's 3000 K example, and two more in the science
and the decision record, replaced: no junction is at 3000 K; the page's "1 quantum" step narrowed to what it holds
(a mean of 1.0004 on the water molecule's softest mode would need about 3300 K). Dropped: "got one values" for a
single number nobody passes.

## Round 6 — the contract consolidated, and the API manual (the user, 2026-11-25)

*"review again to consolidate, clean up holistically so that we have a consistent and complete contract"*; *"api
document/manual should be updated with clear text and equation/diagram/example/pseudocode to explain exactly what it
does and examples how it is used"*. § 9.1 rewritten as a manual — every call of the four modules with its equation,
pseudocode, refusals and an example whose output was printed by running it (90 examples, re-run by two readers, every
output identical); the molecule's calls to come moved into § 9.3. Three readers — consistency (13), the manual's
correctness (8), usable by a scientist (8). Applied (`r3d2b2b4a-round6-applied.md`): § 9.1 the manual, kept; today's
route from a chain to a written frame set (§ 9.2), run end to end; the open-temperature refusal pointing to what
exists; the `.on` rules as the code applies them; the units of a re-weighting; one key for keys differing by trailing
zeros; spread, frame and frame 0 defined; one home — `parse.md` § 3 and the guides pointing to the manual; untrue
lines in `vibration.md` § 6.11 and § 8.1; the science map; a test's 3000 K. Dropped: "got one values".

## The reader of the last fixes (after round 6): 3 findings

The re-weighting's unit in `Distribution.weights_at`'s docstring; § 9.2's classical row; § 9.3 and the plan saying
§ 9.2–§ 9.4 move, § 9.1 stays. Below the bar but applied: the stream key a sequence, its refusal and its test.

## The e2e run

One run, last, after round 6's fixes: `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py`, `tests/test_vibration_mechanics_e2e.py` and `tests/test_spectrum_results_e2e.py` — 134 passed in 17 minutes; `tests/test_transport_on_a_real_junction_e2e.py` alone — 229 passed in 18 minutes; none failed. Each file named for what only a real run or the browser shows: the four tools loading beside a PySCF and a SIESTA job in their bundles; `molbuilder frameset from-mode` on a real record without the drive; the spectrum tab's *Write frame set…* dialog without the driven choice, and the page loading; a real junction's frame set re-averaged.

## Topic → test

| topic | test |
|---|---|
| each chain's list, the events' means, the order of events, reduction | `tests/test_oscillator_states.py` (`test_each_chain_makes_the_sciences_list`, `test_the_steps_move_the_mean_as_the_science_says`, `test_two_chains_that_make_one_state_reduce_to_one`) |
| quanta removed: cold, too cold for a number, k removals, a thermal state of no quanta | `::test_a_quantum_taken_from_a_cold_mode_leaves_twice_its_mean`, `::test_a_quantum_taken_from_a_mode_too_cold_for_a_number_is_its_limit`, `::test_quanta_taken_from_a_thermal_state_are_their_formula`, `::test_a_thermal_state_of_no_quanta_is_the_ground_states_curve` |
| the precision and the cut, placed on its level | `::test_a_lists_precision_is_a_parameter`, `::test_a_list_is_cut_where_the_averages_asked_of_it_stop_changing`, `::test_a_list_leaves_out_less_than_its_precision_and_no_level_more` |
| the closed families against the list's route; a level's rule; the worked chain | `::test_an_exact_family_and_the_lists_route_are_one_distribution`, `::test_a_thermal_state_with_a_quantum_taken_is_its_list_distribution`, `::test_a_levels_own_rule_reproduces_its_averages`, `::test_the_worked_chains_three_frames_are_the_sciences` |
| every family: odd averages, quantiles, its own frames read as its own | `::test_every_odd_average_is_zero`, `::test_a_quantile_is_where_the_distribution_reaches_its_chance`, `::test_a_distributions_own_frames_are_read_as_its_own` |
| the search from wrong starts; draws where F reaches Φ(z); one quantum's middle | `::test_the_search_finds_the_point_from_every_wrong_start`, `::test_each_draw_is_where_the_distribution_reaches_phi_of_its_number`, `::test_draws_from_one_normal_number_follow_the_distribution`, `::test_one_quantums_middle_is_at_rest` |
| the oscillator: its numbers, its levels, its ladder, the classical forms | `::test_the_oscillators_own_numbers_are_the_constants`, `::test_a_levels_wavefunction_is_hermites`, `::test_the_ladder_operators_are_the_sciences`, `::test_the_classical_forms_have_no_zero_point_motion` |
| the statistics and streams | `::test_a_means_error_is_the_spread_over_root_n`, `::test_a_paired_sets_even_and_odd_parts_are_their_spreads`, `::test_fewer_than_two_units_leave_no_spread`, `::test_a_stream_is_numpys_generator_seeded_with_its_key` |
| refusals and words; an open temperature; read-only lists; a typed list's speck | `::test_a_step_that_is_none_is_refused_by_name`, `::test_a_states_words_are_its_steps`, `::test_an_open_temperature_is_filled_where_the_chain_is_placed`, `::test_a_states_list_cannot_be_edited`, `::test_a_typed_lists_rounding_speck_below_zero_is_zero` |
| the modes chosen in steps, their words and refusals | `tests/test_vibration_mechanics.py::test_the_modes_that_move_are_chosen_in_steps`, `::test_a_mode_selections_steps_are_said_in_the_contracts_words`, `::test_a_mode_selection_that_moves_nothing_is_refused_by_name` |
| frames re-weighted to another state; a classical state's temperature | `tests/test_vibration_mechanics.py::test_frames_re_weighted_to_another_state_reproduce_its_averages`, `::test_a_classical_state_takes_the_records_temperature_and_says_so` |
| a set's state beyond its frames said; a state given no temperature | `tests/test_vibration_frames.py::test_a_state_beyond_the_frames_is_said_not_refused`, `::test_a_state_given_no_temperature_takes_the_sets_and_says_so` |
| end to end | `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py` (the bundles), `tests/test_vibration_mechanics_e2e.py` (the command line), `tests/test_spectrum_results_e2e.py` (the dialog and the page), `tests/test_transport_on_a_real_junction_e2e.py` (a junction's re-average) |
