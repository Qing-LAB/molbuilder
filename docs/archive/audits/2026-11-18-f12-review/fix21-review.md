# F12 fix 21 (the molecule's motion in `vibration`, 2.80.0, R3d2b2a) — the review

**Verdict: closed with 2.80.0.** The first of the motion toolkit's three releases (the user's design of
2026-11-23, `engines/vibration-frames.md`; the plan's R3d2b2 split into R3d2b2a, b and c): the travelling module
`vibration` gains every state's own Gauss rule, draws, re-weighting, and the molecule's motion and snapshots.

The lessons: a key that must stay unique cannot be kept from a member that may leave — key a group by what made it;
three readers found the collision a round-1 fix opened, by three routes; a rule written once per caller drifts
within a release (the temperature rule, the words of a driven state, the geometry) — one function each; words a
writer will write as a set's row are part of the contract, and a group of one is singular; a statistical row's
tolerance comes from its own error, or it fails one seed in four; a convergence test one ulp too strict runs every
point to its cap; a refusal the contract does not state is a floor of ours (`pooled` of one set).

## What was built

`vibration`: the states one table — a driven mode by its swing quanta alone the coherent state, warm with a
temperature, the classical swing carrying the classical bell curve; `cdf` on arrays (`_NumberDistribution` for n
quanta); every state's frames its own Gauss rule (Lanczos on its measure; a swing's discretized, Gautschi 1982;
Chebyshev's for the bare swing), exactly symmetric, in the set's order, an even rule's frame 0 prepended (`Frames`:
`q_amu12_ang`, `state`, the moves and the mover); `draw` from the mode's own streams; `weights_at` → `Weights`;
`temperature_of_quanta`; the `Vibration` holding what every mode shares (`mode`, `displaced`, `with_frequency_scale`,
`motion`, `read_path`, `structure`); `Motion` (`with_mode`, `with_drive` keyed by the modes made with, `said`,
`set_apart_modes`, `sample`); `Snapshots` (`standard_error` → `Estimate`, `parts` → `Parts`, `pieces`); `pooled`.

## Round 1 — five readers (correctness 7, one home 11, notes and tests 14, science 4, API 13): 49

Decisions K1–K15 (`r3d2b2a-round1-decisions.md`): a drive group keeping its stream when a member leaves (K1, its
key replaced by L1); one motion by content (K2); a pooled set's values either way (K3); `v.read_path` the path,
`v.structure` filled (K4); a state beside state arguments refused (K5); a set-apart mode at the motion's temperature
(K6); what an analysis dropped said (K7); a cold driven mode's two baselines (K8); columns in the result's order
(K9); three frames blind to the shape (K10); one home each (K11); the n-quanta quantile's convergence, 17 s → 2.8 s
at n = 1000 (K12); tests (K13); documents with Gautschi 1982 (K14); R3d2b2b's needs in the plan (K15). Dropped:
seeds of 2³² or more; a sub-kelvin carried bell curve's arrays. The science reader found the science right: every
rule exact and symmetric to m = 30 and Q₀/σ = 2000, § 5.4's table to every digit, the errors calibrated over 600
sets.

## Round 2 — five new readers (correctness 6, one home 6, notes and tests 13, science 6, API 11): 42

Decisions L1–L12 (`r3d2b2a-round2-decisions.md`). **L1, a defect three readers found independently**: K1's kept key
let a mode that left a group start another under the same key — two lasers phase-locked (correlation 0.66, z = +321),
silently; a group is now keyed by the modes it was made with, `[seed, 0, *modes]`, unique among the groups a motion
holds. Also: pooling a pooled set (L2); a driven state's words one function, a group of one singular, a classical
group's temperature said (L3); the classical motion's baseline naming `classical=False` (L4); the temperature rule
one function (L5); the few-frames sentence (L6); refusals by name (L7); the read path as given (L8); notes (L9);
documents — § 2a.3's reason conditioning (L10); tests (L11); `info.sampling`'s needs for R3d2b2b (L12).

## The full-text readers and the discipline reader (fulltext-A 10, fulltext-B 9, discipline 12): 31

Decisions M1–M10 (`r3d2b2a-round3-decisions.md`): one set pools as itself (M1, product stance); a motion's words as
it stands (M2); the reprs (M3); one geometry rule for frames and snapshots (M4); `Estimate`/`Parts` compared as
objects (M5); `ModeState` handing on its state's fields (M6); `__all__` (M7); tests — the contract's refusals, the
tolerances from their own errors (M8); notes (M9); documents (M10). Dropped: § 9's visible-change list (the design's
one home of them, G18); the three fields R3d2b2b fills.

## The reader of the last fixes

Ten findings, none breaking the workflow: the pair-error row loosened by M8 let an error off by √2 through (the reader's mutant passed 39 of 201 seeds) — now 40 000 pairs, the tolerance 12 %, caught at every seed; a drive group's warm advice named a call the motion refuses — now the same drive made anew; M2 untested — now a row; `ModeState`'s handed-on fields visible to `dir()`; the note's numbers as measured; a one-mode group's leaving words; four document sentences (§ 3.1's advice, § 3.3's order, § 6.11 § 6's `ModeState` members, the plan's `info.sampling` list with the modes left out). All fixed; the targeted set and the documents' tests, 347 passed.

## The e2e run

The vibration road end to end, once, last: 366 passed — the H₂ record and the script door, the spectrum results and the spectra page on PySCF and SIESTA, the PySCF and SIESTA vibrations, a real junction's frame set and its re-average, the page with no spectrum, the vibration view, the Results export; 0 failed. The whole fast suite before the last fixes: 6591 passed, 0 failed; the targeted set after them, 347 passed.

## Process

- The discipline reader's first two searches were `grep -rn` from the repository root and read through `projects/`
  (nothing printed); later briefs say `git grep`.
- A `sed` of mine left a doubled "(the" in the plan's header; fixed with the third pass.

## Topic → test

| topic | test (`tests/test_vibration_mechanics.py`) |
|---|---|
| each state's draws, its own distribution | `test_each_states_draws_follow_its_own_distribution` |
| a draw's seed continued | `test_a_draw_continues_its_seed_from_the_modes_own_stream` |
| every rule symmetric, in the set's order | `test_a_modes_frames_are_exactly_symmetric_in_the_sets_order` |
| the swing's rule as § 2a.3 works it | `test_a_swings_frames_are_its_own_gauss_rule_as_the_science_works_them` |
| re-weighting, § 5.4's table | `test_frames_re_weighted_to_another_state_follow_the_sciences_table` |
| a mean number of quanta's temperature | `test_a_mean_number_of_quanta_names_its_temperature` |
| snapshots' covariance | `test_snapshots_of_a_thermal_motion_have_the_sciences_covariance` |
| a drive group's covariance and closed form | `test_a_drive_groups_modes_swing_in_step_by_the_persons_lags` |
| mirrored pairs | `test_mirrored_pairs_cancel_the_odd_part_exactly` |
| the seed, continuation, pooling | `test_a_seed_holds_a_set_and_its_continuation` |
| the drive key | `test_a_drive_group_keeps_its_phases_when_a_member_is_set_apart_again`, `test_two_drives_made_apart_keep_apart_phases` |
| a driven mode's normals its baseline's | `test_a_driven_modes_normal_numbers_are_its_baselines_own` |
| the motion's temperature, classical forms | `test_a_motions_set_apart_modes_take_its_temperature_and_its_classical_form` |
| the analyses' indexing and words | `test_every_analysis_takes_one_value_per_frame_frame_0_first`, `test_an_analysis_says_what_it_dropped_and_few_frames_say_their_degree` |
| the mover mode | `test_the_mode_moving_a_frames_atom_most_is_judged_by_its_move` |
| n quanta's average of e^{aQ} | `test_n_quantas_average_of_an_exponential_is_laguerres` |
| the motion's words and refusals | `test_a_motion_says_what_it_takes_and_refuses_by_name`, `test_the_refusals_the_contract_states_and_one_set_pooled` |
| order, one vibration's states | `test_a_motion_keeps_the_results_order_and_one_vibrations_states` |
| geometry and scale | `test_a_vibrations_geometry_and_its_frequencies_scaled` |
| what a set says, pooling read twice | `test_what_a_set_says_and_what_pools` |
| the read path and the structure, end to end | `tests/test_vibration_mechanics_e2e.py::test_h2s_bond_moves_by_the_textbooks_zero_point_amplitude` |
