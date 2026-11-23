# F12 fix 19 (the vibration framework's module, its states and the script door, 2.78.0, R3d2a) — the review

**Verdict: closed with 2.78.0.** The first of the four releases that build the vibration framework the
user decided on 2026-11-22 (`engines/vibration.md` § 6.11, its § 9; the theory
`science/vibrational-averaging.md` § 2b): the module `vibration`, a mode's states and their frames,
and the script door `postprocess.vibration`.

The lessons: a shared reading made one home asks only for what its caller uses — round 1's one
reading of each atom's element resolved every atom's label, so a held atom's label the table cannot
read dropped the load door's shares, which the door had served; a new door's names are the record's
from its first release, one name per quantity, before a person's script keys on them; a function that
refuses each wrong combination by its own branch is the code enumerating wrongness (`testing.md` § 3b)
— the kinds declared once as data give one refusal and one rule for the record's temperature; a
number state at large n is built from the oscillator's own operator, never from its moments or a
recurrence that underflows; a floor the contract does not state (a classical mode at 0 K) is the code
deciding, not the person; a new door a person can call is a middle-number move.

## What was built

- **The module `vibration`** (L2, numpy and the constants; in `mb_pyscf.pyz` and `mb_vibration.pyz`):
  the mechanics (`checked_temperature_k`, `zero_point_amplitude_amu12_ang`, `mean_quanta` over an
  array too, `spread_of_quanta`, `thermal_spread_amu12_ang`, `classical_spread_amu12_ang`,
  `thermal_spread_of`); the mode geometry (`mode_over_atoms`, `largest_move`, `q_for_largest_move`,
  `atom_moves`, `displaced`, `coordinate_of`); `element_shares`; the states of § 2b.3 (`state`,
  `State`: n quanta, thermal or random quanta, a swing over a bath, classical, a classical swing —
  their density, distribution, moments, spread, mean quanta; the kinds one table, `_STATES`) and
  their frames (`Frames`: Gauss–Hermite; a number state's own Gauss rule by Lanczos on the position
  operator; a swing's equal phases folded, times its bath's rule); `Mode`, `ModeState`, `Vibration`.
- **The callers on it**: `spectra/derived.py` (the derived values a result states),
  `normal_modes.vibrational_thermo` (the occupation), `mode_frames`, `frame_average`,
  `siesta_vibration` (its bundle), the load door's shares (`SpectraResults.atoms_weighted` for its
  free atoms, `vibration.element_shares`).
- **The script door**: `postprocess.vibration(path)` — `sidecars.spectra.read_vibration_result` (a
  result from either file by its role), `SpectraResults.vibration` (each mode a `vibration.Mode`, the
  record's fields by their names, the record's temperature).

## Round 1 — four readers (correctness A, correctness B, one home, notes and tests): 33 findings

| # | finding (readers) | verdict |
|---|---|---|
| 1 | A swing's bath named by nothing: the module's `state()` took 0 K, the person's door the record's temperature; § 6.11 § 6 said 0 K (cA3, oH1, nt2, cB3) | **fixed** — `state()` refuses a quantum swing with no bath, by name; `Mode.state` takes the record's temperature and says so; § 6.11 § 6 says it |
| 2 | A temperature or mean quanta beside a classical swing silently dropped, yet said "given" (cA1, nt3) | **fixed** — refused by name; `state()`'s docstring lists the five kinds |
| 3 | A number state broke from ~248 quanta: `cdf(0)` 1.0 by `min(1, nan)`, `pdf` nan, `frames` and `moment` failing (cA2) | **fixed** — `moment` by the oscillator's position operator, `frames` by Lanczos on it from `|n>`, the density by the scaled recurrence (the Gaussian applied once, in logs), the distribution in panels fine for `n` zeros, every panel's points asked at once; a row at 1000 quanta |
| 4 | numpy integers refused as "not whole" (cA4) | **fixed** — `_whole`, any `numbers.Integral`; a row |
| 5 | A classical state at 0 K: nan / ZeroDivisionError; a classical swing of 0 quanta: `cdf(0)` 0 (cA5) | **fixed** — classical at 0 K refused by name; the classical swing's distribution asks the upper end first |
| 6 | `_SWING_PHASES = 256` "for any bath" overstated: wrong at thousands of swing quanta (cA6, nt10) | **fixed** — the phases scale with the swing against its bath's spread (`_swing_phases`, 256 at least) |
| 7 | Science § 2b.5 claims the eigenfunctions' orthogonality is tested (cA7, nt9) | **fixed** — the row states what is tested (the turning-point weights) |
| 8 | `Mode.element_shares` over every atom — a held-only element at 0.0, against § 6.11 § 3 "over each element's free atoms"; `zero_point_displacement_ang` over every atom under the record's name (oH2, cB1) | **fixed** — `Mode.free_atoms`; the shares over them; the zero-point field the record's shape; a row |
| 9 | Each atom's element and mass resolved in the load door and in `SpectraResults.vibration` (oH3), and the label row testing its own helper (nt1) | **fixed** — `SpectraResults.atoms_weighted()`, asked by both (a label the table cannot read still leaves the result loading); the label row moved to the load door's e2e (`real_water_linear`, H1/H2) |
| 10 | Docs not updated: `parse.md` § 3, `postprocess` header, § 8.1's rows, § 6.11 § 1's spread row, `vibrational-optical-response.md`, `pyscf.md`, `plan.md` (oH4, nt6, nt7, cB5) | **fixed** |
| 11 | Notes untrue: `normal_modes` header, the `results.py` section heading, `core.js`'s `spectra.derived` (oH5, cB5) | **fixed** |
| 12 | `temperature_of_quanta` exported, uncalled, untested (oH6, nt4) | **fixed** — removed |
| 13 | The spread spelled by coth and by `2n+1`; the quanta check three times (oH7) | **fixed** — `thermal_spread_amu12_ang` is `spread_of_quanta` at the temperature's quanta; one `_checked_quanta` |
| 14 | `vibrational_thermo` spells ħω/k_BT a second way (oH8) | **fixed** — `nu * CM1_KELVIN / T`, the module's constant |
| 15 | Two behaviours no test sees break — a frame's geometry, a swing's bath weights; § 6.11 § 7 over-claims (nt4) | **fixed** — rows: a frame's every atom at `R + Q_j L`; a swing over a bath exact to its degree |
| 16 | Two rows took an expectation from molbuilder's output: the number state's frames against `s.pdf`, the spring's energy against `eig[0]` (nt8) | **fixed** — the closed-form density (`scipy.special.eval_hermite`); `k / mu` |
| 17 | The post-processing guide says every call is the e2e's (nt5) | **fixed** — says which test makes which |
| 18 | `test_motion_share.py` leftovers: the mapping clause; refusal rows with no rule (nt11) | **fixed** |
| 19 | `real_h2_vibration` a session fixture one module reads; H2 typed again (nt12) | **fixed** — the module's own fixture; built on `support.road.H2`, its held atom dropped |
| 20 | `(§ 4.9)` in a message a script shows (nt13) | **fixed** |
| 21 | The load door's shares not 2.77.26's to the last bit — the total summed per element (cB2) | **fixed** — summed per atom, as the function always did |
| 22 | A mode at exactly 0.0 cm⁻¹ makes the thermochemistry NaN (cB4) | **not a defect** — an exactly-zero eigenvalue no engine's force constants give (SIESTA's nine digits), as the reader says |
| 23 | Copies the plan defers: the maker's `_ordered_nodes` and inline largest move, `frame_average`'s swing moments, the page's `onWriteFrameSet` and `_swingReport` (oH note) | **later** — named in the plan under R3d2b |

Tally: 33 reports (cA 7, oH 8, nt 13, cB 5) in 23 groups — fixed 21, not a defect 1, later 1.

## Round 2 — three new readers (correctness, one home, notes and tests): 19 findings

| # | finding (readers) | verdict |
|---|---|---|
| 1 | `_swing_phases`' note says the swing moves "less than a sixteenth of its bath's spread" between two phases — it moves up to 2π/16 ≈ 0.39 of it; round 1's #6 has no row (a fixed 256 passes every row, 96 % off at 10⁴ swing quanta) (cA1, nt1) | **fixed** — the note says sixteen phases per bath spread of amplitude, at most 2π/16 between two; a row at 10⁴ swing quanta over a 0 K bath against the phase integral (the turning phase its break point), failing on a fixed 256 |
| 2 | `cdf` held to no value for four of five kinds — the Gaussian's √2, the swing's sign, the classical swing's rate, round 1's #5 all pass (nt2) | **fixed** — one table of eight rows from the closed forms (the Gaussian at ±σ and −2σ, the classical Gaussian, the classical swing at ±Q₀/2 and of no quanta, a swing over a bath by the phase-averaged distribution); each of the four mutants fails its rows |
| 3 | The record's temperature through the script door never read from a real result (nt3) | **fixed** — the e2e row asserts `v.temperature_k` the JSON's `thermo.temperature_K` from either file, `v.source`, `m.state()` the record's and its spread `Q_zp sqrt(coth)` from CODATA |
| 4 | `exact_to_degree` unchecked for a swing over a bath and for n quanta; `nodes=1` unsaid (nt4) | **fixed** — both asserted from the science (2(m−1)−1; 2m−1); the docstring says one node |
| 5 | The free-atom zero-point row checked by shape only; `largest_zero_point_move_ang` untested (nt5) | **fixed** — by value (`Q_zp` from CODATA times the free rows); the H₂ row asserts half the bond |
| 6 | Three of `state()`'s refusals unpinned; one id wrong (nt6) | **fixed** — rows for nothing named, a temperature with mean quanta, classical with no temperature; the id corrected |
| 7 | `atoms_weighted()` resolves every atom: a held atom's label the table does not know drops the load door's shares, which 2.77.26 served (oH1) | **fixed** — `atoms_weighted(atoms)` resolves the atoms asked, the door its free atoms; the door's shares 2.77.26's on that input and the five others (`compare_head`). The script door on such a label refuses by the resolver's message: only a hand-edited record carries one (prep refuses the label) — not on the road |
| 8 | A state's spread spelled twice (`spread_of_quanta`, `sqrt(moment(2))`), the Gaussian's density and distribution twice, a row's length twice (oH2) | **fixed** — a quantum state's spread is `spread_of_quanta`'s; `_gaussian_density`, `_gaussian_distribution`, `_row_lengths` |
| 9 | The record's temperature read by `SpectraResults.vibration` and the door's frame-set fallback, § 9 item 6's rule in two places (oH3) | **later** — named in the plan's R3d2b row (the frame-set maker folds into the module there) |
| 10 | The free atoms' shares composed by the door and by `Mode.element_shares` (oH4) | **later** — named in R3d2b (the record stores the shares; the door reads them) |
| 11 | Docs: `web/spectra.md` § 4.2 and § 8.1's `results.py` row name `mass_of`, not `atoms_weighted`; § 6.11 § 1's largest-move row names only the page (oH5, nt7) | **fixed** |
| 12 | The guide says a state given no temperature takes the record's — untrue for quanta, mean quanta and a classical swing alone (nt8) | **fixed** — the docstring's words |
| 13 | The guide's opening says every example ran on a real calculation; § 2a says most did not (nt9) | **fixed** — "save § 2a's states and frames" |
| 14 | `Mode.element_shares`' note: "a held atom's element has none" — false for gold partly held (nt10) | **fixed** — "a held atom adds nothing to its element's share" |
| 15 | `real_runs`' header lists `h2_vibration` among conftest's session fixtures, "on the road's water" (nt11) | **fixed** — its own line, a module's own run, a free H₂ |
| 16 | A `test_motion_share` refusal row cites § 6.6, which states no such rule (nt12) | **fixed** — § 6.3's `(n_free, 3)` rows |
| 17 | `postprocess.reaveraged`'s motion and weights differ in their last digit too — not in the visible-change list (cA2) | **fixed** — in the changelog row's visible changes |

Not counted: `tests/spectra/test_derived.py` now tests the module and the maker, not `spectra.derived`
(nt observation) — **later**, renamed in R3d2b with the maker's fold.

Tally: 19 reports (cA 2, oH 5, nt 12) in 17 groups — fixed 15, later 2. Each new row seen failing on
its mutant (13 mutants, scratch copy); the targeted set 129 passed.

## Round 3 — two full-text readers (A, B) and the discipline reader: 19 findings

| # | finding (readers) | verdict |
|---|---|---|
| 1 | The version moves the last number for a release whose headline is a new door a person can call (conventions § 2: middle = new functionality; precedents 2.66.0 `read_steps`, 2.52.0 `reaveraged`) (d1) | **fixed** — 2.78.0 (`__init__`, the plan) |
| 2 | The frames' coordinate under three names (`Frames.positions_amu12_ang`, the dict's `coordinates_amu12_ang`, the record's `nodes_amu12_ang`), a mode's frames a raw dict whose `positions_ang` meant geometries; `Mode`'s fields not the record's (`rows`, `free_atoms`, `elements` holding resolved symbols) (d2, fA4) | **fixed** — `Frames.nodes_amu12_ang` (§ 6.11 § 5's name), `Frames.positions_ang` each frame's geometry where the frames are a mode's (`ModeState.frames` returns `Frames`); `Mode.eigenvector_canonical`, `free_atom_idxs`, `element_symbols`; the guide and § 6.11 § 6 say them |
| 3 | `state()` enumerates every wrong combination — seven refusals, eight wrong-example rows; which states take a temperature decided again in `Mode.state` (testing § 3b: "a test needing many wrong examples means the code enumerates wrongness … the code's defect, fixed in the code") (d3) | **fixed** — the kinds one table (`_STATES`, the sets of arguments naming each); one refusal naming the ways; `Mode.state` takes the record's temperature for a set that names a state once a temperature is added, read off the table; the test one example |
| 4 | A classical state at 0 K refused — a floor of ours (product stance: "an error only when a setting leaves NOTHING to compute"); the contract written to the code (d4) | **fixed** — at rest at Q = 0: spread 0, its distribution a step there, its density a point, its frames at 0; the classical swing of no quanta the same point; two distribution rows; § 6.11 § 6 says it |
| 5 | Tombstones and release tags in § 6.11 ("what `spectra/derived.py` held", "Built so far", "(2.77.27)", "folded in R3d2b") (d5) | **fixed** — § 6.11 in the present tense: what the module holds, what is still to come, the schedule plan § 2's alone |
| 6 | Science § 2b.4 says the n-quanta rule is computed from its moments; the code is Lanczos on the position operator (fA1, fB5, d6) | **fixed** |
| 7 | The guide's opening still claims § 2a's first block ran on a real calculation (fB3, d7a) | **fixed** — "save § 2a's" |
| 8 | "Never from molbuilder's own output" in the test header, science § 2b.5, § 6.11 § 7, beside relational rows (d7b) | **fixed** — never from its constants; a row holding two of its answers holds them to the contract's relation |
| 9 | A classical state given no temperature taking the record's is untested (d7c) | **fixed** — a row (the record's 500 K, said so, the classical spread from CODATA); fails on its mutant |
| 10 | `test_a_whole_number_of_any_integer_type_is_taken` names no failure or contract; its failure is a loud refusal (testing § 3b) (d8) | **retired** |
| 11 | `Mode.coordinate` on moves of other atoms fails with numpy's reshape error, not by name (fB1) | **fixed** — `coordinate_of` refuses moves of other atoms by name |
| 12 | The thermochemistry's change understated for modes softer than a few cm⁻¹ (13th digit at 0.5 cm⁻¹, 4 % at 10⁻⁹) (fB2) | **fixed** — the changelog row says it |
| 13 | One rule — a finite number of 0 or more, never a boolean — spelled in `checked_temperature_k` and `_checked_quanta` (fB4) | **fixed** — `_finite_non_negative` |
| 14 | § 8.1's `vibration` row omits its bundles; § 5.5's layer table omits `vibration` (fB6) | **fixed** |
| 15 | § 6.11's opening cites plan § 3 (R3e) for R3d2's § 2 (fA2) | **fixed** |
| 16 | § 6.11 § 2 says `spectra/derived.py` states its values "each by the module" — the activity classes are `spectra.activity`'s (fA3) | **fixed** — "their mechanics the module's" |
| 17 | The frame rule's default three typed in `vibration`'s two `frames` beside `mode_frames.NODES` (d9) | **later** — named in R3d2b, one constant the module's |

Tally: 19 reports (fA 4, fB 6, d 9) in 17 groups — fixed 15, retired 1, later 1. Targeted set 124
passed; the three new rules (the classical record temperature, at rest at 0 K, the one refusal) each
fail their row on a mutant.

## The reader of the last fixes: 6 findings

| # | finding | verdict |
|---|---|---|
| 1 | `coordinate_of` checks the moves' count, not their layout: moves given as x, y, z rows over the atoms (`u.T`) pass and give a wrong Q silently (0.0717 for 0.1316) | **fixed** — refused unless a row per atom, every atom's, the shape named; a row (`u.T` on H₂), failing on the count check |
| 2 | `Mode.eigenvector_canonical` named the record's field but held every atom's rows: `Q_zp L ≠ zero_point_displacement_ang` within one mode | **fixed** — the field the record's, a row per free atom; the pattern over every atom its own property, `eigenvector_canonical_over_atoms`; the free-atom row holds `Q_zp` times the record's rows and the held atom still |
| 3 | The density at rest untested (a density 0 everywhere, or NaN, passes) | **fixed** — a row for both states at rest (the classical mode at 0 K, a classical swing of no quanta): infinite at 0, nothing elsewhere, no spread, every frame at 0; failing on both mutants |
| 4 | The record's temperature under two names (`Vibration.temperature_k`, `Mode.record_temperature_k`), `temperature_k` also the state's | **fixed** — `Vibration.record_temperature_k` |
| 5 | § 8.1's `vibration` row names one asker of eight; § 5.5's row says "every quantity" | **fixed** |
| 6 | "The relation the science states" where the relation is the contract's (§ 9 item 6) | **fixed** — "the contract" in science § 2b.5 and § 6.11 § 7 |

Notes taken: the record-temperature test named for the classical case too; `Frames.positions_ang` not
compared (derived from the nodes and the mode), so two frame sets compare by their rule. Targeted set
127 passed; the three new rows fail on their mutants.

## The reader of the very last fixes: 3 findings

| # | finding | verdict |
|---|---|---|
| 1 | On a mode with a held atom nothing tests `atom_rms_ang` or the moved geometry (`displaced`, each frame's `positions_ang`) — either pointed back at the free rows passes every row | **fixed** — one row: two free hydrogens beside a held gold, the gold's r.m.s. 0 and each hydrogen's `σ|L_H|`, every frame and a displaced geometry with the gold at `R_Au`; failing on both slips |
| 2 | § 6.11 § 7's sentence after round 4's #6 reads the closed forms as the contract's relation; the hot limit and Mehler's sum are the science's | **fixed** — one wording in § 6.11 § 7, science § 2b.5 and the test header: the relation the science or the contract states (Mehler's sum and the hot limit, § 2b.3; the record's temperature, § 9 item 6) |
| 3 | `Mode.eigenvector_canonical` carries no unit beside § 6.11 § 6's "its unit in its name" | **fixed** — the data file's names, `eigenvector_canonical_per_amu12` and `eigenvector_canonical_over_atoms_per_amu12` |

Targeted set 127 passed.

## End to end

359 passed (`tools/testrun.py run e2e` on the seven files this change touches, 35 min 57 s): the script door on a real free PySCF H₂ from both of its files, the spectrum results and the spectra page on PySCF and SIESTA, the PySCF vibrations, a real junction's frame set and its re-average, the page with no spectrum, the vibration view. The page-load check of `lib/spectra/core.js` (two comments changed) is the SIESTA vibration's and the spectrum module's page rows.

## Topic → test

| change | test | seen failing |
|---|---|---|
| the zero-point amplitude, the bond move, a spring between two masses (§ 2b.1, § 2b.2) | `test_vibration_mechanics::test_h2s_zero_point_amplitude_…`, `test_a_spring_between_two_masses_…` | at import on 2.77.26 (the module new) |
| n quanta exactly, at any n (§ 2b.3; round 1 #3) | `test_n_quanta_exactly_…`, `test_a_thousand_quanta_…` | the rescaling and the panels undone, in a scratch copy |
| thermal is Mehler's sum; Bose–Einstein; the hot limit | `test_thermal_is_mehlers_sum_…`, `test_a_temperatures_quanta_…`, `test_hot_the_thermal_spread_…` | at import on 2.77.26 |
| a swing over a bath, at three and at ten thousand quanta (round 2 #1) | `test_a_swing_over_a_bath_is_the_bath_carried_along_the_swing` | a fixed 256 phases (96 % off at 10⁴) |
| each state's distribution (round 2 #2); at rest at 0 K (round 3 #4) | `test_each_states_distribution_is_its_closed_form`, `test_a_classical_mode_at_rest_sits_at_zero` | four distribution mutants; the 0 K refusal; a density 0 or NaN at rest |
| each rule's frames to its stated degree (round 2 #4) | the three frame rows, `test_a_swing_over_a_bath_is_averaged_exactly_…` | the stated degree 1 or 2m |
| one refusal for a state named no way of its kinds (round 3 #3) | `test_a_state_named_no_way_of_its_kinds_is_refused_naming_them` | a set beyond the table taken |
| every temperature explicit (§ 9 item 6) | `test_a_temperature_not_given_is_the_records_…`, `test_a_swings_bath_and_a_classical_state_take_the_records_temperature`; the e2e row | the classical case skipped; `"given"` said |
| a held atom still; the free atoms' shares and zero-point field (round 1 #8; the last reader #1) | `test_a_held_atom_stays_still_and_adds_nothing_to_its_elements_share` | the first rows taken; the free rows for every atom |
| moves laid out other than a row per atom refused (round 4 #1) | `test_moves_laid_out_other_than_a_row_per_atom_are_refused` | the count check |
| the script door from either file, against CODATA | `test_vibration_mechanics_e2e` | at import on 2.77.26 |
| the load door's shares 2.77.26's (round 1 #21, round 2 #7) | `test_spectrum_results_e2e::test_each_elements_share_…[water, held, linear]`; `compare_head.py` | not seeable — the same answer is the point |
| the swing's spread by its mean quanta; a numpy integer as a whole number; `Frames.positions_ang` left out of a comparison | — | untested |
