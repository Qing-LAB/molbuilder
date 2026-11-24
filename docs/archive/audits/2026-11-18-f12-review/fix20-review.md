# F12 fix 20 (the vibration record: each mode's mechanics stated, schema 15, data file @11, 2.79.0, R3d2b1) — the review

**Verdict: closed with 2.79.0.** The second of the releases that build the vibration framework the user
decided on 2026-11-22 and 2026-11-23 (`engines/vibration.md` § 6.11; `engines/vibration-frames.md`): the
vibration calculation is the foundation, and its record states every mode's data — each free atom's share of
the motion, the largest zero-point move, Gaussian's reduced mass and force constant, and each mode's spread and
mean quanta at the modes' temperatures.

The lessons: a list a file is checked against must be the file's own — a check that rebuilds a display grid
from today's constants turns the next change to that grid into every file on disk refused; a read check's
tolerance must stand above what the constants' last digits move through the quantity (a mean quanta in the cold
multiplies a constant's error by up to 745), and below anything physical; a subnormal number's last bit is
no longer small, so a check needs an absolute floor where `exp` is not rounded alike on every machine; one
composition per quantity — the record's writer and the person's `Mode` call the same function, and a mode with
no motion gets one answer; machinery for a payload no person writes is the code designing for a state off the
road (product stance), however reasonable a reviewer's scenario; no test builds an older release's file, and a
refusal is shown by one example of its one rule.

## What was built

- **`vibration`**: `motion_shares_of` (each atom's `m|L|²` over the sum), `shares_by` (summed by each atom's
  group or groups — element, label, region — largest first), `reduced_mass_amu` (Gaussian's `1/Σ|L|²`, `None`
  for no motion), `force_constant_mdyn_per_ang` (`μω²`, an imaginary mode's positive),
  `zero_point_displacement_ang`, `largest_zero_point_move_ang`, `THERMAL_ROW_KEYS`, `thermal_row`;
  `mean_quanta` refuses a frequency that is not positive; `Mode` gains `species_labels`, `regions`,
  `mode_temperatures_k` and `motion_share`, `motion_shares(by=…)`, `reduced_mass_amu`,
  `force_constant_mdyn_per_ang`, `thermal`. `element_shares` (function and property) deleted.
- **`constants.FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2`**, derived from its siblings; `ncdata`'s unit
  `mdyn/angstrom`.
- **`spectra.normal_modes`**: `MODE_TEMPERATURES_BELOW_GRID_K` (0, 4, 10, 20, 30, 40 K; the user's choice),
  `mode_temperatures(temperature_K, stated_k=None)` — the modes' temperatures' one door.
- **`spectra.derived`**: the v15 keys, `thermal_block`, `stated_temperatures`, `free_masses_of`,
  `payload_with_derived` (the one writer derives at the payload's level, every engine alike),
  `disagreements(payload, version, masses)` — each row of `modes_thermal` checked at the file's own
  temperatures, the tolerance a relative 10⁻⁶ with a floor of the smallest normal double, a refusal naming the
  mode, the temperature and the first differing element.
- **`spectra.results`**: schema 15; `mode_temperatures_k` (the file's list, kept); `_free_masses` through
  `atoms_weighted`; `modes_thermal` stated from the first write (`null` before the modes exist);
  `atoms_weighted` by `Structure.chemical_symbols`; `vibration()` hands each `Mode` its labels, regions and
  the modes' temperatures.
- **`spectra.datafile`**: @11 — `/modes`' four variables, `/modes/thermal` always written (its own
  `temperature_k` axis; an imaginary row read back `None` by `has_imag`), refusing a block key it has no
  variable for.
- **The load door**: the element shares the per-atom shares summed by the label's name, asked only where a mode
  states shares.

## Round 1 — five readers (correctness, one home, notes and tests, science, API): 28 findings

Decisions E1–E11 (`r3d2b1-round1-decisions.md`, kept with this record's working files):

| decision | findings | verdict |
|---|---|---|
| E1 the floor for a subnormal mean quanta | correctness 1, API 6, science 1 | **fixed** (the tolerance revised in H1) — a real lab result (`bdt` mode 13, 2026.56 cm⁻¹) stores 2.66·10⁻³¹⁷ at 4 K; one ulp moved refused the file |
| E2 the modes' temperatures the file's own | API 1, one home 1, 8, 9 | **fixed** — the check rebuilt the list from today's constants; a 59-point grid in memory refused every v15 file |
| E3 `modes_thermal` null from the first write | API 3 | **fixed** |
| E4 a file before v15, a disagreeing file, tested | notes 1 | **fixed**, then retired by J2/J3 (the user's rules on older files and one example) |
| E5 the record's numbers held to the science | notes 5, 6, 10 | **fixed** — the largest move, the mean quanta, 0 K, `m.thermal`, shares by label, SIESTA's shares by `m|L|²` |
| E6 one composition per mode | one home 4 | **fixed** (completed by H7) |
| E7 the masses one reading; a pre-v7 unknown label loads | one home 5, API 2, correctness (dropped ×2) | **fixed** — measured: such a v6 file loads without shares, as 2.78.0 |
| E8 the block's keys one table | one home 6 | **fixed** |
| E9 documents and notes | API 4, one home 2, 3, 7, 8, 9, notes 2, 3, 4, 7, 8, 9 | **fixed** |
| E10 a hand-built `Mode` without labels | correctness (dropped) | fixed, then **reversed** by J8 (an invented refusal) |
| E11 visible changes | one home 2, API 5 | **listed** |
| — `vibrational_thermo` at exactly 0.0 cm⁻¹ | correctness | **dropped** — 2.78.0 wrote NaN the writer refused |

## Round 2 — five new readers: 26 findings

Decisions H1–H10 (`r3d2b1-round2-decisions.md`):

| decision | findings | verdict |
|---|---|---|
| H1 the tolerance a relative 10⁻⁶ | science 1 | **fixed** — with k_B exact or CODATA 2022, 17 of 17 v15 files were refused under 10⁻⁹ (a mean quanta's error × x up to 745; the force constant 1.35·10⁻⁹); put to the user with the commit |
| H2 the floor said as the code does it | science 2, notes 4, correctness 2 | **fixed** |
| H3 the modes' temperatures one name and one door | one home 6, 8, API 2, correctness 3 | **fixed** (its form checks cut by J1) |
| H4 the writer refuses a block it cannot state | API 1, notes 5 | fixed, then **cut** by J1 |
| H5 refusals name the place | API 3 | **fixed** |
| H6 shares by region the same shape | API 4, one home 1 | **fixed** |
| H7 the zero-point displacement composed once | one home 5 | **fixed** |
| H8 tests | notes 1, 2, 3, correctness 1, API 1, 2 | **fixed** (partly retired by J1–J3) |
| H9 documents | one home 2, 3, 4, 7, API 5, 6, notes 6, 7 | **fixed** |
| H10 the per-mode copies go to the `Vibration` in R3d2b2 | API 6 | **planned** (plan, `vibration-frames.md` § 9) |
| — `to_dict` keeps an older file's version; a v4–v6 result written again refused | notes, science, correctness | **dropped** — no road writes an older file again |

## The full-text readers and the discipline reader: 29 findings

Decisions J1–J18 (`r3d2b1-round3-decisions.md`):

| decision | findings | verdict |
|---|---|---|
| J1 no machinery for a person's own temperature list | discipline 1 | **fixed** — no document offers a person a door that writes a result payload; the form checks, the writer's refusal, their rows and sentences gone; the reader's refusal of an unknown block key stays |
| J2 no test builds an older release's file | discipline 2 | **retired** — measured instead: real 2.78.0 v14 results read on this tree with the v15 values computed, the load door within 10⁻¹² of 2.78.0's (correctness r1, r2, API r1, discipline, full-text A) |
| J3 one rule, one example | discipline 3 | **fixed** — the moved share kept; the one-ulp case kept (another machine's `exp`, which no run here causes) |
| J4 a test whose premise is gone | discipline 4 | **retired** — `test_either_stored_eigenvector_gives_the_same_answer` |
| J5–J18 notes, documents, one home | discipline 5–10, 12–14, full-text A 1–6, full-text B 1–8 | **fixed** |
| — refusal words name keys; release steps; a timing note | discipline 11, 15; full-text A | **dropped** — a foreign file alone reaches the words; the steps done at release; 300 modes read and loaded 0.24 s → 0.37 s, recorded |

## The reader of the last fixes: 7 findings, all fixed

Four notes still saying a list is checked (J1); the one-ulp e2e case at x = 725 never reaching the floor —
moved to x = 740, where one ulp is a percent of the value and the file is refused without the floor
(measured by the reader on a real H₂ result); "the stored per-atom shares" in four places; § 6.11 § 2's "one
name" overclaiming (the eigenvector's two names); a docstring's "two-body 0.504"; `web/spectra.md`'s "either
stored eigenvector"; the kept hand-edited case's named need.

## The e2e run

The vibration road end to end, once, last: 366 passed — the H₂ record and its rules, the spectrum results and the spectra page on PySCF and SIESTA, the PySCF and SIESTA vibrations, a real junction's frame set and its re-average, the page with no spectrum, the vibration view, the Results export (37 min 40 s); 0 failed. The whole fast suite before it: 6551 passed, 0 failed.

## Process

- Round 1's notes reader ran `rm -rf` on a copied folder inside its own scratch — against the never-delete rule;
  nothing outside scratch touched. Later briefs said "never delete a file, not even in scratch".
- The round 1 brief called `h2a/H2` a PySCF H₂; it is a SIESTA run (the science reader). Corrected in round 2.
- Five readers of round 1 stopped at a usage limit and were resumed where they stopped.

## Topic → test

| topic | test |
|---|---|
| Gaussian's reduced mass and force constant; an imaginary mode's positive; no motion `None` | `tests/test_vibration_mechanics.py::test_the_reduced_mass_and_force_constant_are_as_gaussian_prints_them` |
| the largest zero-point move on any axis | `…::test_the_largest_zero_point_move_is_an_atoms_whole_move_on_any_axis` |
| two-body shares and Gaussian's μ | `…::test_a_spring_between_two_masses_moves_as_the_two_body_problem_says` |
| shares by element, label, region | `…::test_the_shares_sum_by_element_label_and_region` |
| the modes' temperatures, a thermal row, an imaginary mode's none | `…::test_a_modes_thermal_row_is_its_spread_at_each_temperature` |
| the record's stored values against CODATA, both files; `m.thermal` the record's row | `tests/test_vibration_mechanics_e2e.py::test_h2s_bond_moves_by_the_textbooks_zero_point_amplitude` |
| a contradicted file refused, by name | `…::test_a_v15_file_whose_stored_values_its_inputs_contradict_is_refused` |
| the file's own temperatures; the floor | `…::test_the_modes_temperatures_are_the_files_own` |
| shares by label on a real labelled water | `tests/test_spectrum_results_e2e.py::test_each_elements_share_is_its_part_of_the_mass_weighted_motion` |
| SIESTA states the v15 keys and the block | `tests/test_siesta_vibration_e2e.py` (the result's derived values) |

**Untested**: the tolerance's value (no row fails if 10⁻⁶ went back to 10⁻⁹ — the constants are not moved in a
test); numpy's AVX-512 `exp` (no such CPU here); a pre-v7 result whose label the mass table does not know
through the load door (measured by hand; no test builds an older release's file); the reader's refusal of a key
outside the block (one example of the one rule is the share).
