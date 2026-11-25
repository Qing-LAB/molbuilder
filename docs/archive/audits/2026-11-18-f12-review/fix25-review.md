# F12 fix 25 (the motion toolkit's doors for any state, 2.83.0, R3d2b2b2b2) — the review

**Verdict: closed with 2.83.0.** The motion toolkit's fifth release, the second half of R3d2b2b2b (the user's decision,
`engines/vibration.md` § 6.11 § 9 item 2: the doors write a mode's own set in any state): `molbuilder frameset
from-mode` and the spectrum tab's *Write frame set…* ask the mode's state every time — thermal, n quanta, mean quanta,
driven, warm driven — and say what it means before the set is written; `spectra/mode_frames.py` retired into
`vibration_frames`. The example notebook follows in R3d2b2c.

The lessons: a number a person may mistype is the one to cost (round 1: 10000 for 10 in the dialog built a gigabyte
operator a mode while they typed — the rule taken on the few states it reaches); a value no person produces is no
finding (the user, round 2: "why the fuck do we need to consider 2^64 quanta?"); a warning's words are science and
need the science reader like any other claim (round 3: "about a chemical bond's energy" — a covalent bond is 3–5 eV,
and a torsion leaves the harmonic picture far below; the user kept 1 eV and had the words made true); a dialog whose
note follows its fields must not move what is under the pointer, by its top or by its note shrinking.

## What was built

`vibration_frames` (the host's), the doors' set: `frameset_from_mode(run_folder, *, mode, out, nodes, temperature_k,
quanta, swing_quanta, mean_quanta)` → `ModeFrames` (moved from `spectra.mode_frames`); `doors_state` (the state asked
every time: one set of the state table's, `vibration.STATES`, each value by `vibration.checked_state`, before any run is
read; `StateRefused`); `read_state_ask` and `STATE_ASK_KEYS` (the page's ask, `persist.unknown_keys`' words);
`readout` (what a state means for a mode before its set is written, the load door's `frameset` block: the state in the
set's words, its mean quanta, spread, outermost move for each offered frame count, a driven state's turning point, the
coherent state's thermal quanta at the record's temperature, the many-quanta words); `OFFERED_NODES`;
`HARMONIC_LIMIT_EV` (the user's 1 eV). `vibration`: `DEFAULT_NODES`, `STATES` (was `_STATES`), `checked_state`,
`State.is_coherent`, the position operator on a window (`_position_operator(lo, hi)`). The load door asks the readout
at `frameset_state`; the write door takes every state's keys and refuses a key it does not take; the command line's
flags for each state, a state refused exit 2. The page: the dialog's state table (`_DIALOG_STATES`), the form dialog's
tick fields and fields shown by a choice (`askFields`' `show`), `.mb-dialog-follows` (anchored at its top, the tree
picker's too), a note that never shrinks while open, the many-quanta confirm (`warningModal.confirm`).

## Round 1 — five readers (correctness 4, one home 9, notes and tests 11, API 4, pages 5): 33

Decisions Y1–Y22 (`r3d2b2b2b2-round1-decisions.md`). **Page breaks once a large number is typed**: the n-quanta rule's
dense operator (correctness 1, pages F1 — n = 10⁴ about 1 GB a mode, a 500 at 10⁶) — the rule and the moments taken on
the window they reach, exact (Y1); the dialog's last state riding on every later load (pages F2, Y2). **A
notebook/command-line break**: a state named twice or a bad value refused only after the run is read, in the script's
names — checked first, by the table's own rule (Y4), a usage error at the command line (Y5). **A silent wrong set**: a
misspelled state key at the write door ignored and the cold coherent state written — refused, naming the key it is
close to (Y3). Also: help true for every state (Y6); the readout asks the doors' rule (Y7); frames named from 1 (Y8);
the doors' numbers the readout's and the set's own rows (Y9); verdicts on two plan items (Y10); one table for the
dialog's fields (Y11); the record's largest zero-point move named (Y12); the coherent state one property (Y13); tests
(Y16); notes (Y17–Y20); the dialog anchored at its top (Y21). Dropped: a result before schema 7 whose label the mass
table lacks (Y15, unverified on a real file).

## Round 2 — five new readers (correctness 1, one home 8, notes and tests, API 4, science 1)

Decisions Z0–Z10 (`r3d2b2b2b2-round2-decisions.md`). **Z0, the user (2026-11-24)**: a state no person produces is no
finding — the 2^64-quanta findings and round 1's million- and billion-quanta test row dropped; many quanta warned of,
never refused — beyond 1 eV in the mode, ⟨n⟩ h c ν̃, said by the readout and the command line and asked to confirm on
the page. Also: values refused in plain words (Z1); `readout`'s `node_counts` (Z2); the new names in the documents
(Z3); the dialog's ask beside the load's own (Z4); the states one table (Z5); a frame spelled one way (Z6);
`ModeFrames.mean_quanta` a number (Z7); the tree picker's anchoring one selector (Z8); notes (Z9); a warm tick at 0 K
the coherent state, said (Z10).

## The full-text readers and the discipline reader (fulltext-A 6, fulltext-B 5, discipline 9): 20

Decisions F0–F14 (`r3d2b2b2b2-round3-decisions.md`). **F0, the user (2026-11-24): keep 1 eV, the words made true** —
round 2's "about a chemical bond's energy" was wrong (a covalent single bond is 3–5 eV; a torsion leaves the harmonic
picture far below): the warning says the mode is far up its ladder, where the harmonic picture may no longer hold.
Also: the dialog's note opens with the state in the set's words, cold and warm the server's, counts "1 quantum" (F1);
the template's note (F2); the shown fields from the one table (F3); the optical-response chapter's inputs a state, not a
temperature alone (F4); the test map (F5); the note never shrinking, the top and *Next* held still, the tree picker's
top held still in the save-where helper (F6); the science note sourced or qualitative, the warning's words everywhere
(F7); no readout where the result states no geometry, asked of the result, the e2e's expectation from the row (F8); the
§ 2d document a documents-only push (F9); the changelog (F10); dead guards out (F11); each door's rows what it adds
(F12); the science table's mode named in `testing.md` § 6 (F13); the contract's built list cut to its open pointer
(F14).

## The reader of the last fixes: 4

All taken (`r3d2b2b2b2-lastfixes-decisions.md`): **G1, "1 quanta" where a mean rounds to one** — F1 picked the noun on
the raw number while the figure shown was rounded (the dialog's own defaults, a driven mode ticked warm at the record's
temperature, read *The mode holds 1 quanta on average*): one rule in `vibration`, the noun the figure's, at the page,
the command line and in the set's words. Also: `ui-contract.md` § 11 states the note never shrinks (G2); the science
paragraph — the Morse ladder cited (Morse 1929, checked against Crossref), a soft stretch, a torsion's gaps shrinking
too, the electrons at one bias (G3); the command line's "no state" asked of a folder that is no run, its check able to
fail (G4); the note held at its fractional height (G5). Not taken: a result before schema 7 whose species label names
no element (older files only, round 1's Y15). The reader checked the thermal readout against 2.82.0's on three real
runs at four temperatures (`compare_head.py`, to ten figures) and derived every claim of the science note.

## The e2e run

One run, last: `tests/test_vibration_mechanics_e2e.py`, `tests/test_vibration_e2e.py`,
`tests/test_siesta_vibration_e2e.py`, `tests/test_spectrum_results_e2e.py`,
`tests/test_transport_on_a_real_junction_e2e.py`, and the two nodes of `tests/test_molbuilder_e2e.py` that save through
the tree picker (`test_saving_to_the_project_writes_the_pair_and_remembers_where`,
`test_a_picture_saved_under_a_name_taken_leaves_the_first_alone` — the page-load check of the changed page scripts)
— **368 passed**, 0 failed (37 min). The fast suite after the last fixes: 6768 passed, 0 failed. Seen failing first,
on mutants in the working tree, each restored: the note allowed to shrink (*Next* rose 19 px, the dialog's top still),
the tree picker unanchored (its top moved), the run read before the state (both command-line refusals), *1 quanta* at
the page (*The mode holds 1 quanta on average*), the command line and the set's words.

## Topic → test

| topic | test |
|---|---|
| what a state means for a mode, per state, in the set's words | `tests/test_vibration_frames.py::test_what_a_state_means_for_a_mode_before_its_set_is_written` |
| the doors ask the state every time, each value by the table's rule | `::test_the_doors_ask_the_state_every_time` |
| many quanta warned of beyond 1 eV, never refused | `::test_the_doors_warn_of_many_quanta` |
| the n-quanta rule on its window; the coherent state one test | `tests/test_vibration_mechanics.py` |
| the write and load doors refuse before any run is read, their own keys | `tests/spectra/test_blueprint.py::TestLoadEndpoint` |
| the science's own table (mode 48) | `tests/spectra/test_what_a_temperature_means.py` |
| end to end: the command line in each state, its refusals exit 2 | `tests/test_vibration_mechanics_e2e.py::test_the_command_line_writes_a_modes_set_in_any_state`, `::test_the_command_line_refuses_a_state_before_any_run` |
| end to end: the load door's readout per state, the script's numbers | `tests/test_spectrum_results_e2e.py::test_what_a_frame_sets_state_means_is_served_as_the_script_says_it` |
| end to end: the dialog — each state's fields, its note, still under the pointer, the confirm, the set written | `::test_write_frame_set_asks_the_state_and_writes_it` |
| end to end: the tree picker's top still as it grows | `tests/support/pages.py::where_and_name_answered`, in every save-where road |
| end to end: transport over the doors' sets | `tests/test_transport_on_a_real_junction_e2e.py` |
