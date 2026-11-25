# F12 fix 24 (the motion toolkit's Python door, 2.82.0, R3d2b2b2b1) — the review

**Verdict: closed with 2.82.0.** The motion toolkit's fourth release, the first half of R3d2b2b2b (the user's design
of 2026-11-23, `engines/vibration-frames.md` § 1, § 4, § 5): a vibration read from its run's folder, its frame sets
written by one writer and read back, the outermost move for any state — from a script. The command line and the page
for any state follow in R3d2b2b2b2.

The lessons: a reader that classifies by what it finds must ask for what the toolkit writes, not what a person may
write too (round 2: a person's one-mode set in the vocabulary's names read back as the toolkit's `Frames`, its degree
and rows the rule's, not its frames'); a rule stated "exact in memory" meets the file's six decimals the moment a set
is read back and written again (round 3: a duplicate frame 0 on every real geometry — the user chose the pair's
precision, the structure's one rule of two places); a door that takes one form must say that form, not every form
another door takes (round 3: the command line told to give a result file it then refused); a test model whose
geometry is round hides what a relaxation's digits show.

## What was built

`vibration_frames` (the host's): `vibration_of(path)`, the one door telling a run's folder from a result file
(`TWO_FORMS` naming both where both are taken); `vibration_of_run(run_folder)`, the run door (the run, the JSON's
sha256, frame 0's relaxation record or `relaxation_said`; `_relaxation_record` moved from `spectra.mode_frames`);
`write_frame_set` in two forms → `Written(path, n_frames, said)` — frame 0 the relaxed geometry (a person's first frame
within `SAME_PLACE_ANG` of it is frame 0), the measured rows, the weights, the provenance rows, the masses, an
`assumptions` row in one wording of each state (`state_words`), § 5.1's rows of its kind, a snapshot set's
`info.sampling` recording its whole motion, frame 0's `info.calculation`; `read_frame_set` → `Frames` (only where the
set states the toolkit's own rule), `Snapshots` (only from `info.sampling`), else `FrameSet`, the vibration given
checked by its result's sha256 and brought to the set's scale, the run reopened from `vibration_run`;
`outermost_move`, `outermost_move_of`; `_mode_set_of`, the one reader of a mode set's rows into its state. `vibration`:
`outermost_move_ang` (the one rule), `Vibration.mode` (the one rule of a mode's number), `Vibration._at_total` (the one
place a vibration is set at a scale), `relaxation_said`, `DRIVEN_KINDS`, `CLASSICAL_KINDS`, `Frames.read_said` /
`read_mode`, `Snapshots` readable without a vibration, `pooled` refusing such a set. The read door:
`sidecars.spectra.result_role_of`, `result_sha256_of`, `spectra.datafile.json_sha256_of`,
`SpectraResults.vibration(..., result_sha256=)`. `postprocess`: `vibration` handing on `vibration_of`,
`write_frame_set`, `read_frame_set`, `pooled`, `outermost_move` for any state. `spectra.mode_frames.frameset_from_mode`:
the pair's name first, the run door, the state's own rule, the one writer.

## Round 1 — five readers (correctness 7, one home 10, notes and tests 14, API 12, science 5): 48

Decisions U1–U23 (`r3d2b2b2b1-round1-decisions.md`). **Road breaks**: `postprocess.read_frame_set` missing (all five
readers, U1); the transport e2e's rows without the new `quantum` row (U22). **A notebook's breaks**: a person's set in
the vocabulary's names not read back (U2); a set read with another vibration silently wrong (U3); a set from a result
file inside a run said to be "from outside a run's folder" (U4). Also: a set read back without its vibration written
again refused by name (U5); its mode kept (U6); said lines with the way on (U7); `vibration` of a wrong path naming both
forms (U8); frames from 1 (U9); the record's temperature (U10); provenance rows refused (U11); the set's temperature
(U12); `outermost_move_of` (U13); one home for the held-atom check, the largest move, the vocabulary index, the
sampling rules, the run's address (U14–U17); notes (U18); the name asked first (U19); the `quantum` row's science
(U20); the frames in one read (U21); tests (U23).

## Round 2 — five new readers (correctness 5, one home 7, notes and tests 6, API 8, science 3): 29

Decisions V1–V17 (`r3d2b2b2b1-round2-decisions.md`). **V1, a notebook break three readers found**: a person's one-mode
set in the vocabulary's names read back as the toolkit's `Frames` — its degree a row's, written again with the rule's
rows and outermost move, not its frames' — a set is a mode's own `Frames` only where it states the toolkit's own rule
(`sampling_rule` gauss-hermite or gauss, `sampling_nodes`) and every frame's coordinate. Also: the vibration given
checked first, by `_provenance` (V2, V3); its scale brought to the set's (V4); one dispatch of the result's two files
(V5); `Vibration.mode` the one rule (V6); the masses refusal the reader's (V7); documents and notes (V8, V9, V11); the
kinds as tuples, `State`'s fields (V10); a snapshot set's assumptions — every mode counted, the background where kept
(V12); tests from the closed forms, the read-back motion whole (V13); a `FrameSet` given to the writer (V14); said
lines (V15); a row that is no mapping (V16); the changelog's moved refusals (V17).

## The full-text readers and the discipline reader (fulltext-A 7, fulltext-B 6, discipline 14): 27

Decisions W0–W16 (`r3d2b2b2b1-round3-decisions.md`). **W0, the user's two answers (2026-11-24)**: (1) both full-text
readers found a person's set read back and written again, as the writer's refusal says, gained a duplicate frame 0 on
real geometries (the file holds six decimals; the rule was exact) — frame 0 recognised at the pair's precision, the
structure's one rule of two places; (2) the fast rows keep the model molecule, its geometry given a relaxation's
digits, a named exception in `process/testing.md` § 6. **W1 (fulltext-A 2, discipline 1)**: the command line's and the
page's refusal named a result file they then refused — the run door names the run's folder alone, the door taking
both forms (`vibration_of`, W2) both. Also: imaginary modes named only where not included (W3); `quantum` follows the
background (W4); `frame0_rows` held to the row rules (W5); the changelog's moved refusals (W6); a scale of 1 the
result's own, the reader's matches exact (W7); a set of one frame refused (W8); the state words from the contract's
table (W9); one said line for a vibration not read from its run's folder (W10); each refusal's class tested (W11);
stand-ins named, the e2e's forged sha256 dropped (W12); `info.sampling`'s keys in the contract (W13); the blueprint
row (W14); maps and notes (W15); documents (W16).

## The reader of the last fixes: 5

All taken (`r3d2b2b2b1-lastfixes-decisions.md`): **X1, the vocabulary's `quantum` meaning in code still the old
words** — W4 changed the contract's row, not the note every set carries; also: the contract's § 8 and "outside a run's
folder" lines (X2), the writer's docstring and `_keeps_background`'s note (X3), the frame-0 row checking frame 0 is
written exactly and a set written again compared frame by frame (X4, its mutant caught), `SAME_PLACE_ANG`'s askers
(X5). The decisions held as built: frame 0 taken only within 10^-6 Å per atom, a person's first-frame rows kept; HEAD's
doors' sets and this tree's read back and written again the same on a real run of points; `_at_total` right for every
caller of `with_frequency_scale`.

## The e2e run

One run, last: `tests/test_vibration_mechanics_e2e.py`, `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py`, `tests/test_spectrum_results_e2e.py`, `tests/test_transport_on_a_real_junction_e2e.py` — **354 passed**, 0 failed (35 min). The fast suite after the last fixes: 6745 passed, 0 failed. No page script changed, so no page-load node was due.

## Process

- The reviewer's scratch: `compare_head.py` sets HOME and XDG but not TMPDIR — one reader's script wrote two
  4-byte folders in /tmp; briefs now say to set TMPDIR for every script.

## Topic → test

| topic | test |
|---|---|
| each recipe's set written with the rows of its kind, its words the contract's | `tests/test_vibration_frames.py::test_a_recipes_set_is_written_with_the_rows_of_its_kind` |
| a set read back as written, its motion whole, pooled further, written again the same | `::test_a_set_read_back_is_the_set_written` |
| a mode set's derived rows against science § 2b.3's closed forms | `::test_a_mode_sets_derived_rows_are_the_sciences` |
| a person's frames, frame 0 at the pair's precision, a person's set written again | `::test_a_persons_frames_are_written_with_what_every_set_needs`, `::test_a_persons_frame_0_is_prepended_unless_it_is_the_relaxed_geometry`, `::test_a_persons_set_read_back_is_written_again_as_it_was` |
| a person's set in the vocabulary's names read as a person's | `::test_a_persons_set_in_the_vocabularys_names_reads_back_as_written` |
| the writer's refusals and what it says | `::test_the_writer_refuses_by_name`, `::test_the_writer_says_what_it_writes_as_given`, `::test_a_set_from_a_result_file_says_the_way_to_a_citation` |
| the reader's and the doors' refusals, each in its class | `::test_the_reader_and_the_run_door_refuse_by_name` |
| a run that no longer reads; imaginary modes named | `::test_a_set_whose_run_does_not_read_now_is_read_without_it`, `::test_a_snapshot_set_names_the_imaginary_modes_it_leaves_still` |
| a set at the result's own frequency | `::test_a_set_at_the_results_own_frequency_states_no_scale` |
| the outermost move for any state, its door by a mode's number | `::test_the_outermost_move_is_the_written_sets_for_any_state`, `::test_the_outermost_move_of_a_mode_by_its_number` |
| the pair's name asked before the run is read | `tests/spectra/test_blueprint.py::TestLoadEndpoint::test_a_pair_already_there_is_refused_before_the_run_is_read` |
| a mode's number one rule; frames from 1 | `tests/test_vibration_mechanics.py` |
| end to end: a real run's folder, its relaxation answer, a set written and reopening its run, any state's outermost move | `tests/test_vibration_mechanics_e2e.py::test_a_run_folder_reads_as_its_vibration_and_writes_its_set` |
| end to end: the doors' sets and transport over them | `tests/test_vibration_e2e.py`, `tests/test_transport_on_a_real_junction_e2e.py`, `tests/test_spectrum_results_e2e.py` |
