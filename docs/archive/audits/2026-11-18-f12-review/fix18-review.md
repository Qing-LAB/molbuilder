# F12 fix 18 (spectra: each fact in one home, 2.77.26, R3d1b) — the review

**Verdict: closed with 2.77.26.** The `r2-6` audit's spectra rows (F1–F6, F8–F13, F15; F7 shipped in
2.77.25, F14 left by the R3 triage, its T rows R4's) and `fix17-review.md` round 1 #28, the spectra
page's *Engine* line, each fact given one home.

The lessons: a change called internal is checked against what a launched stage is compared by — two
tidy-ups changed statements the PySCF deck's same-calculation reading compares, which would have
refused every launched PySCF stage prepared again in place, and they were taken back to a release that
changes those decks and says so (round 1); a lookup moved onto a shared path is asked only where its
input exists — round 1's partition fix asked the table for the dipole view's empty partition and broke
a held PySCF result's load, which round 2 found and the held-water e2e had pinned all along; a check at
load raises what no two-way import catches, or a drift reads as a module not found.

## What was built

- **One conversion of a SIESTA reading into frames** (F8): `parse.engines.siesta_frames`
  (`trajectory_of`, `structure_of`), asked by the registered parser and the vibration's steps; the
  relaxation record the record step writes is the viewer's of the same output; it travels in
  `mb_vibration.pyz` with `frame`.
- **The result's keys, one table** (F12): `spectra.result_keys` — the reader's refusal and the data
  file's tables, checked against each other as the data file loads; a key the data file cannot state
  refused at write in every block.
- **The SIESTA grammar's tables** (F5, F13): `FC_STEP_PATTERN`, `fc_step_count`, `CHARGE_PARTITIONS`,
  `charge_partition`.
- **Words from their homes** (F4, F6, F9, F3): the route words, a point's ended words, the optical
  citation's two words, the clock (`runrecord.stamp`) for the SIESTA result, the sweep, the sidecar and
  the parse log, the pair's suffixes and the one rule a sidecar's name follows (`runfiles.sidecar_of`).
- **The record readers** (F1, F2): both sidecars through `persist.object_of` (`refuse_non_finite`);
  the result's version by one rule.
- **The progress log's SCF table** (F15): `trajectory_log.keys.SCF_TABLE`, the writer's header and rows
  and the reader's places from it, one atom-row formatter.
- **The *Engine* line** (#28): `engine_words` among the spectra page's constants.
- **F10, F11**: the structure the result states through `Structure(frozen_atoms=)` and
  `to_dict(view=False)`.
- **Withdrawn**: the emitter told its engine by the deck, the PySCF script's clock (round 1 #2) — later.

## Round 1 — eight readers (correctness A/B/C, one-home A/B/C, notes-and-tests, pages): 49 findings

Grouped by owner; each reader's ids in brackets (cA correctness A, oA one-home A, …, nt notes-tests, pg pages).

| # | finding (readers) | verdict |
|---|---|---|
| 1 | A script whose first import is the spectra reader (`from molbuilder.sidecars.spectra import parse_spectra_json`, § 6.8's door) dies at load, `No module named 'persist'`: the result module now loads the parse package (the SIESTA grammar), whose registered spectra parser imports the half-loaded reader (cA1, cB2, oC1; oA's "fails on 2.77.25 too" re-measured: 2.77.25 loads it) | **fixed** — `parse/sidecars/spectra.SpectraSidecarFileParser.parse` imports the reader inside itself, the codec pair's way (`backend-architecture.md` § 2); `tests/spectra/test_script_doors.py` (§ 6.8's door and `postprocess`, each the first import of a fresh isolated python), failing at its assertion on round 1's tree |
| 2 | A PySCF stage launched under 2.77.25 could not be prepared again in place: the deck's emitter line gained `engine=`/`generator=` (F15) and the vibration script's clock became `runrecord.stamp` (F9's PySCF part, which the finding had not listed) — statements the same-calculation reading compares (cB1, oB1, nt1) | **fixed by withdrawal** — both deck-text changes taken back; the PySCF decks' compared statements are 2.77.25's (`deck_record.same_calculation` of both releases' rendered optimization and vibration decks: no difference); the emitter keeps its engine and writer, with a note why; *later* in the plan: both with the release that next changes the PySCF decks' statements, said as its visible change |
| 3 | `load_text` (now `load_record`, reading bytes) and the deleted `molstruct.stamp()` still named in `structure.md`, `structure-molstruct.md`, `structure-annotations.md`, two tests' docstrings; `state.js` cites `molstruct.PAIRED_SUFFIX`; the codec's notes still say "text" and list its old imports; `is_sidecar` "the one module that names the suffix" (cC1, oC2, oC8, nt2, nt3) | **fixed** — each renamed or said as it now is |
| 4 | The sidecar's name derived twice — `runfiles.sidecar_of` (new) and `molstruct.sidecar_path_for` — disagreeing on `A.XYZ`; `structure.md` § 2.4 says no second copy (cC2, oC3, nt3) | **fixed** — one rule, `runfiles.sidecar_of` (the last suffix replaced, the suffix not read, as the codec always did); `sidecar_path_for` asks it; § 2.4 names it and `SIDECAR_SUFFIX` |
| 5 | `ROLE_OPTIMIZED_SIDECAR` moved under the engine-files heading (cC3, oC8) | **fixed** — a note: a row of `WRITTEN`, its name by the pair's rule |
| 6 | The reader's switch `finite=` beside the writer's `refuse_non_finite=`; `job-contracts.md` § 6.1 states only the writer's (oC4) | **fixed** — one name, `refuse_non_finite`, on both; § 6.1 states the reader's; `tests/test_persist.py` rows (NaN, an infinity, an overflow refused by name; an underflow read as 0) |
| 7 | `GEOMETRY_SUFFIXES` types `.xyz` beside `PAIRED_SUFFIX`; `structure.py:90` a third copy (oC5) | **fixed** for the codec (`(PAIRED_SUFFIX, ".pdb")`); `structure.py` *left as is* — it predates, and L1 cannot ask the codec |
| 8 | `parse/_log._now_iso` the clock's body again (oC6) | **fixed** — the parse log asks `runrecord.stamp` |
| 9 | A sidecar that is not UTF-8 is now refused in `persist.unread_words`' words (oC7) | **accepted** — changelog |
| 10 | `runrecord.stamp`'s note does not list what it stamps now (oC8, nt10) | **fixed** |
| 11 | The reader now refuses an `ir_route`/`raman_route` outside its words; § 6.7 does not say it, `datafiles.md` and § 6.5 list `analytic` as a Raman route (cA2, oA1, nt9) | **accepted** (only a hand-edited file meets it; no writer ever wrote another word; the reader refuses an unknown phase word likewise) — § 6.7 states it, with the version rule; the route rows corrected; changelog |
| 12 | `OPTICAL_STATES` changed order in its refusal (cA3, oA5, oB8, cB note) | **fixed** — `_run_ending.ENDED_WORDS` in the order the docs list (`done`, `not done`, `failed`, `not run`): the tuple is 2.77.25's again |
| 13 | A bare 6 left in `checked_partial_charges` (cA5, oA3) | **fixed** — `len(FC_STEP_PATTERN)` |
| 14 | `"pending"`/`"finished"` still spelled in `siesta_vibration`, `raman` (oA3) | **fixed** — `OPTICAL_PENDING`/`OPTICAL_FINISHED` |
| 15 | A partition's words re-derived by case-folding — `siesta_reader`, `methods`, `datafile` (oA3, oB5) | **fixed** in the reader and the Methods paragraph (`siesta_grammar.charge_partition`, the table's row for a record's or SIESTA's word); the data file *left as is* — asking the grammar would load the parse package with the data file (finding 1's road) |
| 16 | `result_keys` says it holds each block's keys; the charges block's are literals; `relaxation`/`thermo` are their builders' (oA4) | **fixed** — `PARTIAL_CHARGES_KEYS`, `CHARGE_STEP_KEYS` moved in; the module's note says which blocks it holds and whose the others are |
| 17 | Notes: § 6.1/§ 6.2/§ 6.10.1 and code notes say `Structure.to_dict()` where the writers call `to_dict(view=False)`; `result_keys` calls `atom_permutation` a sibling; a pointer left where the key tables stood; `results.py` says L1 (oA6) | **fixed**, but the layer label *left as is* (predates; R5's documents) |
| 18 | § 4.1's list of the PySCF script's imports lacks `stamp` (cA4, oA6) | **not a defect** after #2 — the script no longer imports it |
| 19 | F8's record differs on a killed run's live-SCF tail: `n_steps` one higher, now the viewer's (cA6, oA2) | **accepted** — changelog; the brief's "no visible change" corrected |
| 20 | PySCF's SCF columns declared twice (`keys.SCF_TABLE`, `molwatch_grammar.SCF_COLUMNS`); the emitter's hook types the keys a third time (oB2) | **fixed** for the two tables — checked against each other at import, the clock named once (`keys.SCF_CLOCK`); the hook's record *left as is* — its keys are checked by the cells table at import and by the round-trip row |
| 21 | Point words still spelled in `runpoints`; `runs-of-points.md` § 6 names only `NOT_RUN` (oB3, cB note) | **fixed** in `runpoints` and § 6; the run states (`finished`, `failed`) are the run door's, not a point's — *not a defect* |
| 22 | The default partition (the user's "default to hirshfeld") became the grammar table's first row; the grammar's note and `parse.md` § 4 do not say what it holds now (oB4) | **fixed** — `infrared.DEFAULT_PARTITION = "hirshfeld"`, checked against the table at load; the grammar's note and § 4's line say its step order and partitions |
| 23 | `"siesta"` typed as the finish's `source_format` (oB7) | **left as is** — the finish's engine word, as its `engine="siesta"` in the same call; the vibration framework (R3d2) rewrites the finish |
| 24 | `parse/engines/molwatch.py`'s sketch of the log lacks the clock column (oB9) | **fixed** — the sketch as the table writes it |
| 25 | `web-api.md` omits `engine_words`; § 6 does not state the fallback; the page's note on `K` says physical constants only (pg1–3, oA6) | **fixed** |
| 26 | The version-as-text test is a second example of one rule, on a file only a script can make (nt4) | **fixed** — retired |
| 27 | The data-file refusal test: three rows of one rule; its "silent" is not (nt5) | **fixed** — one row, the docstring saying what changed (the refusal at write, not at the JSON's next read); the load check's own refusal *untested* (no drift to load) |
| 28 | `THE_DECKS` typed in three test files (nt6) | **not a defect** after #2 — gone with the withdrawal |
| 29 | The SCF round trip's docstring vs its 0 for the first cycle's change (nt7) | **fixed** — cites `SCF_COLUMNS`' definition |
| 30 | `pyscf/input.py`'s note "the emitter being every engine's" (oB6, nt8) | **not a defect** after #2 — gone |
| 31 | Untested: the sweep's `generated_at`; `object_of(refuse_non_finite=True)` (nt11) | **fixed** — an assertion in the sweep e2e; the persist rows (#6) |
| 32 | Two e2e docstrings do not name their new checks (nt12) | **fixed** |

Tally: 49 findings by reader count (pg 3, cC 3, oC 8, cA 6, oA 6, oB 9, cB 2, nt 12) in 32 groups — fixed 24 groups (two of them by withdrawal or in part), accepted 3, left as is 2 (#23, and the parts of #7, #15, #17, #20 said there), not a defect 3 (#18, #28, #30).

## Round 2 — five new readers (correctness A, B, C; one home, all parts; notes and tests): 17 findings

| # | finding (readers) | verdict |
|---|---|---|
| 1 | Round 1's fix #15 broke loading a PySCF vibration that holds atoms, Compute IR on: the Methods sentence asked the partition table for the dipole view's empty partition before it branched, and `/api/spectra/load` answered 500 where 2.77.25 served the result (cA1, cB1, oH1) | **fixed** — the table is asked only for a charges view (`methods.infrared_view_sentence`); `test_vibration_e2e::test_water_with_its_oxygen_held_reports_three_vibrations` failed on the round-2 tree (`KeyError: ''`) and passes |
| 2 | The pair's names: `molstruct.geometry_for` reads the rule backwards by its own slicing; `structure-molstruct.md`'s row calls `sidecar_path_for` "the one pairing rule"; the catalogue's geometry rows type `.xyz` beside `PAIRED_SUFFIX` (oH2) | **fixed** for the row (it names `runfiles.sidecar_of`); `geometry_for` *left as is* — it reads the two suffixes from their one home, backwards; the catalogue's rows *left as is* — each role's name is its own spelling |
| 3 | A point's words still typed in `vibration_phases.count_words`, `transport/run_words`, `transport/record`, where `runs-of-points.md` § 6 now says every surface reads them from their home (oH3, cB note, nt5) | **fixed** — each asks `runpoints`' `DONE`, `NOT_DONE`, `FAILED`, `NOT_RUN` |
| 4 | `ENDED_WORDS` ordered for the spectra refusal, `OPTICAL_STATES` taking its order from it (oH4) | **fixed** — `OPTICAL_STATES` lists its own words in its own order; the run's words' note says nothing of the spectra file |
| 5 | The reader's alias text `"Hirshfeld|Voronoi Atomic Populations:"`; the page types the partition options and default (oH5) | **fixed** for the reader (from the table); the page's *later* — R4 (the pages) |
| 6 | The run summary compares literal route words (oH6) | **later** — R4 (the pages); it predates |
| 7 | The step count `1 + len(FC_STEP_PATTERN)·n` written three times (oH7) | **fixed** — `siesta_grammar.fc_step_count`, asked by the result, the finish and the run's size words |
| 8 | The script-door test's `postprocess` row passes on round 1's tree and cites § 5.11 § 10 (nt1) | **fixed** — one row, § 6.8's door |
| 9 | Two emitter tests redundant with the new round trip (nt2) | **fixed** — retired (`test_scf_cycle_hook_snapshots_wall_time`, `test_opt_step_hook_emits_wall_time_column_in_scf_rows`); the buffer's own test kept |
| 10 | Two of the four `persist` rows: an infinity takes NaN's path; the underflow's zero is no document's rule (nt3) | **fixed** — two rows, a non-finite word and an overflowing literal |
| 11 | `test_molwatch_emitter`'s head "focuses on the writer's output bytes" (nt4) | **fixed** |
| 12 | `result_keys`' note "imports nothing that loads the parse package" (nt6) | **fixed** — said of its own imports, and why the data file can ask it |
| 13 | A sweep's `generated_at` stated by no owning document (nt7) | **fixed** — § 5.9's key table |
| 14 | Three blank lines left by the retired test; a docstring quote attributed to § 6.10.5 that is the data file's (nt cosmetic) | **fixed** |
| 15 | `parse/engines/molwatch.py`'s sketch places `wall_time:` after the table and shows no `kind:` (cB note) | **fixed** — the preview and a step, as the writer writes them |
| 16 | `parse_spectra_json`'s relative import has no fallback beside a job (cC note) | **not a defect** — it predates, and nothing reads the result beside a job (the finish writes it) |
| 17 | Correctness C: no defect (cC) | — |

Tally: 17 = fixed 14 (two in part), later 1 (#6, and #5's page half), not a defect 1, none 1.

## The two independent full-text readers, and the discipline reader: 18 findings

| # | finding (reader) | verdict |
|---|---|---|
| D1 | The two `.molstruct.json` refusal tests edited to the code's new words, which no owning section states; both trees raise the error naming the file, so "seen failing" rested on the wording alone (discipline 1) | **fixed** — they assert the error naming the file (`job-contracts.md` § 6.1), and leave the list of rows seen failing |
| D2 | The data-file refusal e2e feeds an invented key, a state the road cannot make (the data file's load check ties its table to the result's, and the reader refuses an unknown key) (discipline 2) | **fixed** — retired; the refusal at write *untested* (changelog) |
| D3 | The new `persist` rows feed a hand-typed NaN and `1e500`, which no writer of ours makes (discipline 3, as 2.69.34 retired the same) | **fixed** — retired; the spectra reader's non-finite refusal *untested* |
| D4 | Six changes with no test the changelog row must name (discipline 4) | **accepted** — the row's *Untested* |
| D5 | The sweep's `generated_at` checked against `stamp()`'s own output (discipline 5) | **fixed** — by the contract's form (§ 5.9) |
| D6 | The spectra reader's note says "the engine filters or nulls NaN" and names a cause, where § 6.8 says the writer refuses (discipline 6) | **fixed** |
| D7 | `runs-of-points.md` § 6 "read from there by every surface" — the pages and the walk script type the words (discipline 7, full-text A 1, full-text B 2) | **fixed** — the sentence names who reads them; the pages' and the walk script's copies *later* — R4 |
| D8 | `engine_words` stated in two docs, the fallback in one (discipline 8) | **fixed** — `web-api.md` points to `spectra.md` § 6 |
| A1 | = D7 | — |
| A2 | The grammar's new tables named in no chapter; `parse.md` § 4 points at § 5d.5, which states neither (full-text A 2, full-text B 5) | **fixed** — § 5d.5's charges row names `CHARGE_PARTITIONS` and `charge_partition`; a row for `FC_STEP_PATTERN` and `fc_step_count` |
| A3 | Three notes short of the code: `siesta_frames`' head; § 5.11 § 2's `mb_pyscf.pyz` list without `result_keys`; § 5.11 § 3's table without `siesta_frames` (full-text A 3) | **fixed** |
| A4 | The three load-time table checks raise `ImportError`, which an importer's two-way import catches and says as a module not found (full-text A 4, full-text B 3) | **fixed** — `ValueError`, each with a note why |
| A5 | A result before v9 is served without `metadata.order_unstated: false` (full-text A 5) | **accepted** — no page reads it, `from_dict` restores it; said in the changelog's *Inside* |
| B1 | § 6.11 § 5 says the data file "today drops [a key] silently" (full-text B 1) | **fixed** |
| B2 | = D7 (and `transport/record`'s `"not ended"` beside `raman.NOT_ENDED`) | the narrowing fixed; the `"not ended"` copy *later* — R4, with the pages' words |
| B3 | = A4 | — |
| B4 | `core.js` repeats the engine-name fallback the server holds for its own surfaces (full-text B 4) | **left as is** — the build door's precedent (`build.py`'s `engine_words`, its fallback in the page), and `spectra.md` § 6 states it |
| B5 | = A2 | — |

Tally: 18 = fixed 11 (D7, A2, A4 counted once each beside their duplicates: A1, B3, B5 are duplicates), accepted 2 (D4, A5), later 1 in part (D7/B2's copies), left as is 1 (B4), duplicates 3.

## The reader of the last fixes

| # | finding | verdict |
|---|---|---|
| 1 | `infrared.DEFAULT_PARTITION`'s note gives the two-way-import reason, which does not hold for a module that does not travel | fixed — the note says the check |
| 2 | § 6.10 § 1's `mb_vibration.pyz` sentence lacks `result_keys`; "the keys it states" misdescribes it | fixed — both bundles' sentences, "the result's keys it checks its tables against" |
| 3 | the reader's note says "the viewer and every reader downstream reject" a non-finite number — Python's reader reads it, the viewer reads the door's answer | fixed — the clause dropped |

Tally: 3 fixed. The three are notes, worded as the reader proposed; their files' tests passed after.

## End to end

248 passed (`tools/testrun.py run e2e`, 22 min 48 s): the SIESTA and PySCF vibrations and their spectra page, a PySCF relaxation's outcome and a person reading its run, SIESTA's relaxation, optimization and stopped runs and a run in the viewer, the monitor on a live run, an edit on a recorded structure, the vibration view. The page-load check of `lib/spectra/core.js` is the SIESTA vibration's and the spectrum module's page rows.

## Topic → test

| change | test | seen failing |
|---|---|---|
| the finish's frames beside the job (F8) | `test_monitor_notify_policy::test_the_shipped_vibration_finish_loads_its_readers_from_the_bundle_alone` | on 2.77.25, at its assertion |
| the record step's relaxation record is the viewer's (F8) | `test_siesta_vibration_e2e` (unticked) | not seeable — the two agreed on 2.77.25; it guards the one conversion |
| the spectra reader as a script's first import (round 1 #1) | `tests/spectra/test_script_doors.py` | on round 1's tree, at its assertion |
| the SCF table, writer and reader (F15) | `test_molwatch_emitter::test_each_scf_value_is_read_back_from_the_column_it_was_written_in` | not seeable — no drift on 2.77.25 (fails there only at the constructor's keywords, since withdrawn) |
| the *Engine* line (#28) | `test_siesta_vibration_e2e` (the free atoms' infrared on the page) | read against 2.77.25's code (`siesta 5.4.2`) |
| a SIESTA result's and a sweep's stamp (F9) | `test_spectrum_results_e2e::TestOnSIESTA::test_its_timestamp_…`; the sweep e2e | read against 2.77.25's code (microseconds) |
| a held PySCF result loads (round 2 #1) | `test_vibration_e2e::test_water_with_its_oxygen_held_reports_three_vibrations` | on round 2's tree (`KeyError: ''`) |
| the sidecars' refusals name the file (F1) | `test_molstruct_json::TestLoadErrors` | not seeable — both releases name it |
| the data file's refusal at write (F12); the reader's non-finite refusal (F1); the load-time checks; the route refusal; the SCF header's spacing; the not-UTF-8 wording; a killed call's live-SCF step | — | untested |
