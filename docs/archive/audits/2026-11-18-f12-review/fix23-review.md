# F12 fix 23 (a row's unit said in one place, 2.81.1, R3d2b2b2a) — the review

**Verdict: closed with 2.81.1.** The motion toolkit's third release, split from R3d2b2b2 when the user set the rule
(2026-11-24: *"we shouldn't allow inconsistency in such important topic"*; *"when we have a quantity that has a unit,
then its unit is then blank"*): a structure's row says its unit in one place (`model/structure.md` § 2.2d), the units
table one list with the unit a name carries one reader (`model/datafiles.md` § 2c, `units.py`).

The lessons: a rule held at a writer judges every copy, molbuilder's own among them — hold it where the rows are
MADE (round 2: the frame generator stopped on a row it only carried, a 500; round 3: a read-only tab could no longer
export an older set, since every tab posts to the same routes — the browser's one maker is the Info page's *Apply*);
and a test is worth what it guards — asked at the commit "what in our api now exists to help these tests pass?",
two review fixes turned out to be code for older records alone (a launched PySCF stage's rows compared as read, a
warning only an older set could raise), which the user's rule of 2026-10-16 says no work goes into: removed with their
tests.

## What was built

`units.py` (floor 0): the units table, one row per unit (`_TABLE`), its columns `UNITS`, `UNIT_IN_NAME`,
`UNIT_ON_A_PAGE` (new), and `unit_of_name` (the longest ending after an underscore; an inverse carries none; a bare name
none). `structure._customized_row(raw, where, *, made=False)` the row's one validator: every reader keeps a row as
written, the same unit beside its name (spelling or page look) and a text's `1` not kept; `made` holds the rule —
`structure.made_row` (asked by `set_customized` and the route `/api/structure/row`, MolView's `customized.made`, which
the Info page's *Apply* asks), `with_frames`' frame rows. Every copy carries rows as read: the save, the download and
the hand-over from any tab, the codec's write (the frame generator, the CLI, `describe`). `units.unit_spelled` reads a
unit's spelling from its page look. `ncdata.check_said` asks `unit_of_name`; `ncdata` and `structure`
travel with `units`. `vibration_frames.Row.unit` from the name, `page_unit` from the table, `unit_written`;
`mode_frames` writes `unit_written`. MolView: `customizedRow` checks the shape, the unit field's words say the rule.

## Round 1 — four readers (correctness 4, one home 9, notes and tests 12, API 7): 32

Decisions P1–P11 (`r3d2b2b2a-round1-decisions.md`). **The user's two choices**, put after round 1 found strict reads
broke records molbuilder had written (decks, results, data files, composed junctions): the rule binds whoever writes a
row, and whatever is already written is read as it is (the older-file machinery and sidecar v18 dropped); the unit slot
takes any spelling. Also: the refusals naming the ending read and the rename (P1); the browser checks the row's shape,
the rule said at the save (P2, Apply carried to R4); one reader of a name's unit (P3, P4); the table named by its home
(P5); the page column (P6); the foreign-unit warning judged unreachable and removed (P7 — undone in round 2); examples,
tests and documents (P8–P11).

## Round 2 — four new readers (correctness 4, one home 7, notes and tests 10, API 8): 29

Decisions Q1–Q15 (`r3d2b2b2a-round2-decisions.md`). **Q1, a regression round 1 made, three readers**: the check sat
in `StructureCodec.write`, so the frame generator refused a row it only carried from the vibration's structure (the
Spectra tab's *Write frame set…* a 500, `frameset from-mode` a traceback), and Save refused what Download and the
hand-over wrote — the rule moved to where rows are made: the script's doors and the web's one writing sequence.
**Q8**: a PySCF stage launched on 2.81.0 was refused at its next prep (its deck's rows compared raw) — its rows
compared as read; **Q7**: P7 undone — rows read as written can carry another unit; the warning back (both removed at
the commit, below). Also: `with_frames`' frame rows held (Q2); every
bad row in one refusal (Q3); the no-unit words (Q4); the page look the same unit (Q5); a bare name no unit (Q6); one row
per unit (Q9); notes, exports, words, the Info page's field, documents (Q10–Q15).

## The full-text readers and the discipline reader (fulltext-A 5, fulltext-B 9, discipline 8): 22

Decisions R0–R13 (`r3d2b2b2a-round3-decisions.md`). **R0, a regression round 2 made, both full-text readers
independently, put to the user**: every read-only viewer posts its Export → Data and hand-over to the routes Q1 had
judging rows, so a frame set or run structure an earlier molbuilder wrote could not leave them, their Info page unable
to state a unit — the user's answer (2026-11-24, "A"): the rule at the Info page's *Apply*, asking the server's one
rule; every copy carries rows as read. Also: the version a last-number move, 2.81.1 (R1); the same-calculation
contract and notes (R2); the changelog's Checked and Untested lines (R3); the rule's table at one door (R4); the deck
read as Python reads it (R5); no document path in a refusal (R6); § 2a rule 2 (R7); § 8, § 9 (R8); a unit's spaces
(R9); notes (R10, R11); the refusal's words in a unit's name and a rename from a page look (R12, R13). **For the user
at the commit**: three tests build an earlier release's records (discipline 1); `ncdata.check_said`'s answers for
names no variable has (discipline 8).

## The reader of the last fixes: 4

All taken (`r3d2b2b2a-lastfixes-decisions.md`): **S1, the Metadata page's Apply, now waiting for the server, read its
form after the answer** — a double click or a Cancel meanwhile said *Not applied -- Cannot read properties of null*;
the form held as clicked, and a browser test of Apply added, which failed on that and on an Apply asking no one. Also:
the doors table's browser row (S2); MolView's five routes and the `made` door in its tables (S3); the refusal's rename
replacing a real unit's ending (S4).

## The e2e run

484 passed, 0 failed (39 minutes): `test_transport_on_a_real_junction_e2e.py`, `test_vibration_e2e.py`,
`test_siesta_vibration_e2e.py`, `test_inspector_registry_e2e.py`, `test_spectrum_results_e2e.py`,
`test_vibration_mechanics_e2e.py`, `test_pages_no_js_errors.py`, `test_molview_e2e.py` (the Metadata page's Apply
among it), `test_a_stage_that_has_run_e2e.py`, `test_molbuilder_e2e.py`.

After the two removals at the commit: the fast suite, 6652 passed, 0 failed; `test_a_stage_that_has_run_e2e.py` and
`test_transport_on_a_real_junction_e2e.py` again, 236 passed, 0 failed.

## At the commit

The user's answers (2026-11-24). To the three tests that built an earlier release's records (discipline 1): *"what in
our api now exists to help these tests pass?"* — two pinned code that exists for older records alone (Q8's comparison,
six lines in `pyscf.script_input._by_what_it_holds`; Q7's warning, twelve lines in `vibration_frames.weights_for_set`),
which the user removed with their tests (*"remove them, then commit"*): a PySCF stage launched before 2.81.1 whose
structure carries a row with its name's unit or a text's `1` is not the same calculation when prepared again, and runs
again; a set written before whose row states another unit is taken in its name's, unsaid. The third — every tab
carrying a record's rows — guards today's doors, kept as the named exception in `process/testing.md` § 6. Then the
fast suite and the end-to-end files that read the removed code again (below).

## Process

- Round 1's strict reads were built before the user was asked how records already written read; the user's choice
  replaced them. A question put to the user on a made-up scenario confused them — name the data structure first and
  propose the simplest rule.

## Topic → test

| topic | test |
|---|---|
| the rule at the script's door: kept, not kept (the table's spelling, a page's look, spaces), refused with the fix | `tests/test_row_units.py::test_a_row_made_says_its_unit_in_one_place` |
| a frame set's rows built through `with_frames` | `::test_a_frame_set_built_says_its_frames_units_in_one_place` |
| rows already written read as written and carried by molbuilder's own write | `::test_rows_already_written_are_read_and_carried_as_written` |
| the Info page's Apply: the server's one rule | `::test_the_info_page_makes_a_row_by_the_one_rule`, `tests/test_molview_model.py::test_a_named_value_is_made_by_the_servers_one_rule`, `tests/test_molview_e2e.py::test_a_named_value_applied_is_made_by_the_servers_rule` (the click, a double click) |
| every tab's save, download and hand-over carrying a record's rows | `tests/test_row_units.py::test_every_tab_carries_a_records_rows_as_written` |
| the page unit of each vocabulary row, from the table | `tests/test_vibration_frames.py::test_a_page_says_a_row_in_the_vocabularys_words_where_its_unit_is_the_vocabularys` |
| a data file's variable and the unit its name carries | `tests/test_data_file_states_itself.py` |
| end to end: the generator's sets, transport over them, the record and `summarize`; MolView's page opening | `tests/test_transport_on_a_real_junction_e2e.py`, the page-load node |
