# F12 fix 30 (the older files dropped, and a file that does not read said as one refusal, 2.87.0) — the review

**Verdict: closed with 2.87.0.** The last of the three releases the baseline review of 2.85.0 and the user's decisions
of 2026-11-26 set (2.85.1 the clean-up, 2.86.0 the weights, 2.87.0 the older files dropped). Each file molbuilder
reads is read at its current version alone — the vibration result at its one schema, its data files at their one
format, the structure pair's sidecar at its one version (saved from 2026-11-04), the transport composition record at
its one format — and a file of another is refused by name, its version and the way on said, at every door a person
reaches it by.

The lessons: **a refusal is a door's, not a reader's alone** — a reader that refuses an older file well still left
eight doors dying of a Python error or a server error (a frame set of a current result, `summarize`, the readings
list, the extension pick, Task setup's folder, `describe --reuses`, `validate`, `modify`), each found only by a reader
who walked the doors with an older file in hand; a table of the doors (`_DOORS_OF_A_CALCULATION`) now holds them.
**A refusal is one kind, said at the edges — never fixed door by door** (the user, 2026-11-27: *"why the fuck every
time you try to fix crash like this?"*; *"generic is the correct way - with only exceptions where we know some versions
can be translated as needed"*; *"versions can be handled differently for structural, calculation setup etc. so you
should investigate rather than one size fitting all"*): the release's first part patched each door a reader found
dying; the second made every reader's refusal of a file it cannot read carry one mark the command line and the web
server say once, and gave each family of files its own way on from an inventory of every reader and door. **A mark on
a type broader than its file refusals is a promise the code does not keep**: the mark went on whole base classes first
(a checkpoint's git error, a parser registry's misconfiguration, a Raman request's values), then on the file refusals
alone, each a marked type of its own. **A gentle door catches the mark, not `ValueError`**: an `Exception`-based marked
refusal passed through two fail-soft catches and closed the Results tab and Task setup on a calculation they had always
opened (round 2's regression). **The way on is the holder's**: a pair's refusal tells a person to save the pair again, which for a calculation's own
copy or its composed junction leads nowhere — the refusal now carries its fact alone (`MolstructVersionError.fact`),
and each door that holds such a pair for a calculation says the calculation's way on.

## The user's decisions (2026-11-26)

- Asked: *"every vibration result before 2026-11-23 (all your October vibrations), every structure pair saved before
  2026-11-04 (cited junctions, saved relaxations, frame sets), and transport calculations before 2026-10-14 would stop
  opening, with no rewrite command (re-run needed). Still drop now?"* — *"Drop all as asked"*; told that the composed
  junctions inside transport calculations made between 2026-10-14 and 2026-11-04 are such pairs too: *"Everything
  before 2026-11-04"*.
- Round 1, told that every calculation described before 2026-11-04 then stops at its next prep, *describe it again*
  its way on: *"Yes, apply them all"*.
- **Mine, said to the user with round 1's findings**: a transport calculation composed before 2026-11-04 has no
  usable record — none of its stages launched, its next prep composes the junction again (nothing ran on the old
  one); one launched, refused. **Mine, in the build**: a run's deck of any age stays read (its retired metadata keys
  set aside, I9's deck half) — decks were not among the files dropped.

## What was built

**The vibration result and its data files** (`spectra/results.py`, `datafile.py`, `optical_data.py`): one version
each, refused by `readable_version` and `persist.check_schema` with *run the vibration again*; every older-only reading
gone — the table's masses (`mass_of`), the structure built from older keys, the map made on read, the older names
and defaults, derived values computed on read, `CHARGES_SCHEMA`, the page's defaults for such files (`core.js`,
`_spectra_inspector.html`), `ncdata`'s older-name mechanism (`Var.earlier`, `named_now`, numbers stored as
attributes). **The structure pair** (`sidecars/molstruct.py`, `parse/sidecars/molstruct.py`, `workingcopy_structure.py`):
v17 alone, `MolstructVersionError` with its `fact`; retired keys refused in a sidecar; the hash-less pin, the pre-v12
`offset_by` default, the `cell_origin` notice gone. **Transport**: the composed junction's version refused by
`compose.recorded_junction` in the calculation's words; `slot-provenance@1` by `read_provenance`; the junction "with
no order record" and its prep step gone (`gather.order_record`, `prep._record_the_order`, the kind's `order_record`
part, the record's and the report's branches). **The doors**: `jobset.rung.structure_for` (a calculation described
before 2026-11-04), `vibration_frames._relaxation_record` (its frame set), `transport/sum._run`,
`web/blueprints/transport._transport_calc` and `reuse.reused`, Task setup's folder (`build._folder_frame_set`),
`cli._struct_for_validate`, `atom_index._stated`. **Documents**: `execution/job-contracts.md` § 6.1 (read and
refused, the one home of the list), `model/structure-molstruct.md`, `model/overview.md` (I8, I9, F3, F4),
`engines/vibration.md` § 6, `engines/transport.md` § 1.1, `engines/stages.md` § 6.3, `engines/vibration-frames.md`
§ 1, `model/datafiles.md`, `model/structure*.md`, `web/spectra.md`, `web/results.md`, `web/web-api.md`,
`execution/{job-system,checkpointing,architecture,project-layout,job-request}.md`, `process/testing.md`.

## Round 1 — three readers (documents 16, code 6, tests 6): 28

All applied, the user's *"Yes, apply them all"*. **The code**: a frame set of a current result whose calculation was
described before 2026-11-04 died (500, a traceback); `summarize`, the readings list and the extension pick died on a
calculation composed before 2.10.0; every calculation described before 2026-11-04 refused at prep with a pair's way on
that leads nowhere; the composed junction's refusal carried the pair's way on inside it; `validate` and `modify`
printed a traceback; `atom-index` answered an older result with false words. **The tests**: three e2e calls still
passed the removed `engine` argument; the junction's test could not see the pair's words; a test guarded a rule this
release removed (retired); one version row tested twice and one never written. **The documents**: the transport page
said refused where an unlaunched calculation is composed again; the report's branch for a junction with no order
record, the order-recovery road and its prep step still described on six pages; retired keys said tolerated; version
numbers spelled against the page's own rule; the dropped files listed four ways (one home now).

## The reader of the fixes: 3 findings

Every fix did what its finding asked; each new test failed on the tree before it. Applied, the user's *"Yes, apply
them"*: Task setup's folder and `molbuilder describe --reuses` still died on a calculation composed before 2.10.0 (the
record now read once by `reuse.reused`, the one door of every reuse; the folder opens saying why; both rows of the
doors' table); a section pointer; a version number spelled. Not applied, said to the user: an unlaunched group whose
every file 2.87.0 would write byte for byte keeps an older junction — no group prepared before 2026-11-04 can be (its
run scripts changed 2026-11-21).

## Part B — a file that does not read, one refusal (the user, 2026-11-27)

The user, shown the inventory of every versioned file (configuration and machine, calculation setup, prep and launch
records, results and data files, structure files — about 34 kinds, each reader, door, crash and gentle catch): *"All of
it in 2.87.0"*. Built: `persist.FileRefused` (a mixin, never a `ValueError`), `RecordUnreadable`, `SchemaMismatch`, the
later words (`written_later`, `LATER_WAY`); `cli_edge.GroupSayingFileRefusals` on the command line, the web app's
`errorhandler(FileRefused)`; each family's way on (`job-contracts.md` § 6.1, its table); the three crashes the
inventory found; the door-by-door catches of the first part removed where the edge holds; every family's gap.

## Round 2 — three readers (documents 13, code 7, tests 9): 29

All applied, the user's *"Yes, apply them all"*; the checkpoint archive that does not verify a refusal (400), the
user's *"A refusal, like the rest"*. **The code**: the Results tab and Task setup refused a calculation whose machine
copy does not read (a regression: their fail-soft catches took `ValueError` alone); a later molbuilder's file told an
older one's way on too; a later molbuilder's composed junction composed over at the next prep; the displacement-sweep
view drew a record of another version as current; `jobset migrate` said *current* for a calculation it could not
translate; a benchmark plan of another version listed as no parser's; `jobs status`' and the Results page's words.
**The tests**: the road's `jobset` helper invoked the group directly, so the group carried an edge of its own made for
the tests (removed; the helper types `molbuilder jobset …`); rows on versions no molbuilder wrote; the mark held by a
test where it bears load (the run deck's route); a catch-all test pinning neither re-raise it named. **The
documents**: the rule's quoted words, the two exceptions by design (`molbuilder.json`'s exit 2, the spectra load's 422),
three files the family table lacked, one home for the gentle list, the sidecar's version spelled.

## The reader of round 2's fixes: 9 findings, and the lead's 6 follow-ups

Every item and follow-up did what it asked. Applied, the user's *"Yes, apply them all"*: the mark still on types
broader than their file refusals (a Raman request's values, the data files' writer checks, absences — each file
refusal now a marked type of its own); the parser base's docstrings; the checkpoint page's *fault*; the report cards'
later words on two pages; a transport sentence; the spectra routes' count; a spelled version; test docstrings naming the
wrong release; four untested extensions; `jobset migrate` on a junction edited since prep answered by state; the
hand-over's way on said. The follow-ups (the lead's, before it): `jobset migrate` on an older composed junction by the
calculation's state as prep; the mark split off `ComposeError`, transport `RecordError`, `SpectraJsonError`; the
transport report opening a calculation whose machine copy does not read; `pseudo`'s edge of its own removed; a
transport description claimed by name. **Not tested, said**: `jobset migrate`'s branch for a calculation whose records
open only once migrate has rewritten them — reached only by an older release's template, which `process/testing.md`
§ 6 forbids a test to build.

## The e2e run

After round 2's last fixes: the fast suite, 7059 passed; end to end, 172 passed (`test_vibration_e2e.py`,
`test_siesta_vibration_e2e.py`, `test_vibration_mechanics_e2e.py`, `test_spectrum_results_e2e.py`,
`test_spectra_no_spectrum_sentence_e2e.py`, `test_transport_decks_e2e.py`, `test_bench_summary_inspector_e2e.py` — the
Results page opening a benchmark folder, loading `results/viewer.js` and the displacement-sweep inspector with no page
error), and 232 passed (`test_transport_on_a_real_junction_e2e.py`). None failed.

## Topic → test

| topic | test |
|---|---|
| a pair of another version refused at each door, the way on followed | `tests/test_molstruct_json.py::TestSchemaVersioning::test_a_version_this_build_does_not_read_is_refused` |
| a calculation described before 2026-11-04 refused at prep with its own way on | `tests/data/prep_protocol.toml` (*a calculation described before 2026-11-04 …*), `tests/test_prep_protocol.py` |
| the composed junction's refusal, and the composition record's | `tests/test_transport_compose.py::test_a_calculation_composed_before_the_one_pair_version_is_refused`, `::test_a_composition_record_of_another_format_is_refused` |
| a calculation composed before 2.10.0 refused at every door | `tests/test_transport_task.py::test_a_calculation_composed_before_2_10_0_is_refused_by_name_at_every_door` |
| a vibration result and its data files of another version, through each door | `tests/test_spectrum_results_e2e.py::test_a_file_of_another_version_is_refused_by_that_version` |
| a frame set of a run whose calculation keeps an older pair | `tests/test_vibration_mechanics_e2e.py::test_a_set_of_a_run_stated_relaxed_whose_calculation_keeps_an_older_pair_says_why` |
| a file that does not read refused at every door, through the two edges, each family's way on, a later molbuilder's words | `tests/test_a_file_that_does_not_read.py` |
| a calculation composed before 2.10.0 refused at every door, launch included; the composition record by name and major | `tests/test_transport_task.py::test_a_calculation_composed_before_2_10_0_is_refused_by_name_at_every_door`, `::test_a_composition_record_is_read_by_its_name_and_major` |
| a checkpoint archive that does not verify, a refusal | `tests/test_checkpoint_routes.py` |
| end to end | `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py`, `tests/test_vibration_mechanics_e2e.py`, `tests/test_spectrum_results_e2e.py`, `tests/test_spectra_no_spectrum_sentence_e2e.py`, `tests/test_transport_decks_e2e.py`, `tests/test_transport_on_a_real_junction_e2e.py`, `tests/test_bench_summary_inspector_e2e.py` (the Results page's scripts) |
