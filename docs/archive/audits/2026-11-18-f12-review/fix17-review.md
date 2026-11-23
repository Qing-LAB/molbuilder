# F12 fix 17 (transport: each fact in one home, 2.77.25, R3d1a) — the review

**Verdict: closed with 2.77.25.** The `r2-5` audit's transport rows (#1, #2, #4–#13, #15), `r2-6` F7
and R3c's carried items (`fix16-review.md`'s *later* rows), each fact given one home; and the user's
answer of 2026-11-22, *"Yes, as recommended"*, to the two spellings: every data file names its engine
in the engine table's words, and a voltage is said one way, *0 V*. The spectra rows of `r2-6` are R3d1b
(2.77.26), split off so each release's change can be read whole.

The lessons, in the user's words where they gave them: a fix that skips a check must skip only what is
built from the refused value, never the whole ladder (round 1 found the first form, round 2 a second
check still reading it); a door unified for its answer must stay read once where its callers read it
once (the frame count, asked 15 times by one decision until round 2); a data file shipped beside a job
is named as the loader spells the bundle's path, `./` too.

## What was built

- **The citation's kind, one word**: `compose.CitedRun.kind` / `CitedPair.kind`; `CitationAnswers.kind`
  and `stated` apart; `citation_defaults.asked_of_the_person(kind)`, asked by the check, the panel,
  prep's source of the pseudopotentials, the seed deck's note and describe's answer; the schema route's
  `source.kind` `"run"`.
- **The ladder by its roles**: `stages.LEAD_RUNGS`, `SEED_RUNG`, `DEVICE_RUNG`, `window.window_rung()`,
  `deck.shape_of`; the sort's lead regions from the kind's roles; the window check judged where its
  spacing is read; `stages.run_folder_at` and `point_key`; `rung.frames_had` the one frame count, held
  once per read by `gather.ReadOnce.frames`.
- **Words and names from their homes**: `runfiles.SPIN_*`, `CHANNEL_*`, `TBT_SPIN_CHANNELS`;
  `identity.COMPUTES_*`; `task.SINGLE_BIAS`, `LOW_BIAS_APPROXIMATION`, `SELF_CONSISTENT`;
  `structure.AXIS_KIND_TRANSPORT`; `kmesh.TRANSPORT_AXIS`; `units.volts_words`; `siesta.layout.keyword`
  and `pyscf.layout.keyword` / `keyword_on`; the engine table's words in every data file
  (`ncdata.state`), `engine_table` travelling with its table; `persist.read_shipped`, the one read of a
  data file shipped with a module.
- **Checks built from a refused value stand aside**: the ladder's loosening and step checks and the
  kind's window check are handed what the setting door refused (`engines/template.md` § 5.3).
- **Reuse**: `_copied_files` and `_measured`, the copy filter and the size count, one each.

## Round 1 — seven readers (correctness ×3 parts, one-home ×3 parts, notes and tests)

| # | finding (reader) | verdict |
|---|---|---|
| 1 | describing an extension that keeps the finished junction (`--reuses`, no `--slot`) crashed in `describe` on `cited.kind` with `cited` None — files written, the CLI a traceback, the tab a 500 (corr-A 1) | fixed — the kind read only where a citation resolved (the new-calculation branch); the e2e `reused` fixture describes `--reuses` with no `--slot` |
| 2 | an e2e helper (`_prepared_decks`) called `point_folders` without `frames` — the real-junction module's fixture would fail (corr-A 2, corr-B 1, home-B 1) | fixed — `frames_had(open_calculation(...))` |
| 3 | `transport/stages` (floor 3) imported `rung.frames_had` (floor 4) in `points_in` and `at_point` — A7 (corr-A 3, corr-B 3, home-B 2) | fixed — `points_in(task, stage, run, *, frames)` and `run_folder_at(task, ..., frames=, downstream=)` are handed the count; transport's points declaration, the record, the gather and reuse ask `frames_had` |
| 4 | the new `stages.at_point` shared its name with `rung.at_point` and shadowed it in the gather (home-A 3, home-B 3) | fixed — renamed `stages.run_folder_at` |
| 5 | documents naming what moved: transport.md § 6.1a `SHAPE_OF_RUNG`, the `stages` row's "tables the shapes", § 2a.11's `points_in`/`point_folders` signatures; structure-annotations.md's lead order (corr-A 4, home-A 4, tests 2, home-B notes) | fixed |
| 6 | before a calculation's first prep, its Results read now lists the cited set's frames (the record asks `frames_had`), where 2.77.24 listed one point (corr-B 2) | accepted — the document's answer and Task setup's; in the changelog |
| 7 | the voltage's one spelling not reached: the extension's meta line `0.0, 0.2 V` (and its § 2a.16 example), and `{v:g} V` typed in gf_files, record, datafile, task, frame_average, validation/transport, validation/siesta, energies (corr-B 4, tests 1, home-B 6) | fixed — `units.volts_words` (moved from `task`, beside `size_words`; `units` travels, so `energies` asks it too, `TRAVELS_WITH`); the meta line *0 V, 0.2 V*, the chapter's example with it; an e2e assertion on the meta line. `task`'s *puts 0.1 V and 0.1 V in one folder* keeps `repr` (it shows two values one token holds); the page's `point-words.js` is the page's own spelling (R4) |
| 8 | the pair seed's note named `jobset init`, a command molbuilder does not have (home-A 1, tests 3) | fixed — `as_typed('describe', '--psml-lib', 'DIR')` |
| 9 | the settings gate typed `rung == "device"` (home-A 2, home-B 7) | fixed — judged on the rung that reads `negf_neq_spacing_ev` (`template.reads`) |
| 10 | `compose._refuse_outside` and `cell.transport_room` indexed the axis as `2` (home-A 5) | fixed — `TRANSPORT_AXIS` |
| 11 | the sort typed both lead role names, against the new note "a role renamed … everywhere" (home-A 6, home-B 7) | fixed — the leads' regions from the kind's roles, left then right; the sort's own `_lead` removed |
| 12 | spin channels only partly closed: `record`, `datafile`, `stages.products_of`, `run_words`, `tbtnc` spelled `up`/`down`/`unpolarized` and `polarized`/`non-polarized`; `conductance.SPIN_CHANNELS` a second name (home-A 7, home-B 5) | fixed — `runfiles.CHANNEL_UP/DOWN/UNPOLARIZED`, `SPIN_POLARIZED/NON_POLARIZED`, the tables built from them; every reader asks them; the alias removed |
| 13 | "a pair's pseudopotentials come from `psml_lib`" decided by three kind comparisons (home-A 8) | fixed — `asked_of_the_person(kind)` takes the kind word; prep's source, the seed's note, describe's answer, the check and the panel ask it |
| 14 | `continuation` typed `"scf"`; `record` defaulted `STAGE_FACT.get(name, "scf")`; `record.TREATMENT_NOTE` keyed `"single-bias"`; `compose` typed the axis kind `"transport"`; `run_words.SCF/PRODUCT` second names; `reuse.__all__` still listed `point_key`; the `kind` ClassVar's note (home-A 9, home-B 8, 9, notes) | fixed — `identity.COMPUTES_SCF/PRODUCT`; `STAGE_FACT[name]`; `task.SINGLE_BIAS`; `structure.AXIS_TRANSPORT`; the aliases removed; the export list and note corrected |
| 15 | the status line opened the calculation a second time (A16) (home-B 4) | fixed — the `scratch_status` part is handed the calculation the status verb opened (`runstatus._scratch_line`, `gf_files.status_line(calc, stage, …)`); architecture.md's line |
| 16 | the re-laid copy's size summed two filters (home-B 10) | fixed — the run's own files counted by the copy filter, as they are copied by it (`_copied_files`); a point's by `_measured` |
| 17 | `siesta.engine_input.keyword_of` a second door for an item's keyword, by `anchor` where the rule is `writes` (home-C 3) | fixed — `siesta.layout.keyword` (the catalogue's `writes`, as `pyscf.layout.keyword`), asked by the bench marks, the GF estimate and the record's temperature; `keyword_of` and `siesta.input._bench_keyword` removed |
| 18 | `siesta.engine_input` types `TS.Elecs.Out-of-core`, `TS.Voltage`, `TS.Contours.nEq.Fermi.Cutoff` (home-B 11) | left as is — the module reads a deck as SIESTA does, its keywords SIESTA's own source facts (its note); `siesta.layout` imports it, so it cannot ask the writer's door |
| 19 | the carried PySCF attributes not all closed: `mf.chkfile` in both decks, `_mf_relax.newton()`, `_NOT_A_SETTING`'s anchor, the SOSCF print line's `max_cycle`, `{mf}.conv_tol`/`conv_tol_grad` (home-C 1) | fixed — `layout.keyword`, and `layout.keyword_on(name, solver)` for another solver object; `_NOT_A_SETTING` by item name. Visible: the SOSCF print line says `mf.max_cycle` (accepted, the changelog); every other rendered PySCF deck line byte-identical but the header's `gmax` unit (compared on both trees) |
| 20 | the PySCF header's note claimed every target by keyword and unit (tests 4, home-C 2) | fixed — the note says what the line does |
| 21 | the header row cited `pyscf.md` § 3, which stated no such rule (tests 5, home-C 5) | fixed — `pyscf.md` § 3 *Optimizer* states the header's line; template.md § 5's `unit` row names it |
| 22 | one refused override emptied the ladder: the stepless warning became a range warning, the loosening, recompute and group findings vanished (home-C 4, corr-C 1) | fixed — the ladder checks run again; the loosening check skips the refused (stage, item) pairs, as the range check does (`template.md` § 5.3: *so does any check on what is built from it*); its own test, failing on 2.77.24 (500) |
| 23 | the text-override case sat in a test about an item its rung does not read (tests 6) | fixed — its own test, `test_a_refused_value_draws_its_refusal_alone_and_the_ladder_is_still_judged`, citing § 5.4 and § 5.3 |
| 24 | the steps file's `derived from SIESTA's output` untested (tests 7) | fixed — the SIESTA vibration e2e's export test asserts it; datafiles.md § 9.5's `source` in the table's words |
| 25 | the engine table read inside a bundle reached only by e2e (tests 8); and not found when the bundle stands on the path as `./mb_vibration.pyz` (corr-C 2) | fixed — `persist.read_shipped(loader, module_file, path)`, the one read of a data file shipped with a module (the engine table, the chat card's layout), naming a file beside the module as the loader spells its bundle's path; a fast row per bundle carrying `ncdata`, bare and `./`, failing on 2.77.24 |
| 26 | `ncdata`'s import note; `read_optical_data`'s "the payload it was written from" (corr-C 3) | fixed |
| 27 | engine names a person sees outside data files: the Raman refusal's `pyscf`, the transport file's typed "TranSIESTA and TBtrans" (home-C 6) | fixed — the engine table's and the kind table's words |
| 28 | the spectra page's Engine row shows `siesta 5.4.2` (home-C 6) | later — R3d1b (the spectra page's wire carries no engine words yet) |
| 29 | architecture.md § 2.1 on `identity`, § 3.3 `TRAVELS_DATA`; run-reports.md § 4.1c (home-B notes, home-C 5) | fixed |

Tally: 29 grouped findings (52 reports) = 26 fixed + 1 accepted + 1 later + 1 left as is.

## Round 2 — four new readers (correctness ×2, one-home and notes-and-tests across the whole change)

| # | finding (reader) | verdict |
|---|---|---|
| 1 | a text value on a transmission-window item still crashed `preflight` — the kind's description check read the refused value (a Save 500); the same on 2.77.24 (r2-corr-B 1) | fixed — `description_checks` handed the refused pairs; the window check stands aside on a rung holding a refused value it reads (`_WINDOW_READS`); architecture § 3.3's sentence; rows, failing on 2.77.24 |
| 2 | `relax_steps: false` drew its refusal and the *relaxes nothing* warning (r2-corr-B 2) | fixed — the stepless check handed the refused pairs; a row, failing on 2.77.24 |
| 3 | the pair seed note's command (round 1 #8) had no test (r2-tests 1) | fixed — the e2e pair case runs through `` `{TYPED} describe --psml-lib DIR` `` |
| 4 | the bias-range warning's list (`6 V, 7 V`, was `6, 7 V`) untested (r2-tests 2) | fixed — the e2e row lists two voltages outside the range |
| 5 | the accepted record-before-prep change has no test (r2-tests 3) | left as is — its input, a cited frame set before prep, only an engine run makes; the door's pre-prep answer is pinned by the decks e2e's `test_a_run_of_points_is_told_from_its_citation_before_any_prep`, the record asks the same door |
| 6 | the in-package data-file row is subsumed by the bundle row (r2-tests 4) | fixed — retired (no mutant failed it alone) |
| 7 | the ladder row's docstring said "between two other stages" (r2-tests 5) | fixed |
| 8 | architecture § 3.2 lacked `units.volts_words`; the Extension row's stand-in task unexplained (r2-tests 6, r2-home 7) | fixed — the row; one sentence in the test |
| 9 | `tbtnc.point_dos` typed `"polarized"`; `record._iv` typed `current_up_a`/`current_down_a`; the TBtrans parsers typed the channel names (r2-corr-A 1, r2-home 3) | fixed — `runfiles.SPIN_POLARIZED`, `CHANNEL_*` in `tbtnc`, `record`, `siesta_grammar.tbt_passes`, `tbtrans_nc`, `tbtrans.CHANNEL_WORDS` |
| 10 | the reuse decision before an extension's first prep asked `frames_had` 11–17 times, each re-reading the finished junction (r2-corr-A, r2-home 1) | fixed — `gather.ReadOnce.frames()` asks the one door once for all its points; the gather, `_decided`, `_member`, `_forecast`, `_rendered`, `_laid_here` take it from there |
| 11 | `collect_record` before a first prep asks `frames_had` per call (~15 sidecar reads) (r2-home 1) | left as is — a calculation never prepared, each read one sidecar; the answer one door's |
| 12 | the schema route's `source.name` for a record stating nothing carries the file's name, where 2.77.24 sent `""` (r2-corr-A) | accepted — the page draws no name for `none`; the name is the file read |
| 13 | the electronic temperature read twice: `record.temperature_k_of` and `gf_files._Deck` (r2-home 2) | fixed — `_Deck` asks `temperature_k_of`; same answers on both trees |
| 14 | `siesta.input` typed `MD.Steps`, `MD.MaxDispl` beside `siesta.layout.keyword` (r2-home 4) | fixed — `keyword("relax_steps")`, `keyword("relax_max_displ")` |
| 15 | the copy helpers' notes claimed more than they did (r2-home 5) | fixed — each says which copier asks it |
| 16 | the transport axis and kind typed: `cell.py:351`, `validation/sidecar`, `validation/transport`'s axes off it, `wizard`'s lateral vectors (r2-home 6) | fixed — `AXIS_TRANSPORT`, `TRANSPORT_AXIS`. `cell.classify_seam`'s `c` is a slab's generic seam axis, left; `modify.py`'s electrode builder's kinds triple — later, R3e (builders) |
| 17 | `frame_average.summary_lines`' aligned column `0.200 V` beside `0.2 V` (r2-home 8) | fixed — `volts_words`' note says every sentence; an aligned table column keeps its width (architecture § 3.2's row says so) |
| 18 | `siesta/engine_input.py`'s only change a re-wrapped `__all__` (r2-home 9) | fixed — restored, the file unchanged |
| 19 | transport.md § 2a.11's doors lacked `run_folder_at`; the spin words' owner unstated (r2-home 7) | fixed — § 2a.11; architecture § 2.1's `runfiles` line |

Tally: 19 grouped findings (24 reports) = 15 fixed + 1 accepted + 3 left as is (5, 11, and 16's `classify_seam`), one part later (16's `modify.py`).

The whole fast suite after round 2's fixes: 6505 passed, 0 failed.

## The two independent full-text readers, and the discipline reader

| # | finding (reader) | verdict |
|---|---|---|
| D1 | `_WINDOW_READS` re-types `window.WINDOW_ITEMS` beside it (discipline 1) | fixed — `_window_reads()`: `window.WINDOW_ITEMS` and the temperature |
| D2 | the `relax_steps: false` row is a state the road cannot make (the setting door reads `false` as the text `'false'`), on a hand-typed task (discipline 2) | fixed — the row and its optimization branch retired; the guard kept (§ 5.3's rule); the window rows stand, failing on 2.77.24 |
| D3 | `engine_table` travels but keeps `persist`/`runfiles` imports inside functions, and its `kind_table` users import one way (discipline 3) | fixed — `persist` and `runfiles` imported both ways in the head; the header names the four functions that are the package's alone |
| D4 | `energies.deck_window` imports `units` one way inside the function, though it travels (discipline 4) | fixed |
| D5 | `structure.AXIS_TRANSPORT` by position and a near-anagram of `TRANSPORT_AXIS`; `task.LOW_BIAS` drops "approximation" (discipline 5) | fixed — `structure.AXIS_KIND_TRANSPORT = "transport"`, `ACROSS_TRANSPORT` and `AXIS_KINDS` built from the words; `task.LOW_BIAS_APPROXIMATION` |
| D6 | `siesta.input._RELAXES` hand-lists `relax_type`'s choices (discipline 6) | fixed — `relax_type`'s choices, `none` aside |
| D7 | negative notes that are not conditional guards (`siesta/layout.py`, `transport/sort.py`, `transport/stages.py`) (discipline 7) | fixed |
| D8 | template.md § 5's `unit` row keeps "— plan F27" (discipline 8) | fixed |
| D9 | the changelog row is not in the tree yet (discipline, mechanics) | the release's last step — lands with the commit |
| A1 | = D1 (full-A 1) | fixed (with D1) |
| A2 | `compose._with_transport` imports `TRANSPORT_AXIS` inside, beside the module's `_TRANSPORT_AXIS` (full-A 2) | fixed |
| A3 | `transport/points.py`: a function between two import lines (full-A 3) | fixed |
| B1 | = D1 (full-B 1) | fixed (with D1) |
| B2 | `cell.py`'s and `task.py`'s module notes name their imports, now without `kmesh` / `units` (full-B 2) | fixed |
| B3 | = A3 (full-B 3) | fixed (with A3) |
| B4 | `tbtnc.point_dos` imports `TBT_SPIN_CHANNELS` twice (full-B 4) | fixed |

Both full-text readers: nothing breaks molbuilder's own workflow (reader A: every path traced gives 2.77.24's answer bar the intended changes; reader B: 684 preflight cases, 33 crashes on 2.77.24, none now). Tally: 15 reports, grouped 12 = 11 fixed + 1 (the changelog) at the commit. Also: the rung notes laid out one name per text (the same words, compared on both trees). The whole fast suite after these fixes: 6504 passed, 0 failed.

## The reader of the last fixes

| # | finding | verdict |
|---|---|---|
| 1 | `engine_table`'s header names four package-only functions, not the two that ask them (`engine_of_config`, `read_deck`) | fixed |
| 2 | `compose.py` imports from `..structure` on two lines | fixed |
| 3 | the window items are read by position (`emin, emax, n`), and `WINDOW_ITEMS`' note does not say the order | fixed — the note says the order every reader unpacks |
| 4 | the changelog draft's deletions name `ReadOnce.frames` and `_laid_here(frames=)`, which round 2 made again | fixed in the draft |

## The e2e run, last

The release's e2e batch — the real junction, the transport decks, the SIESTA and PySCF vibrations, the
PySCF relaxation outcome, the monitor's live run, the binary's keywords: 330 passed, 1 failed. The one
was the new seed-note row's own setup: it prepared its copy from the test's body, off the road the
module's fixture sets up (the projects tree pointed elsewhere) — fixed: the copy is made by the fixture,
as its other copies are. The real-junction module again: 229 passed, 0 failed.
