# rev-R3: review of the e2e revisions (task setup, bench, build, chemistry card) and the method's `chosen`

Read whole: the 10 test files in scope, `molbuilder/electronic_state.py`, `web/blueprints/build.py` (`/api/structure/analyze`, `_analyze_response`), and `_shared.config_from_params`/`_item_to_field`. Read the page JS each new step drives: `task-setup/viewer.js` (`standingRows`, `runCard`, `runRow`, `renderCameOver`, `defaultText`), `lib/form-schema.js` (`renderField`, `makeSelect`, `_followActiveWhen`, `collectForm`, `setValues`), `structure-optimization/viewer.js` (`initFormsFromSchema`, `saveFormState`/`restoreFormState`, `PANEL_TAG`), `lib/workspace` `workspaceId`, `lib/chemistry.js`, `lib/inspectors/bench-summary.js` (`refresh`, `POLL_MS`), and `lib/task-handover.js`. Also read `cell.check`, the catalogue rows for every field the tests drive, and the contracts in the brief as they read now. Nothing was run.

## Ranked findings

### Will fail on correct code
None found. Every new browser step was traced to the code that answers it:
- **Run card row.** `mpi_np` is `run_card = true` (catalogue:846-848), so `standingRows()` includes it. `runRow` gives `data-kind="unstated"` because there is no value, no `execution.mpi_np`, no lane and an empty `defaultText` (no default, not in the template). The group card is first in the DOM and visible, and `_rows` reads only `:not([hidden])`.
- **What came over.** For an optimization, `_folder_kind` gives `cites=None, carries=[]`. The rows are Calculation, Run id, Engine, Structure, `Formula`=`H2` (`Structure.formula`) and `Atoms`=`2` (task.py:1022). None is `—`.
- **Greying test.**
  - The ids are `p-`/`py-` + name with dashes (`_item_to_field`), and the items are in the catalogue at 490/524/549/578/1662/1680. `solvent` is `type="str"`, so a text box and `fill` works. `none` and `Broyden` are offered for optimization.
  - The blank `relax_type` reads `f.default` = `CG` (`valueOf`), so the step fields are live at load.
  - The `.lock-reason` span is created from `active_words` (`_shared.py:818-823`) as a direct child.
  - After a reload, `restoreFormState` → `setValues` dispatches input/change, and `apply` follows them.
- **Workspace ids.** `workspaceId("structure-opt:panel")` = `ws-structure-opt-panel`.
- **Hand-over.** The answer always carries `notices` (build.py:1286). `cell.check` raises nothing for a typed 18 Å cell with no stated vacuum and the atoms inside.
- **Bench cadence.** `refresh()` calls `fetch` synchronously, before its first `await`, so `firing`=15000 is recorded. The poll uses `root.setTimeout`.

### Expectation from a contract that contradicts another contract
1. **(medium) Two words for "nobody chose this".** Locations: `electronic_state.py:302-303`, the new sentence in `chemistry-correctness.md` § 2a.1a, and the tests `test_chemistry_card_e2e.py:137` and `test_structure_analyze_endpoint.py:55-73`.
   - The card now says the blank Method is `default` (`Resolved(cfg.method, "default", "default")`).
   - `form-schema.md` § 1.1 says the words a source is said in "ride once, as the schema's `source_words` (`template.SOURCE_WORDS` … *not chosen* …), and every surface says them from there". `template.SOURCE_WORDS["default"]` is `"not chosen"`, and the Method field's own caption beside the card reads *not chosen · recommended DFT* (`blankWords` → `said(ctx,"default")`).
   - So one fact is shown on one page in two vocabularies, from two sources.
   - Fix (one line): say the `why` from `template.SOURCE_WORDS["default"]`, or reconcile § 2a.1a with § 1.1 so one word governs. The tests' strings then follow.

### Vacuous assertions
None found.
- The `unstated` row: fails if `standingRows` drops the `run_card` items, or if the row is born filled.
- The facts rows: fail on a slot row drawn by kind, or on the structure pair dropped.
- `15000 in armed`: fails on `POLL_MS` ≠ 15000; at 3000 the three polls arrive inside 10 s but none was armed by 15000.
- The greying states table: fails on the `"set"` branch dropped, a blank read as null, or a lock applied at render only.
- `said` = default/stated: fails if `_method` ignores `chosen`, or if the route treats `""` as chosen.

### Handcrafted input or planted state
None new.
- The bench `sweep` now goes through `conftest.write_machine_record` and the job request's run card (`support.road.write_request`, the editor's road already used across `test_task_setup_prep_e2e.py`). It replaces a hand-written `calc/environment.json`, so this is a net reduction.
- `Capabilities.file_picker_roots` is still patched, as listed for F9.

### Retirements
All safe. None removes the only guard on a contract rule.
- `test_the_k_grid_is_not_stored_as_text`: `[4,4,1] == "[4, 4, 1]"` is already False, so `test_a_typed_k_grid_lands_as_three_numbers` guards it.
- `test_siesta_form_renders_fields_after_init`: its JS-error check moved into `test_no_field_renders_loose_outside_a_card`.
- `TestTabSwitching`: the greying test switches pyscf → siesta and then `select_option`s a SIESTA control, which needs it visible.
- `test_sidebar_load_updates_info_atoms`: `test_a_second_file_replaces_the_first_in_one_visit` asserts commit → 3.
- `test_the_declared_cadence_is_the_trajectory_viewers`: absorbed as `__armedBy`.
- Bench `_sandbox`: conftest now isolates HOME and XDG; cwd `molbuilder.json` is never read (`runtime_config.machine_config_shadow`).

### Product defects (item 6)
2. **(low–medium) A second reader of "blank" in the analyze route** (`build.py:199`).
   - `chosen = {k … if v not in (None, "")}` restates the blank rule that `config_from_params` takes from `template.read_text`. That rule is `None`, or a string that is only whitespace (template.py:2904).
   - A whitespace-only `method` (unreachable from a select, reachable by any other caller) would be read as the catalogue's DFT and still be said `stated`.
   - Fix (one line): take `chosen` from the same door, for example `config_from_params` returning the names it set (its `kwargs`), or `read_text(item, v)[0] is not None`.
3. **Callers checked: no deck or prep path can get `default`.**
   - All 7 other callers omit `chosen`: siesta/input.py:679, siesta/seam.py:37, pyscf/input.py:209, pyscf/vibration_deck.py:98, validation/siesta.py:204, validation/__init__.py:238, transport/rung.py:199. `None` keeps `stated`.
   - Nothing else reads `method.source` except `chemistry.js:139`, where `!== "rule"` shows the row, as intended. `detection-chip.js` reads no method.
   - The Spectrum tab's parts (state/common/engine) and the Transport tab's locked fields reach `chosen` correctly. Transport is SIESTA, where the method comes from the rule.
   - Charge and spin have no catalogue default, so ignoring `chosen` for them is equivalent.

### Docs and residue
4. `chemistry-correctness.md:347` and `:368`: § 2a.1's class sketch still lists `source` as `"stated" · "implied" · "recorded" · "detected" · "rule"`, and the signature as `electronic_state(struct, cfg, *, kind)`. `default` and `chosen` are missing. Fix: add both.
5. `test_structure_analyze_endpoint.py:6-14`: the module docstring says "API-level, and only for what the pages cannot reach", and "The ANSWER is pinned where it is made and where it is read …". The new answer test at :55-73 pins an answer that the pages do reach. The test name says "the card", but it asks the route. Fix: amend the docstring and name.
6. `lib/inspectors/bench-summary.js:607-612`: `pollMs` is exported "so B4's cadence is checkable without a test sitting through it". After the retirement, nothing reads it (grep: no reader in tests or molbuilder). Fix: drop the export and its comment.
7. `test_task_setup_cell_types_e2e.py:203`: "shared by the four tests above" is now three. Fix: say "the tests above".

### Unverified (code read, not run)
- **Bench `sweep`.** The run card holds only `use_gpu`/`diag_algorithm`/`gpu_count`, with no `mpi_np`, `omp_threads` or `allocation`, on a workstation record written by `write_machine_record`. The bench lane asks for no shape and writes no header (prep.py:1961-1984: `bench_asks` drops the shape; `emit_sbatch=False`). Tier 1's `test_prep_bench_fold.calc` passes with the same card plus the shape, on a calculation-copy record of the same topology. Expected to pass; not proven.
- **JS-error assertions.** `test_no_field_renders_loose_outside_a_card` (both engines) and the greying test, the latter after a reload, now require zero `console.error`. No `console.error` was found on the boot path (viewer.js:463/480 fire only on a schema failure). Whether the headless page logs any resource error was not checked.

## Verdict
The revisions hold up against the code they drive. No new browser step names a selector, label, row state or id that the page JS does not produce for correct code, and each new assertion can fail on the defect it names. The run-card standing row, the What-came-over rows, the active_when greying across a reload, the hand-over's clean branch and the 15 s re-poll all trace through to the page JS and server code that answers them. The retirements leave no contract rule unguarded. In the product code, `chosen` is correctly kept to the card's route, and no deck or prep caller can be told `default`. Two things remain. The new word `default` competes with `form-schema.md` § 1.1's single source vocabulary, where the same fact is *not chosen*. And the route keeps its own reader of "blank" beside `read_text`. The rest is docs drift: § 2a.1's sketch, the endpoint file's docstring, the now-unread `pollMs`, and a stale count.
