# fix-W2 — test_build_e2e.py, test_this_machine_e2e.py, test_chemistry_card_e2e.py, test_spectrum_form_e2e.py

Read first: fix_brief.md, W2.md, verified.md (W2), testing.md § 3b and § 5, all four files, form-schema.md §§ 1.1–1.3 and § 1a,
chemistry-correctness.md §§ 2.5 and 2a.1a, web-api.md (the analyze row), handover-procedure.md §§ 2–3, spectra.md § 5,
this-machine.md § 3. Code read: form-schema.js (`_followActiveWhen`, `setValues`, the label/lock-reason drawing),
structure-optimization/viewer.js (`PANEL_TAG`, form-state save/restore, Send), lib/task-handover.js, workspace/dispatcher.js
(`workspaceId`), blueprints/_shared.py (`catalogue_to_form_schema`, `_item_to_field`, `for_writing`), blueprints/build.py
(hand-over answer, analyze `chosen`), electronic_state.py (`_method`, `said`), cell.check, describe.refuse_described,
catalogue rows relax_type/relax_steps/relax_force_tol/relax_max_displ/solvent/solvent_method, conftest
`_isolated_workspace_store`.

## 1. Per test touched

| test (file:line after edit) | action | contract § | what it now guards, and one code change that would turn it red |
|---|---|---|---|
| fixture `_open_build` (build:125, prune at :152) | fixed | workspace.md § 4 | prunes BOTH tags the page saves under, through the same route: `ws-structure-opt-panel` (the tab's `structure-opt:panel`, where the structure is kept) and `ws-structure-opt` (MolView's bare tag). Comment corrected (the old reason "one workspace shared by the tests" is untrue; conftest `_isolated_workspace_store` already gives each test its own store). This is setup, so no single change turns it red. It stops being a no-op if the conftest isolation goes away |
| `test_siesta_form_renders_fields_after_init` (was build:171) | retired | — | duplicate of :176 plus a vacuous "Loading from schema" check. Its JS-error check moved into :176 |
| `test_no_field_renders_loose_outside_a_card` [siesta, pyscf] (build:176) | fixed (took over the JS-error check) | form-schema.md § 1a rule 1 | every fieldset sits inside a card, and the tab boots with no pageerror or console.error. Red if: a throw in viewer.js init(), or a field whose `card` is not in `cards` |
| `TestTabSwitching.test_clicking_siesta_tab_returns_to_siesta_panel` (was build:263) | retired | — | weak duplicate: the SIESTA panel is already visible before any click. The strip is covered by test_tab_strip_js.py, and the PySCF panel is reached by :1095 and chemistry:162 |
| `test_sidebar_load_updates_info_atoms` (was build:288) | retired | — | duplicate of the first half of :339 (commit → 3) |
| module docstring (build:1-27) | docstring | — | drops the placeholder and tab-switching claims and the stale "guard fails CLOSED". Adds the lock test |
| `test_a_finding_about_the_mesh_sits_beside_the_mesh` (build:573) | docstring | form-schema.md § 1.1 (refusal beside the field) | the archived plan citation "plan § 5w K7" is replaced by the live § |
| `TestSendToTaskSetup` docstring (build:656) | docstring | describe.refuse_described | "fails CLOSED on a read error" is stale: the guard reads nothing, a file is there or it is not |
| `test_send_writes_the_handover_into_the_selected_folder` (build:764) | docstring (NO GOAL) | handover-procedure.md § 2 steps 1-2, § 3 | GOAL and silent failure stated. Assertions unchanged. Red if: the Send writes nothing, or the page navigates away |
| `test_send_writes_and_STAYS` (build:807, assert :836) | fixed (vacuous → positive) | handover-procedure.md § 2.1 | the absence check "cell was checked" (text found nowhere) is replaced by the hand-over door's answer the page received, `notices == []`. This shows the stated-cell fixture reaches the clean branch (checked by reading cell.check: an 18 Å stated box with atoms centred gives no Issue). Red if: the fixture or the gate starts producing a notice for a stated wide box (e.g. `cell.vacuum_defaulted` raised under a manual cell) |
| `test_a_field_waits_on_the_field_the_catalogue_names` (build:1221) | **added** (OWED #1) | form-schema.md § 1.3 (quoted: the two Structure-optimization locks; "greyed out while any named field holds another value … and follows each change"), § 1.1 (a blank applies the kind's recommendation) | `p-relax-steps/-force-tol/-max-displ` are live at load (blank = CG), greyed with the reason line under `none`, and live under Broyden. `py-solvent-method` is greyed with no solvent and live once "water" is typed. After a reload the restore writes the held values and the greying follows: steps greyed, PCM live. No JS errors. Red if: the `"set"` branch is dropped or "set" is compared as a value; a blank is read as null rather than the default; the lock is applied at render only; `setValues` stops dispatching input/change; or the catalogue's `active_when` is removed |
| `test_the_card_answers_for_each_form_and_fills_nothing` (chemistry:108, assert :137) | fixed (WRONG DETAIL) | chemistry-correctness.md § 2a.1a end; form-schema.md § 1.1; web-api analyze row | PySCF's blank Method now reads "DFT — default", not "stated". The "Fe is an open-d metal … verify it against experiment" assertions are kept (§ 2.5). Red if: `_method` ignores `chosen` (says "stated"), or the analyze route stops passing `chosen` |
| chemistry module docstring (chemistry:4) | docstring | chemistry-correctness.md § 2a.5 | "plan § 5s.3, P3's done-condition" is replaced by the live § |
| `test_a_periodic_structure_moves_the_spectrum_strip_to_siesta` (chemistry:212) | docstring (WRONG DETAIL) | spectra.md § 5 step 4 | the reason was "PySCF's gate refuses". It now reads: PySCF computes a repeating structure as an isolated cluster, a note and not a refusal |
| `test_saving_writes_the_file_the_page_tells_you_to_write` (this-machine:228; message :247) | docstring/message | this-machine.md § 3, § 3.1 | `<config dir>/notify` → `<config dir>/secrets/notify`. Assertions unchanged (they already go through `default_notify_path()`) |
| comment above it (this-machine:222) | citation | this-machine.md § 3 | what a save clears is in § 3, not § 3.1 |
| `test_the_selection_locks_what_it_would_ignore` (spectrum:26, line 35) | citation | form-schema.md § 1.1 | "plan K7" is replaced by the live § |

## 2. SHARED (not made)
- `_ready_for_a_pick` and `_register_tmp_as_picker_root` are imported by test_spectrum_form_e2e.py from another test module
  (test_chemistry_card_e2e). They belong in `tests/support/`.
- The picker-root patch is copied in build and chemistry (`_register_tmp_as_picker_root`). The one door is the conftest
  `isolated_projects_root` (testing.md § 2a). One helper is needed, or the fixtures switch over (F9).
- Contract drift from W2.md, for the doc owner: handover-procedure § 2.2's "every element" (already reads "species"
  today, OK); chemistry-correctness § 2a.3's PySCF `free` cell vs ES6 / the catalogue `not_offered`.
- `_open_build`'s prune repeats the conftest `_isolated_workspace_store`. It was kept as instructed. If F9 rules the prune
  redundant, delete it.

## 3. F9 (left, one line each)
- build:1074 `test_send_refuses_a_folder_that_is_already_described`: a hand-typed `task.json`, and NOT E2E (move to a tier-1 route row).
- build `_open_on` plants `molbuilder.current_dir` in sessionStorage instead of clicking in the sidebar.
- build Send fixtures write `proj/opt/<calc>`, and `opt` is not a canonical topic (`optimization`).
- build/chemistry `_register_tmp_as_picker_root` patches `Capabilities.file_picker_roots` (handcrafted).
- build :257/:306/:339/:415/:494… drive `publishCommit`/`setShared` through the API rather than by row clicks.
- build :216 engine-key badges: WEAK (`count() > 0`). It should be assert-per-field or be retired.
- build :608 PySCF half: WEAK (water may make PySCF answer `[]`). Load Fe so both engines answer with findings.
- build :494 + :573: one test with two table rows. :764 + :807: one Send table (no-cell / stated-cell rows).
- build :306 setShared-alone: likely subsumed by :959. Confirm with a mutant before retiring.
- build :891 text pins (`'folder covers all 2 species'`, `'pseudopotentials are not settled'`). It should assert the `where`/severity instead.
- this-machine :189/:228/:283: one test with a row per credential shape. The 0600-and-location half of :228 is tier 1's (test_notify_setup_api).
- chemistry :108 offered-options half (lines ~152-159): NOT E2E, belongs in test_what_a_kind_offers.py.
- chemistry :212: move it to test_spectrum_form_e2e.py.
- spectrum :201: WEAK, a proxy through the live check. OWED #2 (the Spectrum Send writing `calculation: "vibration"`) was
  not added because no test presses the Spectrum Send.
- OWED #3 (a labelled species survives a Send): skipped. molview.md does not state how a labelled species is drawn.

## 4. Checks
- pyflakes on the four files: clean (rc 0, no output).
- collect-only: 36 tests collected (build 22: 24 before, −3 retired, +1 added; this-machine 5; chemistry 4; spectrum 5). No errors.
