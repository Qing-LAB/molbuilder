# rev-R1 — `tests/test_transport_on_a_real_junction_e2e.py` revisions + 2.65.4 product code

Read: the brief; fix-T.md, T1.md, T2.md; `git diff HEAD` of the test file (every hunk) and of `molbuilder/`
(transport/reuse.py, rung.py, gather.py, describe.py, describe.py); the test file's fixtures (1–940), the
`reused` fixture (4254–4600) and every changed test whole; product modules read whole: `transport/reuse.py`
(1–1810), `transport/rung.py`, `transport/gather.py`, `transport/describe.py`, `describe.py`. Followed into:
`jobset/prep.py` (520–700, 2700–2780: PLANNED scope, step 4a `decide`, `plan_stage` per member),
`jobset/planned.py` (whole), `pseudos.py` (PLANNED, `psml_sources`, `_psml_paths`), `siesta/seam.py`
(`pseudo_dir`, `siesta_provide_pseudos`, `screen_pseudos`), `siesta/input.copy_pseudopotentials`,
`script_emit.render_deck` (the gate), `validation/__init__.validate`, `validation/siesta.py`
(`_check_siesta_pseudo_coverage`, `validate_siesta`), `transport/citation_defaults.py` (template's `psml_lib`),
`transport/stages.py` (`frames_of`, `rung_points`, `Point`), `transport/compose.py` (`read_provenance`,
`load_compose_record`, `write_compose_record`, `refuse_inside_citation`), `jobset/_cli.py` (describe printing,
`_resolve_bundle`), `jobset/commands.py`, `web/blueprints/transport.py` (describe route, `_describe_reused`),
`jobset/runstatus.py` (`StageStatus`), `jobset/continuation.point_rows`, `runrecord`, `runwrap`
(`_effective_parameters_block`), `parse/dirs/setup.setup_rows`, `siesta_grammar` stop meanings, `structure.py`
(`XYZ_DECIMALS`, frames invariant). Contract: transport.md § 1.1 *The first prep*, § 2a.16 whole (as it reads
now, and its 2.65.4 diff), § 3.1, § 6.3; job-system § 5.1; runs-of-points § 3; web-api describe row.
Nothing run.

## Findings, ranked

### Will fail (product defects the batch meets)

**1. An extension's reuse decision renders its decks before any pseudopotential is in its folder — the settings
gate refuses, so every extension's first prep is refused and its describe has no manifest.**
- Path: `prep task` step 4a calls `reuse.decide` (jobset/prep.py:2741) BEFORE any member is planned; describe
  calls it through `_manifest_of` (transport/describe.py:427, 458–472). `_decided` → `_member` →
  `gather.ran_otherwise` (reuse.py:1310) → `ReadOnce.deck_now` → `rung_deck_now` → `render_deck(...,
  dest_dir=base)` (gather.py:444–447) → the settings gate (`script_emit.render_deck` 1356–1381: `_validate`,
  then `_report`, which raises `ValidationError` on an error) → `validate_siesta` →
  `_check_siesta_pseudo_coverage` (validation/siesta.py:51–92): `psml_sources(['H'], dest_dir=<extension>)`
  finds nothing (`pseudos/` and the root are empty — describe now writes three files), and the template's
  `psml_lib` is unset for a run-cited finished calculation (citation_defaults.py:294–298: a cited run's
  `psml_lib` is *cited*, no value; no catalogue default/recommendation) → **"error"** because `dest_dir` is set
  → `user_error_as_prep` → `PrepError`. `decide` catches only `LaunchRecordError` (reuse.py:835–838), so the
  prep is refused; `_manifest_of` turns it into `manifest None` + *warn: what it will reuse is not known yet*.
- `reuse.carry_pseudos` runs only in the rung's `provide_data` (rung.py:461–466), i.e. after step 4a. In
  2.65.3 `write_reusing` copied `H.psml` into the new folder at describe (`git diff`: removed
  `psml_lib=finished.base / PSEUDO_DIRNAME` and `pseudo_species=…`), which is what let the decision render.
- Contract: § 2a.16 *Extending it*: "What the extension will reuse is one answer (`reuse.decide`) … said once
  it is described — on the Transport tab beside Describe's answer, by `molbuilder describe --reuses`, on Task
  setup's Preview and by `prep task`"; *Describing it*: "Until that prep, every reader asks the one door for
  the junction the calculation has or will have" — the pseudopotentials it will have got no door.
- Effect on the batch: `reused` fixture `_taken(same, "prep", …)` (4319) asserts exit 0 → the fixture errors →
  every test taking `reused` errors (4627, 4675×5, 4811, 4840, 4899, the window/finer/low-bias tests, 5110,
  5267, 5285, the browser test). Also the new `left out` asks (4572–4577; asserted 5206–5220) and the page's
  `answer["manifest"]` get `None`. Only an extension of a pair-cited calculation (`fone`, `psml_lib` set) escapes.
- Fix: run the decision with the finished calculation's pseudopotentials planned in — e.g. `_decided`, when
  the folder holds no record, plans `carry_pseudos` into a scratch `Plan` and decides under
  `pseudos.PLANNED.set(view)` (both roads, prep step 4a and `_manifest_of`); say it in § 2a.16 *Describing it*.

**2. Before the first prep, `_member` and `_laid_here` count the NEW calculation's frames from disk (1) — a
frame-set extension's device and transmission are judged as one geometry and copied in whole.**
- reuse.py:1255 `points = rung_points(calc.task, stage, frames=frames_of(calc.base))` and reuse.py:1408
  (`_laid_here`, same) — `stages.frames_of` reads `slot-provenance.json` (stages.py:224–232), absent until the
  first prep's plan is carried out → 1. `_decided` itself was moved to `once.frames()` (reuse.py:883, 912) and
  `_rendered` (975); these two were missed.
- On `framed` (O single voltage, one geometry; framed a 3-frame set, single voltage): `points = ()`,
  `theirs = ()` → the run-once branch → `ran_otherwise(once, "device", O's run, point=None)` renders the
  3-frame junction at point `None` = frame 0 (`Structure.positions` is `frames[0]`, structure.py:859–869; no
  deck writer reads `frames`) = O's deck → `differs == []` → device copied in; `_laid_here` → `{}` → copied
  whole; transmission likewise. Frames 1–2 are never computed, while the rung builder writes 3-frame point
  folders (rung.py:450–451). In 2.65.3 describe had written the record, so `frames_of` was 3 and the device
  was *prepared here* (`points and not theirs`).
- Contract: § 2a.16 *A frame set of the same junction*: "only the first frame can be reused, and every other
  frame is computed"; *Describing it*: "every reader asks the one door … its frame count among them".
- Test: 4840 (`read_copied_from(run) is None`, a point per frame, frame 0 taken over) fails once #1 is fixed;
  the describe/bare-prep manifest of `framed` (and the browser test's plan) says groups 2, 3 copied in.
- Fix: pass the decision's `once.frames()` into `_member` and `_laid_here` (the one door), never `frames_of`
  on a calculation whose record may not be written yet.

### Wrong expectation / weak

**3. The 2.65.4 pins in `test_a_question_the_finished_results_answer_is_copied_in` (4717–4734) cannot fail on
the failures their docstring names, and their tolerance is not the contract's.**
- "a first prep that composes the junction afresh from its citation": composing T's citation again gives the
  same atoms at the same places (compose is deterministic) — survives; "fetches pseudopotentials the finished
  runs never read": without `carry_pseudos`, `provide_pseudos` copies J's `pseudos/`, the same library file T
  copied — byte-equal, survives. Only "no record written" / "no pseudos" would turn it red.
- `np.allclose(ours.positions, ran.positions, atol=1e-9)` keeps numpy's default `rtol=1e-5` → ~1e-4 Å on these
  coordinates (up to ~22 Å) — neither "atom for atom" (comment 4719) nor the contract's `SAME_PLACE_ANG` 10⁻⁶ Å
  (§ 2a.16 table); elements and frame-0 positions only (no cell, labels, order record).
- For `first`/`fone` the extension's junction is its citation composed and checked by `frame_set_of`
  (reuse.py:209–211), not "the junction the finished calculation ran" (comment 4718).
- Fix: compare through `compose.junction_difference` (the one comparison) with `rtol=0`; state the failure the
  fixture can show (no record / no `pseudos/` after the first prep), or say carry_pseudos is unpinned.

**4. 5316–5317: `not (page / "pseudos").exists()` is vacuous** — no describe ever made `pseudos/` (2.65.3 copied
`H.psml` to the folder's root); the `written == three files` line (5314–5315) carries the check. Fix: assert
`not (page / "H.psml").exists()`, or drop the `"pseudos"` row.

**5. 1067 docstring quotes job-system § 5.1 trimmed**: § 5.1 says "writes the description — the template,
`task.json`, `job-request.json` **and the data files**"; the exactly-three expectation rests on transport § 1.1
/ § 3.1 (data files at prep), which the docstring also cites — quote it whole and say why none travel here.

**6. 1620–1628 reads the run's setup rows below their parser**: `read_parameters_fence(run_of(...).session_log
.read_text())` — the record's door for "what the engine will read" is `parse/dirs/setup.setup_rows` (it reads
the same fence from `wrapper_text`); parse § 5d.3a states the record's rows. The third assertion
(`all(reads(...))`) restates how runwrap builds the block (`declarations` by `template.reads`). Low.

### Product (no batch failure)

**7. The extension branch of the one door skips the launched-stage rule.** rung.py:282 takes the finished
junction whenever `read_provenance(base) is None`, with no `launched_stage` check; `composed_for_prep` then
writes it (gather.py:359–362) and the gather's `composed_junction` reads it — never `no_usable_record`. § 1.1:
"A prep that finds no usable record while a stage is launched is refused and writes nothing". An extension
whose `slot-provenance.json` is gone after its groups were copied in/launched gets the finished junction written
anew. Fix: take the finished junction only while no stage is launched; otherwise `no_usable_record`.

**8. Not "exactly once per prep".** `transport_rung_of` calls `composed_for_prep` per member (prep plans every
member into one plan, prep.py:2758–2770), and `read_provenance` reads the disk, not the plan — so a first
`prep task --group 1 --group 2 --group 3` plans the four record files 5× and runs `finished_of` +
`finished_junction` 5× (for a frame set, `compose_junction` + `frame_set_refusal` each time), plus 2× in
`_decided` (`junction_record`, then `ReadOnce.junction` again) and 3× per describe. Same bytes, so harmless to
the plan's identity; the same per-rung pattern pre-exists for a citation's first prep. Fix: compose once per
prep (seed `ReadOnce._junction` from `_decided`'s `composed`; hand the prep's junction to the rungs).

**9. Readers of a finished calculation's junction bypass the one door.** `finished_junction`/`frame_set_of`
(reuse.py:145, 213) and `web/blueprints/transport.py:405` (`_describe_reused`) read `load_compose_record` /
`recorded_junction` directly: an extension of a described-not-yet-prepared extension reads "no record" (warn,
composes afresh), and the tab's picker draws no junction for it. § 2a.16: "every reader asks the one door". Low.

**10. `write_reusing` lets `SortError` / `PermutationError` / `DeckRecordUnreadable` escape describe.**
`finished_junction` maps only `ComposeError` → `ReuseError` (reuse.py:217–218); `transport.describe` catches
`ReuseError`/`DescribeError` (describe.py:418–423) → CLI traceback / HTTP 500. Prep catches them
(`junction_record`'s `compose_refusals()`). Pre-existing since 2.65.3. Fix: catch `compose_refusals()` there.

### Docs / docstrings

**11. transport.md 4123–4128** (*A frame set of the same junction*): "The junction is the set's, every frame,
recorded in the new folder **when it is described** (`compose.write_compose_record`)" — contradicts 2.65.4's
*Describing it* (first prep). Fix: "…recorded by its first prep".

**12. reuse.py:6–18** (module docstring): describe writes "its composed junction … and its pseudopotentials …
or, in place of its junction, a frame set … composed here" — stale.

**13. web/blueprints/transport.py:670–676** (describe route docstring: "it writes … its junction — its own, or a
frame set of it") and **803–805** ("a frame set written in its junction's place") — stale.

**14. describe.py:3–7**: "and the composed junction's writer (`transport.compose`) for a calculation that reuses
a finished one" — the import was removed.

**15. transport.md § 1.1 *The first prep* (697–706)** says a prep with no usable record composes afresh; it does
not name the extension's first prep, which takes the finished record (§ 2a.16) — add the cross-reference; and
`_cli.py` `--reuses` help ("starts as its description and its composed junction, whole") reads as written at
describe. Low.

## Checked and right
- Edited-pair and frame-0 refusals asked again at prep: `junction_record` → `_finished_junction` →
  `finished_junction` (`load_compose_record` pin → `ComposeError` → `ReuseError`; `frame_set_refusal` →
  `ReuseError`) → `PrepError` (rung.py:303–307); from step 4a it propagates as the prep's refusal.
- `carry_pseudos` runs under `pseudos.PLANNED` (prep.py:520–529 wraps `_plan_calculation`, whose
  `rung.provide_data(plan)` is at 630–633); it and the later `provide_pseudos` / `siesta_provide_pseudos` compute
  `want` through PLANNED-aware `psml_sources` (`_is_file`), so the later steps see its copies and copy nothing.
- No caller still passes `junction=` to `write_description`; no `psml_lib=finished` left.
- New assertions that should pass on correct code (engine words unverified): 1067 (headline, three files, prep
  line ending `--bundle P/transport/<name>`; `noun` "transport calculation"; `_resolve_bundle` absolute under
  root); the refusal rows (`inside` words = `refuse_inside_citation` with `shown`; `another engine` =
  describe.py:221–226); the tab rows (rename → `Extension.refusal` first; `bias: []` → describe.py:274–276;
  `mixing_weight` → `shared_of` 138–143); `left out` (once #1 is fixed: `_copied_in == [1,2,3]`; `device:/
  transmission: takes over 0 V, 0.2 V from …; walks 0.4 V`); `continued_from` None (cold) / `04_device/run-2`
  (warm; `StageStatus.dir` is the stage folder's name, `RunRef` str is base-relative); point `detail` keys exist
  (`continuation.point_rows` 710) — whether they carry TBtrans's / SIESTA's words is unverified (product § 6.3).
- Fixture `junction`: `set_by_the_person(relax.bundle, RELAXED_WITH)` — the road's door, source `person`.
- Retirements: none lost a guard; the 5285 presence check was correctly reversed — but 2.65.4's
  `carry_pseudos` now has no pin that can fail (#3).

## Verdict
The test revisions mostly track the contract and their new words match the code, but the batch will not get far:
2.65.4 moved the pseudopotentials out of describe without giving the reuse decision — which runs before any rung
plans them — a way to see the files the first prep will take, so the SIESTA settings gate refuses every
run-cited extension's decision (#1), and the whole `reused` fixture errors at its first `prep`. Behind it, the
"has or will have" door was threaded through `_decided` but not `_member`/`_laid_here`, so a frame-set extension
would be copied in as one geometry (#2). Fix #1 and #2 in product code before the run; tighten #3–#4; the
launched-stage bypass (#7) and the stale docs (#11–#14) should go in the same release.
