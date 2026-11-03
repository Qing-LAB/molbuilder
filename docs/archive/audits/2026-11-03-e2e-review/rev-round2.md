# rev-round2 — fresh eyes on 2.65.4 option B + round-1 fixes, the card's method word, the catalogue's transport count, the JS, the tests in scope, and this round's docs

Read whole: `transport/reuse.py`, `transport/rung.py`, `transport/gather.py`, `transport/describe.py`,
`describe.py`, `electronic_state.py`, `web/blueprints/_shared.py` (`stated_params`, `config_from_params`),
`web/blueprints/build.py` (`_analyze_response`), the catalogue's `spin_treatment`/`unpaired_electrons` rows,
`lib/inspectors/trajectory.js`/`bench-summary.js`/`system-load-monitor.js`/`molview/history.js` (diffs and the
comments around them), the test blocks named, `git diff HEAD -- docs/`. Followed into: `jobset/prep.py` (PLANNED
scope 516–529; step 4a 2719–2775; `plan_stage` 2085–2210; `_launch_as_written`), `jobset/_cli.py` (bare-prep
offer 1290–1300; describe printing 407–436), `jobset/submit.py` (take-over 2660–2790), `jobset/planned.Plan`,
`pseudos.py` (`PLANNED`, `_is_file`, `where_bytes_are`, `psml_sources`, `_psml_paths`, `parse_psml_header`),
`siesta/seam.pseudo_dir`, `jobset/naming.open_container`, `siesta/input.copy_pseudopotentials`/`_spin_facts`,
`validation/siesta.py` (coverage, `_psml_files`), `validation/__init__.validate`, `validation/chemistry.py`
(`_spin_findings`), `config/siesta.count_must_float`, `template.offered`/`why_not_offered`/`read_text`/`SOURCE_WORDS`,
`transport/compose.py` (`load_compose_record`, `write_compose_record`, `read_provenance`, `junction_difference`,
`recorded_junction`), `transport/stages.py` (`frames_of`, `rung_points`, `point_folders`), `transport/points.py`,
`transport/citation_defaults.py`, `transport/deck.py` (spin section), `jobset/ready.launched_stage`,
`jobset/materialize.ever_launched`, `web/blueprints/transport.py` (describe route, `_describe_reused`),
`web/blueprints/system_load._gpu_snapshot`, `lib/chemistry.js` `_stateBlock`, `molview/model.js` history wiring,
`tests/support/road.until`, `tests/support/real_runs.stage_status`, `tests/test_inspector_registry_e2e.py`
(registry_answers, heading test). Contract: transport.md § 1.1, § 2a.16 (*Describing it*, *A frame set…*,
*On the page*), § 3.1 spin note, § 3459's row; chemistry-correctness § 2a.1/2a.1a, ES4–ES6, § 2a.3; template.md
§ 6.3a; science/overview.md § spin table; changelog 2.65.1. Nothing run.

## Findings, ranked

### Can fail on correct code (timing)

**1. `tests/test_sequence_of_groups_e2e.py:129` — the new wait accepts only `"running"`.**
`until(lambda: stage_status(left, "coarse").state == "running", within=300, …)`. If group 1's run has already
ended when the poll starts — the case the fixture's own `reached_last` read (l.116, asserted false at 244, 268,
298) exists to say, and also the gap between group 1 ending and group 2's launch record — the state is
`finished`, never `running`: `until` asserts after 300 s and the **module fixture errors**, so every test of the
module (the `whole` calculation's included) errors instead of the one designed assertion failing. Fix: wait until
the state has left `pending`/`queued` (running or ended), and let the later asserts say what happened.

### Weak / vacuous

**2. `tests/test_transport_on_a_real_junction_e2e.py:4740–4743` — the pseudopotential check cannot fail on the
failure the docstring names** ("pseudopotentials other than the files its runs read", l.4704–4706). In every
`_ANSWERED` row the fallback source the first prep would use without `reuse.carry_pseudos` — the cited run's
`pseudos/` (`rung.provide_pseudos`) or the template's library (`seam.provide_data`) — holds the same bytes the
finished calculation copied; only "nothing carried" turns it red (as a `FileNotFoundError`). R1 #3's second half
stands. Fix: say carry_pseudos's preference is unpinned by this fixture, keep the check as "present and equal".

**3. `tests/validation/test_chemistry.py:138–160`** — the docstring's silent failure is "the person's value never
reaching the deck", but only the gate's finding (info, "written as stated") is asserted; a `transport_parts`/
`configure` that floated a stated count (rung.py:213–217) would pass. Fails on the old catalogue (ES4 error) —
good. Fix: narrow the docstring to the gate, or also render the rung's spec through the rung builder (no engine)
and read `Spin.Fix`/`Spin.Total` back.

### Product

**4. (medium) A frame-set extension's frame count before its first prep is not the one door's outside the
reuse module.** `transport/points._declared_rung_points` (points.py:26–27) → `stages.frames_of` reads only the
disk record → **1** until the first prep writes it; 2.65.3 wrote the record at describe, so these read the set's
count. Readers that now say one geometry for a described, not-yet-prepared frame-set extension (e.g. `framed`):
Task setup's run card (`build._runs_points` → file list l.2251, `_folder_walks` l.2416 — the device and
transmission shown as one job with its own `.sbatch`, no walk, where they will walk 3 frames); `ready.at_launch`
previews (`gather.gather_sources` → `frames_of`); `prep.py:2153` (a copied-in per-point member's launch preview at
the first prep of an extension of a frame-set calculation). transport.md:709 ("until then the one read answers
that junction to every reader") and :3961–3963 ("every reader asks the one door … its frame count among them")
say otherwise. Within the reuse module it is right now (`_member`, `_laid_here`, `_decided`, `_rendered` use
`ReadOnce.frames()`); prep's own planning reads through the plan (`read=plan.read_text`, `point_folders(frames=
composed.junction.n_frames)`); launch reads the record on disk. Fix: have the transport declaration answer the
count through `rung.junction_record` while an extension has no record here, or narrow both doc sentences to the
reuse decision and its manifest.

**5. (low) `validation/chemistry.py:249–253` — ES5's way out on transport now misleads.** Restricted + a stated
count on SIESTA transport says "State … unrestricted (…; its count floats here -- TranSIESTA cannot hold …)".
Since this round a count kept stated beside `unrestricted` is written as stated (ES6 info) and TranSIESTA stops;
it floats only if left blank. Fix: "…unrestricted with the count left blank, which floats here -- …".

**6. (low) `reuse._as_its_first_prep_leaves_it` → `carry_pseudos` → `copy_pseudopotentials(plan=…)` prints
`ok: <src> -> <dst> (to copy)` (siesta/input.py:249)** on every `molbuilder describe --reuses`, bare `prep`
offer, step 4a and the tab's plan — a question that copies nothing says it will copy, and the prep then prints
the same lines again from `provide_data`. (Precedes the headline, so `_described_as_printed` is unaffected.)
Fix: carry the decision's view without the copier's print (e.g. plan the copies directly, or a quiet flag).

**7. (low) `transport/describe.py:436–440`** — the CLI's after-line "its first prep takes the junction and the
pseudopotentials <finished> ran with" is wrong for a frame-set extension (it takes the set) and for a finished
record that does not read (composed afresh — the warn note says so). Fix: "the junction it describes".

**8. (low) `web/blueprints/build.py:194–198`** reads the form twice through the same door (`config_from_params`
already calls `stated_params`); `stated_params`, now a door another blueprint imports, is missing from
`_shared.__all__` (l.998). Behaviour of every other `config_from_params` caller (build.py:899, :1040) is
unchanged — the body moved verbatim.

### Docs / docstrings

**9. `docs/engines/transport.md:3459`** — the shared-values table row says "the count floats, since TranSIESTA
holds no fixed total spin (`Spin.Fix` is never written here, § 3.1's spin note)". Now false: a stated count is
written (`Spin.Fix` + `Spin.Total`, transport/deck.py:200 via `_spin_facts`), as § 3.1 (l.4861–4864) says.

**10. `molbuilder/data/catalogue.template.toml:1222`** (the item's help, shown on the form): "free … the only
value under non-collinear and spin-orbit, and under unrestricted on a transport calculation" — contradicts the
`offered` row it sits above (every count on SIESTA transport) and ES6 ("a stated one is written as stated").

**11. `docs/science/chemistry-correctness.md:590`** — the `free` row still says "the only value beside
`unrestricted`" (SIESTA transport) and "the only value under `non-collinear` and `spin-orbit`" (SIESTA), against
the `a count` row edited this round (l.589) and the non-collinear row (l.584) of the same tables.

**12. `molbuilder/electronic_state.py:65–71` and `molbuilder/config/siesta.py:505–506`** — `count_must_float`'s
docstrings say a stated count "is refused by name (the settings gate)" / "is refused naming this reason"; the gate
says it as information and the deck writes it (validation/chemistry.py:270–281, ES6).

**13. `docs/engines/template.md:1686–1689` and `docs/science/overview.md:193`** (not edited this round) — list
"a fixed spin count on a transport rung" as a check refused by name / an **error**; ES6 since 2.65.1 and this
round's catalogue make it information.

**14. `molbuilder/web/static/lib/molview/history.js:316–331, :350` and `lib/molview/ui.js:1066–1070`** — still
quote the explicit-save model ("does NOT record a state; the user decides…", "the sequence grows from opening a
structure, an explicit save, and a load", "save on every edit and every keystroke becomes a milestone", "Stepping
forward lasts until you save", "an explicit-save history") — against molview.md § 11.2 as edited; round 1 fixed
only :289 and :381 (R4 #11).

**15. `tests/test_inspector_registry_e2e.py:147–151`** — the heading test's docstring still names "the engine's
alone, *SIESTA trajectory*", the branch removed from trajectory.js and presenters.md § 2 this round.

**16. `tests/test_structure_analyze_endpoint.py:7–15`** — the module docstring still says "API-level, and only for
what the pages cannot reach" and "The ANSWER is pinned … the card … in test_chemistry_card_e2e.py", beside the new
answer test in this file (R3 #5 half-fixed).

**17. `docs/web/web-api.md:652`** (`POST /api/transport/describe`) — still lists "its junction or a frame set of
it" among what a reuses describe writes (and "written in its place", since 2.39.0), beside the new "its junction
and its pseudopotentials taken by its first prep, not written here" — say "its junction's citation", as the route's
docstring now does.

## Checked and right
- **PLANNED view**: `decide` is never called under prep's own view (step 4a, prep.py:2741, sits outside
  `prep_calculation`'s `PLANNED.set`); `_as_its_first_prep_leaves_it` sets a scratch `Plan` only when none is set,
  after `junction_record` + `carry_pseudos` succeeded, and resets in `finally` on return and on any exception (a
  refusal in setup never sets it). `Plan` reads fall back to disk; `psml_sources`, `_psml_paths`,
  `parse_psml_header` (via `where_bytes_are`), `_psml_files` and the coverage check all read through it, so the
  settings gate passes on the carried files (R1 #1 closed). Nothing is carried out; `open_container` only plans.
  ContextVar per thread: the tab's plan route and describe route cannot leak it.
- **Frames inside reuse**: `_member`, `_laid_here`, `_decided`, `_rendered` use `ReadOnce.frames()` (R1 #2
  closed); `_at`/`_given_at`/`_unfinished`/`finished_window` read the finished calculation's or a calc already
  launched here — right.
- **One door**: `junction_record` takes the finished junction only with no record and nothing launched
  (`_launched`, a launch record that does not read counted as launched → `no_usable_record` refuses) — R1 #7
  closed; `finished_junction`/`frame_set_of` turn every `compose_refusals()` into `ReuseError` (R1 #10 closed);
  `composed_for_prep` writes it (`not here`); its read-back provenance names the task's citation and rename in
  all three shapes (same junction, frame set, one geometry of a set). Fresh-vs-recorded frame-set coordinates
  differ ≤ 8.7e-7 Å, inside `SAME_PLACE_ANG`, so the decision's renders compare as prep's do.
- **No byte change in prep's plan**: the record is written per member with the same bytes (no timestamp in
  `write_compose_record`); `carry_pseudos` copies once (later members see it through PLANNED); the decision's
  scratch plan never reaches the prep's.
- **Launch take-over** (`submit.py:2729`): first launch is after the first prep — `ReadOnce` reads the record
  on disk. CLI bare-prep offer and describe's `_manifest_of` (staging under the tree, `jobset` None →
  `_launched` False) both get the view.
- **Method word**: `_method` says `default` with `SOURCE_WORDS["default"]` ("default: not chosen" on the card)
  only where `chosen` is passed and lacks `method`; `stated_params` uses `read_text`'s blank rule (whitespace →
  blank); the 4 parametrized rows and the card e2e assertions match the code; `config_from_params` unchanged.
- **Catalogue**: with both rows gone, `offered("unpaired_electrons","siesta","transport")` is every choice; the
  gate's ES4 passes, ES6 says info; `electronic_state._spin` keeps a stated count stated; the deck's spin
  section writes `Spin.Fix` + `Spin.Total`; citation defaults still leave a cited fixed count blank (§ 3.1);
  `test_what_a_kind_offers` and the new validation test fail on the old catalogue and pass on this one.
- **JS**: trajectory heading = words+noun or *Trajectory* (presenters.md § 2 and presenters.toml agree, the
  words-only row gone); `pollMs` has no reader left; the GPU notice matches `_gpu_snapshot`'s 3 s TTL and no
  NVML word remains in system-load-monitor.js.
- **Tests in scope**: the `_ANSWERED` block's junction checks now use the one comparison (`frame_set_refusal`,
  `junction_difference` where the citations agree) with `SAME_PLACE_ANG`; the three-file check runs on rows made by
  `described()`; the tab test's absence list now names `H.psml` (R1 #4 closed); presenters.toml rows are ones
  `registry_answers` checks against the engine and kind tables.
- **Docs** edited this round read true to the code, but for #9, #11, #17 and the frame-count sentence in #4:
  molview.md (R4 #11/#12/#15 fixed; a saved state holds the selection, model.js:216–220), workspace.md,
  trajectory.md (date fixed), presenters.md (recount fixed), notifications.md (`persist-ok`), tabs.md § 5.5,
  task-setup.md (*not offered here*, viewer.js:419), results.md § 6.3, handover-procedure § 2.2, running-a-job
  banners, transport.md 4127 and § 1.1 cross-reference.

## Verdict
Option B's product code now does what § 2a.16 asks on every road the batch takes: the reuse decision sees the
pseudopotentials and frame count its first prep will have, through a planned view that is set only where none
is, always reset, never written and never leaked; the one door refuses and takes the finished junction exactly
as the contract says; prep's plan gets the same bytes. What remains is outside the reuse module — the
declaration's frame count before the first prep (#4), which the docs claim is the one door's — and one fixture
wait (#1) that turns a fast job into a module-wide 5-minute error. The stated-count change is right in code but
left four places saying the count floats or is refused (#9–#13), and the explicit-save comments survive in
history.js/ui.js (#14). Fix #1 before the batch; #4 and the docs in the same release.
