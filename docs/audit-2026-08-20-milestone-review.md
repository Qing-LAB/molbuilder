# Milestone review, 2026-08-20 — three passes over `b76d9883..ec34c2e6`

**Role:** audit report
**Domain:** *(root — cross-domain review: jobset/bench backend, MolView/
projects UI, validation, and their contracts)*
**Scope:** the 21 commits landed 2026-08-19 after the full-text review at
`0d7bdf3c` — Task I/II (bench→run connection, probe consent), the four-item
MolView/checkpoint UI plan, the export/persistence follow-ups, the
kgrid-as-statement rule + run-periodicity bridge, and the grouped bench
submission + `jobset.sh` launcher.
**Method:** the three-pass protocol — (1) the freshest unit against its
governing contract sections, re-read first; (2) every commit since the
milestone, walked from `git log` with per-feature receive-side traversals
(`process/code-audit.md` § 3.2); (3) the tests themselves, paired with four
mutation runs.  Three scoped searches fanned out for pass 2; **every claim
below was re-verified against source by the reviewer** — several agent
claims died on check (noted in § 4), which is the § 1.1 rule earning its
keep.

**At review end, no fix had been applied** — each finding carried a
proposal and waited for a decision; the tree stood at `ec34c2e6`,
byte-identical to what was pushed (every mutation restored from scratchpad
copies, verified with `git status`/`git diff --quiet`).

> **Resolution (2026-08-20, the follow-on session):** every finding was
> then discussed one by one and **fixed with an explicit yes each** —
> F1–F8 and nits N1–N5 land in the commit this record ships in, every
> load-bearing fix mutation-verified red first.  Two findings grew under
> discussion: F6 became the full two-doors widening **plus** sidecar
> schema 8 (the user's two-kinds contract and additive-versioning rule,
> `model/structure-molstruct.md` § 1–2), and F4's "part 2" turned out to
> already exist — the idiom fix opened it, it flagged a fifth sibling of
> F3 (`.docs-view-bar`), and its matcher was refined to the subject
> compound.  **N6** (the pre-existing system-load e2e failure) stays
> parked for its own investigation.

---

## 1. Findings — confirmed, ordered by weight

### F3 — the checkpoint action bar is a ghost: visible when the code says hidden

**The scenario.** Open the checkpoint view on a run directory whose folder
has no checkpoint history yet.  The module shows the empty-state with its
*Initialize* button and sets `elActions.hidden = true`
(`checkpoint.js:263`) — but Commit / Tag / Refresh / List / Graph stay on
screen beside it.  The template's initial `hidden` attribute
(`_projects_sidebar.html:194`) is equally dead, so the bar also renders in
the moment before the first state fetch answers.

**Why.** `.ps-checkpoint-actions { display: flex; … }`
(`projects-sidebar.css:1447`) with **no `[hidden]` guard** — the exact
`process/code-audit.md` § 3.1 trap, already fixed at six other selectors in
these same two stylesheets (e.g. `.ps-lock-banner[hidden]`,
`.ps-checkpoint[hidden]` at :1367).  The rule dates to `bf82356f`
(2026-07-24); the milestone's panel rework (`e16eb043`) inherited it and
made the hidden-actions state reachable in ordinary use.

**Why the graduated audit test stayed green** — see F4.

**Proposal.** One rule, the file's own established pattern:
`.ps-checkpoint-actions[hidden] { display: none; }`.

### F5 — the per-tab memory forgets the FILE half: three raw readers the contract forbids

**The scenario.** Work in Results with `run-3/` open and `bdt.out`
selected.  Switch to Modify, click any structure there, come back to
Results.  The sidebar restores Results' own *folder* — and then tries to
mark Modify's *file* in it, so Results' own selection is gone.

**Why.** `projects.md` § 2 says both slots are keyed per page and —
verbatim — *"nothing reads the raw storage keys directly, because a second
reader is how the keying would silently fork."*  Three unconditional raw
reads survived `81824019` in `list.js`:

* `list.js:816` — `restoreSelection()` reads the **shared** `SS_FILE` one
  line above its own compliant `readSelectionSlot(SS_DIR)`;
* `list.js:619` — `_renderList`'s selected-row marker;
* `list.js:745` — the selection-preservation rule across a re-list.

Plus two guarded fallbacks in `lib/results/file-picker.js:828,833` (lower
weight — they run only when the projects namespace is absent).

**Proposal.** Route all three through `readSelectionSlot(SS_FILE)`, and add
the source-text invariant test the § 2 sentence begs for: no
`sessionStorage.getItem(SS_FILE|SS_DIR)` outside `state.js` (the one
documented bypass, `handOffSelection`, is a writer and is already
pinned by `test_a_handoff_lands_in_the_target_tabs_own_slot`).

### F2 — a grouped trial's launch record is invisible to `status`, and the docstring says otherwise

**The scenario.** `jobset submit bench tight` rides six trials on one job.
Every trial's `bench-<POINT>/run.json` is stamped.  Nothing can show it:
`runstatus._launch_record` reads `run.json` only from `run-<n>` attempt
subdirectories (`runstatus.py:153-157` via `latest_attempt`), and a sweep
trial is *its own* attempt — so `launch` is always `None` for it.  Handed a
sweep set, `jobset_status` answers **"pending — prepped, not launched (no
run.json)"** (`runstatus.py:190-195`) — the exact false statement
`project-layout.md` § 1.6 created `run.json` to prevent.  Today the CLI
`status` verb loads only the root set (`_cli.py:240`), so the false line
does not print — the trials simply appear nowhere.

`submit_bench_group`'s docstring claims the stamp is *"so `status`, the
picker and a later single-trial re-run all see the truth"* — the picker and
the re-run refusal do (both use `was_launched`); **`status` does not**, and
no test asserts § 1.6's *"queued as job N"* promise for sweep trials
(`test_status_seq_is_none_for_a_sweep_point` pins only `seq`).

A sibling honesty gap: `summarize`'s census reads artifacts only, so a
trial the group never reached (wall clock, cancellation) is `unknown` —
indistinguishable from never-submitted, though its `run.json` says
launched.

**Proposal (decision needed).** Either teach `runstatus` the rule
`submit.py` already states — *"an attempt-less dir (a sweep trial) is ITS
OWN attempt"* (`submit.py:322`) — so `_launch_record` falls back to the
trial dir itself, with a test asserting the § 1.6 sentence; or retract the
docstring's `status` claim.  The first honors § 1.6; the second is one
line.

### F1 — § 2.3.2's re-run parenthetical omits the move-aside

**The scenario.** A trial times out inside the group.  § 2.3.2: *"Naming a
trial (`jobset submit bench tight G1K8C2`) still submits that one alone
(how a single point is re-run)."*  The user names it — and is refused:
*"already launched … To measure this point again, move the trial's
directory aside yourself"* (`submit.py:327-336`).  The refusal is right —
it is R2's § 1.5 immutability (2026-08-12), deliberately kept — but the new
contract sentence reads as an unconditional capability.

**Proposal.** Amend the parenthetical in the owning document:
*"how a single point is re-run — after moving the old trial's directory
aside (§ 1.5: a trial measures its point once)"* — then sweep restatements
(none found in code; the `submit_bench_group` docstring's "see the truth"
wording is accurate on this half).

### F6 — the export envelope is narrower than the codec's field set, and § 11.3 does not say so

**Proven half.** `structureForServer` (`model-jobs.js:255-267`) sends
`{elements, positions, metadata:{regions, cell, cell_origin, axis_kind,
vacuum}}`.  The model *holds* residue facts (`model-jobs.js:135-138` folds
`residue_name` into `facts.residue`) and never sends them —
`groupByLabel` walks `facts.labels` only — while the receiving envelope
accepts `atom_names` / `residue_*` / `title` (`_shared.py:185-188`).  So a
PDB-derived structure exported through MolView drops its residue identity.

**Contingent half.** The codec's sidecar also carries per-atom annotation
*channels* (`structure.py`, `METADATA_FIELDS`), which the export envelope
cannot express; whether any production pair reaches MolView carrying
channels was not established in this review.

**Not a loss (agent claim killed on check):** `pbc` — it never rides the
wire by design; the one deserialiser derives it from `axis_kind`, and
`test_every_metadata_field_survives_the_whole_export_pipeline` proves the
derivation end to end.

**Proposal (decision needed).** Either widen the envelope (send what the
model holds), or state the export's exact field set in `molview.md`
§ 11.3 so the boundary is a documented rule rather than an accident.

> **RESOLVED 2026-08-20 (user: option A, then the format ruling).**  The two
> viewer doors now fold/unfold the identity columns and channels (per-atom,
> riding each atom — the user's stated invariant), and sidecar **schema 8**
> persists the identity columns real-only with a readable set {7, 8}
> (`model/structure-molstruct.md` § 1–2: the two-kinds contract, the
> additive-bump rule).  Pinned by node round-trip, disk→disk through the
> real translators, and a delete-atom edit round-trip; all kill sites
> verified red.

### F7 — the periodicity gate's verdicts never reach the user, and one door reports the wrong filename

Three small holes, one theme — the report exists and is thrown away:

* **Export** computes `notices` (`build.py:845-846`) → the door reads only
  `ok` + `files` (`molview-doors.js:82`) and drops them.
* **Save** never runs `checked_periodicity` at all — the same structure
  export would refuse (fatal cell) is written silently.  Bytes are *not*
  at risk: the gate corrects nothing (`periodicity_gate.py`: "IT CORRECTS
  NOTHING"), so when both succeed the pairs are identical; the asymmetry
  is refusal + silence only.
* **`saveBinary`** requests `auto_rename` and then reports the name it
  *wanted* (`molview-doors.js:175`), ignoring the server's `path` — after
  a collision the toast names a file that does not exist.

**Proposal.** Surface `notices` through the door's result; report the
server's `path`; and decide whether save runs the same gate wrapper
(refusal parity) or stays silent by design — either answer belongs in
§ 11.3/§ 11.7.

### F8 — `529e2ec6` retired two rules and left their tests standing (caught by the full suite, swept in this review)

**The scenario.** The kgrid decision (2026-08-20) retired the
forgotten-axis warn and the span-ratio heuristic — in `validation.md` and
in code — but two tests in `tests/validation/test_geometry.py` still
asserted them, and nothing failed at delivery because the commit's
targeted runs never included that file.  The first full `none2e` since
(this review's) went red on both:

* `test_kgrid_one_on_periodic_axis_is_warn` — asserted the retired warn
  outright ("never assert a rule no document states").
* `test_kgrid_periodic_crystal_partial_span_no_false_positive` — its
  *intent* (a real crystal at ~50 % span with a full mesh must stay
  silent) is valid under the new rule too, but its *fixture* was a z-line
  (x/y extents zero), so the transverse axes really did hold 6 Å of
  emptiness — a geometry the statement rule legitimately hints on, and
  not the crystal the docstring names.

**Action taken (the one change this review made, as the completion of
`529e2ec6`'s own already-decided sweep — "fix the rule in the owning
document, then sweep the restatements"):** the first test is replaced by
its inverse pin (`test_kgrid_one_on_a_periodic_axis_is_silent` — k=1
beside a sampled sibling stays silent); the second keeps its scientific
pin on a fixture that models its own docstring (an 8-corner block
spanning 50 % of every axis).  Both new pins were mutation-checked red
(gate flipped to `k == 0`; `VACUUM_HINT_A` squeezed to 2.0) and the file
restored clean.

**The reviewer's own miss, on the record:** pass 3 enumerated and
mutation-checked the rule's *new* tests but never swept for *old* tests of
the retired rules — the retire-half of "tests serve the contract."  The
full-suite run existed for exactly this, and it is what caught it.

### F4 — the CSS `[hidden]` audit test is double-blind, which is how F3 survived

* **(a) The idiom gap.** `checkpoint.js` binds its elements by *bare
  assignment to pre-declared `let`s* (`elActions =
  document.getElementById(…)` inside `_attach()`).  None of the test's
  three collection regexes (`const/let/var` declaration, `$()`-direct,
  `els.prop =` table) match it — the panel's **entire element table** is
  invisible to the audit.
* **(b) The selector gap.** Step 3 matches only `#id` selectors
  (`_ID_IN_SELECTOR_RE`), and the class-side scan knows only
  `el("div","cls")`-created elements — so an element *hidden by id* and
  *styled by class from a template* can never be cross-referenced, even
  after (a) is fixed.

**Proposal.** Add the bare-assignment idiom to the id scan (one regex), and
extend the audit to map template ids to their classes (the templates are
already on disk) — or, minimally, add `.ps-checkpoint-actions` to the
known-violations ledger the test maintains and fix F3.

---

## 2. Nits and notes (no action unless wanted)

* **N1** — the `bench-grouped` ledger entry logs *all* sweep trials
  (`_cli.py:1329`), while the group rides only the pending ones; the
  subsequent `launched` entry has the truth per job.  Logging the pending
  list (or renaming the key `sweep=`) would make the decision entry exact.
* **N2** — prep step 4b swallows a launcher-write failure with no note
  (`prep.py:308-313` `except Exception: pass`): a bundle can silently keep
  a *stale* `jobset.sh` baking another machine's repo path.  A
  `log.produced`-style note on the skip would keep the record honest.
* **N3** — `topology_field_types` types a plain (non-Optional) field as
  `str` (`environment.py`: `get_args()` of a bare type is empty).
  Unreachable today — every `Topology` field is Optional — but the
  docstring's "a field added is automatically declarable" would silently
  mistype the first plain one.
* **N4 (science, hint-only paths — all fail soft)** — the kgrid gap uses
  the axis norm, not the perpendicular image distance, so a skewed cell
  overstates the gap; the atom span is Cartesian while the length is a
  lattice-vector norm (same caveat); wrapped coordinates make
  `extent ≈ length` and suppress the hint.  `validation.md`'s "the gap is
  the real vacuum" is exact for orthogonal, unwrapped cells — the common
  case here — and approximate otherwise.  Worth one qualifying clause in
  the doc; the code's hint-not-refusal posture already absorbs the error.
* **N5** — the workspace id is a pure function of the owner tag
  (`dispatcher.js:98-101`, deliberate); page isolation rests on the seven
  owner literals staying distinct (they are: `modify`, `structure-opt`,
  `transport`, `results:trajectory`, `results:structure`, `spectra`,
  `molview-demo`).  A future page reusing a literal shares cameras with
  zero warning — a sentence in `workspace.md` § 4 would make the
  convention a stated rule.
* **N6** — standing e2e failure, predating this milestone:
  `test_system_load_monitor_e2e.py::TestABrokenDriverIsVisible::…` (last
  e2e batch: 89/90).  Not chased in this review; re-run e2e after the
  next fix lands.

## 3. What held (checked and clean)

* **Grouped bench vs § 2.3.2 / job-contracts / § 6.1** — one launch act;
  the `.sbatch` delegates `bash bench-group.run.sh "$@"` through runwrap's
  one header emitter (hyphen passes `_SAFE_WRAPPER_NAME_RE`); sequencer
  cwd/log paths correct; GNU `timeout`'s own-process-group kill takes
  mpirun and ranks with the wrapper; `MB_LAUNCHED_BY` reaches trials both
  ways; the shield is unreachable on machine-generated sweeps (the
  translation always sets both knobs + `gres`, so envelope GPU headers
  always have their `gres`); wall = pending × bound × 1.1 + 300 s;
  dry-run parity with the single path; a second grouped submit correctly
  refuses ("all trials launched").
* **`jobset.sh` launcher** — bakes repo + env at generation, verbatim
  preamble + activation, `cd` bundle, `exec python -m molbuilder`; § 6.1's
  two-layer premise held, proven at delivery by the PATH-shim execution
  test.
* **VibrationView seal** — `lib/vibrationview/_export.js` is
  **byte-identical** to its pre-touch state (`git diff 1ce4b40d^ HEAD` is
  empty).
* **kgrid rule (science)** — k=1 never validated; isolated (flat bands =
  pure cost) vs transport (fake Bloch periodicity) messages are each
  correct; `kgrid` is non-Optional with default `(1,1,1)` so the loop has
  no None edge; `scf_must_converge` is the right SIESTA keyword with the
  right default semantics, same text in both homes.
* **Periodicity bridge** — the run's own box wins over the input pair's;
  broken pair degrades to lattice-only; deterministic pair pick; attached
  in both load responses.
* **Checkpoint swap** — `_applyOpen` is the one owner; leaving a run dir
  restores the file list and re-enables the filter; no other writer
  touches `#ps-list` visibility; `initialized` spelling agrees
  server → module → caller.
* **`_apply_run_config` doctrine** — explicit flags win; an unreadable
  user-edited file *stops* prep; decisions hit the ledger; the no-verdict
  policy note names the engine's runtime policy instead of going quiet.
* **Export wire** — field names/nesting agree end to end (`frames` beside
  the envelope; `name`/`path`; `overwrite`/`needsOverwrite`; multipart
  field names for upload); the pair leaves as one store-zip on the
  download path.

## 4. Pass 3 — the tests themselves

**Mutation runs (all restored from scratchpad copies afterwards; tree
verified clean):**

| planted break | expectation | result |
|---|---|---|
| kgrid gate `k == 1` → `k == 0` | the four statement-rule tests fail | **4 red** ✓ |
| bridge never attaches `axis_kind` | `TestRunPeriodicityBridge` fails | **1 red** ✓ |
| ui-context `matches` forced `true` | stale-pose refusal fails | **1 red** ✓ |
| door zips only the first file | archive-contents pin fails | **1 red** ✓ |

(The sequencer's own set — `set -e` kills the walk, envelope shrunk to
min, shield removed — was mutation-proven at delivery and recorded in
`76b8b783`/`ec34c2e6`.)

**Uncovered contract rules found:** § 1.6's "queued as job N" for sweep
trials (F2 — code cannot satisfy it, so the test must follow the fix);
`projects.md` § 2's no-raw-readers sentence (F5 — invariant test proposed);
`molview.md` § 11.3's export field set beyond the five proven fields (F6).

**Agent-claim attrition (the § 1.1 rule at work):** of the scoped-search
claims, the notable kills were "pbc is lost on export" (false — derived by
design, and the pipeline pin proves it) and "the `-k kgrid` tests are the
rule's tests" (my own first run selected two unrelated tests; the four
real ones all go red under mutation).

## 5. Suite state

`none2e` over the reviewed tree (`ec34c2e6`): **6931 ran — 6922 passed,
3 failed, 6 skipped**.  The three failures, each owned:

1. `test_docs_structure.py::test_every_new_tree_doc_has_a_provenance_header`
   — **this record's own first write** lacked the `**Domain:**` line.
   Fixed in place.
2–3. the two `tests/validation/test_geometry.py` kgrid tests — **F8**,
   swept as recorded there.

After the header fix and the F8 sweep, both files pass (37/37) and the
failed set re-runs green; the two sweep pins were mutation-checked red
first.  The prior full green (6915/0/6) was at the `81824019` tree.  e2e
was not re-run in this review; its last batch stands at 89/90 with N6 the
one standing failure.
