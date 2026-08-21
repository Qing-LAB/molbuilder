# Audit 2026-08-21 — jobset · engines · execution · the two tabs, full-text

**Role:** review record
**Domain:** the whole reviewed span — jobset/execution, engine emission,
validation, the structure-optimization and spectrum tabs, and the
documentation set.
**Method:** the R×3 protocol (docs/process/code-audit.md): five parallel
full-text readers, every load-bearing claim re-verified against source
before entering this record (spot-verified marks: ✓).  Suite state at
start: none2e 6780/0.  **Nothing below is fixed without a yes per
bucket** — except the same-day regressions in bucket 1, fixed on sight.

---

## Bucket 1 — same-day regressions: FOUND AND FIXED (commit 49a0c15f)

- **R1 ✓ (error, fixed):** every vibration deck with a declared ECP was a
  `SyntaxError: keyword argument repeated` — the honesty gate's "silent"
  verdict on `ecp`/`ecp_atoms` was a water-probe artifact (no ECP
  candidate), and the block added beside the lift's EXISTING resolution
  emitted a second `ecp=` into `gto.M`.  Duplicate deleted; the honesty
  gate now `compile()`s every probe render; a gold-dimer pin asserts one
  kwarg and a compiling deck.
- **R2 ✓ (error, fixed):** the spectrum tab threw a strict-mode
  `ReferenceError` on every form edit — the P3 sweep removed
  `_formDirty`'s declaration and reader but left three writers.  Gone.

## Bucket 2 — errors on reachable paths, pre-existing (awaiting yes)

- **E-A1 ✓** task-setup viewer: `loadSweepChoices(String(task.engine))`
  stringifies the `{name}` object → `?engine=%5Bobject%20Object%5D` → 400
  → sticky empty sweepable cache.  Opening a folder holding a task.json
  mislabels bench rows and disables the add-a-setting picker.  (viewer.js
  :578, :597; the file's own header declares this bug fixed — it was, on
  one of three accessors.)
- **E-A3 ✓** the optimization tab's live preflight drops findings on
  `ok:false` (`if (r.ok) renderIssues(...)`, viewer.js:303) — the exact
  behavior the endpoint's 2026-06-14 change exists to serve.  The spectrum
  tab renders regardless.  Align both.
- **E-V2e** the vibration path's frozen-atoms verdicts contradict each
  other in one response: the engine validator says "held fixed during
  relaxation" (keyed on `optimize`/`optimizer`, which the kind ignores)
  while the kind's Pattern-B check says the label "is NOT consumed" —
  and BOTH are wrong: the Hessian mask consumes it, and the relaxation
  does NOT constrain those atoms (no constraints reach geomeTRIC).
  **Contains a science decision:** should the vibration relaxation
  constrain frozen atoms (a frozen slab that relaxes is the wrong
  geometry for the partial Hessian)?
- **E-V4a** `validation/spectra.py` `__all__` exports the deleted
  `PySCFSpectraEngine` — `import *` raises.
- **E-J1** `prep bench` on a PySCF description: the recorded guard
  ("refuses at the seam") lapsed when the PySCF seam landed; what stops
  it now is an accident (SIESTA-named measurement pins failing resolve)
  whose refusal blames pins the user never wrote.  The deferred
  GPU-sweep hazard behind it is now reachable-by-engine.
- **E-J2** `prep bench`'s success hint prints a command the CLI refuses
  (`submit bench <trial>` — the grammar is `submit bench <stage>`), and
  "one trial per invocation" contradicts the shipped grouped submission.
- **E-J3** a SIESTA wrapper whose restart answer is unreadable gets the
  PySCF continuation-contract header (the three-way conditional's None
  arm; key it on the suffix).
- **E-T4** task-setup's `--from` hint mis-numbers the previous attempt
  when a stage is disabled (enabled-filtered index vs the ladder-position
  seq contract).
- **E-B9** the hand-over file's `_what` hardcodes "from the
  Structure-optimization tab" — a Spectrum send writes the wrong
  provenance.
- **E-B10** `lib/task-handover.js` keys notice severity on `n.severity`;
  periodicity notices carry `level` — the error-escalation arm can never
  fire.
- **G-1a** `validation/stages.py::validate_ladder` omits `calculation`
  (a vibration ladder routed there would skip kind science) and has zero
  production callers — wire it or retire it.
- **G-1b** gate ③ does not actually fire at Task-setup save (codec
  checks only; the schema half runs at describe/dispatch) — workflow § 9
  overclaims.  Run `validation.task.preflight` at save, or narrow the row.
- **G-1c** RKS+spin>0 raises a ValueError in the deck door before the
  gate — a stack trace where the preflight gave two warnings.  Let the
  gate own the refusal.
- **G-1d** `_validate_pyscf` has no parity check — a PySCF optimization
  deck with impossible spin dies inside PySCF at runtime; SIESTA and the
  vibration kind both check.  One shared call closes it.

## Bucket 3 — the double-fire family (needs a dedup design ruling)

On a vibration deck BOTH the type-keyed engine validator and the
kind-keyed science run, so grid-level, open-shell-metal, and
restricted-with-spin warns each fire twice (once with the wrong
optimization rationale), and the radical family fires five times with
two different charge derivations that can disagree (`resolve_net_charge`
auto-detection vs the view's `or 0`).  Options: the kind entry
suppresses the engine copy's overlapping block; or validate() dedups by
`where`; plus route the view's charge through `resolve_net_charge`
(V-3c) and translate the kind findings' `where` ids back to catalogue
names so they land on the form's cards (V-3b).

## Bucket 4 — documentation reconciliation sweep (mechanical; one yes covers it)

Verified representatives (✓ = spot-checked by hand): web-api.md
contradicts itself on the spectra routes ✓ (line 414 vs 424); workflow
§ 7 still says "two whole workflows have not migrated" against its own
§ 9; roadmap's migration box / deferrals / § 1 / § 5-IR clause still
gate a delivered spectra migration; web/spectra.md's header box + § 1 +
§ 6 "stands until P3" describe the pre-P2 page; engines/pyscf.md still
teaches SpectraConfig-through-dataclass and the retired thermo.txt row;
form-schema.md's spectra claims; job-system.md's three
one-trial-per-invocation spots vs the shipped grouped bench;
handover-procedure "the bar Spectrum has to clear" + "twelve" (now 14)
vibration items ×3 docs; script-preparation "only two writers" prose vs
its own three-✓ table, "nine slots" vs twelve DeckSpec fields, and the
44/45/12 vs 43/54/14 counts; job-contracts § 6.1 omits `calculation` +
`bench` keys; ~30 decayed line anchors across validation.md +
engines/pyscf.md; `engines/siesta-gpu.md` cited but nonexistent;
gate-②'s owner (handover-procedure) never states the cell-gate rule —
write it in; gate-④'s § pointer off by a section; tabs.md two revisions
behind on both tabs; four README one-line summaries stale; toc.json
omits both live plans; D-series jobset docstring decay (describe copies
vs "never copied", PySCF wrapper's unconditional-resume claims ×3,
deleted-machinery citations ×8).

## Bucket 5 — retirement / consolidation decisions (each its own yes)

- **C-plans** archive the two DELIVERED plans substance-first (the
  folder's own rule): re-home web/spectra.md's two plan citations and
  template.md § 5's missing `calculations` + `refs` rows first.
- **C-SpectraConfig** finish the started retirement: no production
  constructor exists; remaining metadata has no live reader; the view is
  the runtime object.  Includes `spectra/selection.py`'s dead half (the
  deck inlines the selector; the module's preview caller is gone).
- **C-shared** `_shared.py`'s three zero-caller helpers
  (`apply_sidecar_if_possible` family, `regions_pattern_b_notice`) —
  kept alive by their own tests; retire or re-wire.  Related contract
  decision: validation.md § 5's Pattern-B notice now fires NOWHERE on
  the optimization route (its two routes were deleted) — re-home the
  MUST or retire it.
- **C-doors** `/api/run/install-wrapper` + `/api/siesta/install-pseudos`
  have zero browser callers — same position fdf/pyscf were in before
  deletion.
- **C-JS** dead residue lists for both tabs (state fields, empty
  STATIC_FORM_IDS with stale comments, no-op restoreFormState, stale
  header comments in core.js/spectra-viewer/templates/build.py
  docstrings — full lists in the readers' reports, all verified
  spot-checks passing).
- **C-jobset** S1-S7: `_pick_trial`'s dead arm, `_SAFE_*` constants,
  five stage-less residue branches with docstrings stating the dead
  shape as live, `DECL_TYPES` bool/int3, the trial-cold rule spelled
  twice, the duplicated read-API comment block, two submit.py residues.
- **C-refs-render** the `refs` citations render nowhere reachable today
  (the long-help path drops the parameter; short-help fields get only a
  tooltip) — pass `f.refs` through and decide legend casing (`title`
  emitted but unread).
- **C-smoke** T1: retire the frequencies smoke test (subject deleted;
  hidden behind an env skip); T2/T3 stale module docstrings.
- **C-autodetect** the auto-detect panel trio is three near-verbatim
  copies across three tabs — extraction meets the ≥2-caller bar.

## The engine-emission reader's findings (landed after the snapshot)

**New errors on reachable paths (bucket 2 additions):**
- **E-M4.7** any RHF/UHF vibration deck with `compute_raman=true` (the
  default) dies with a NameError at the Raman phase — `_build_mf_at`
  evaluates the `dft` name on its `force_cpu` branch, but the imports
  emit `dft` only for RKS/UKS.  The failure lands AFTER the full
  Hessian is paid for.  (Lifted code; latent since the old generator.)
- **E-M3.1** `_LiftView.charge` = `net_charge or 0` — the vibration
  deck DROPS the phosphate auto-detection the optimization deck runs
  (`resolve_net_charge`): a nucleic-acid vibration with charge unset is
  silently a different calculation than its optimization sibling.
  (Same fact as validation V-3c; two readers, one finding.)
- **E-M7.1** Pattern-B falsely warns on the reserved `frozen_atoms`
  label itself ("the engine does NOT consume") — since schema 7 the
  frozen set lives INSIDE regions; exclude `FROZEN_LABEL`.  And on the
  vibration path Pattern-A is structurally vacuous (the view's frozen
  set IS the sidecar's — it compares the sidecar to itself).
- **E-M1.5** the `use_gpu` catalogue help — the form-facing text — still
  advertises the retired CPU fallback ("falls back to CPU"), directly
  contradicting the stop-not-fallback rule the emitted probe enforces;
  the same stale sentence is EMITTED into every vibration deck.
- **E-M4.6r** the residual behind the fixed ECP crash: the vibration
  spec sets `check_rules=None`, so a non-compiling vibration deck is
  never parse-checked at prep — the optimization deck's
  `layout.check_rules` (`ast.parse`) is one line to reuse.
- **E-M5.2** the optimization deck passes geomeTRIC-named convergence
  kwargs to berny UNCONDITIONALLY — whichever way pyberny reacts
  (error or silent ignore) is wrong.
- **E-M6.3** the vibration spec's `line=` slot carries a wrong DFT test
  (`method != "HF"` classifies RHF/UHF as DFT) — a no-op today (zero
  Sections), a latent bug the day a Section is added.
- **E-M4.1** `vibration_emitters.__all__` names the deleted generator —
  same class as validation's finding on `validation/spectra.py`.

**Contract-vs-code on the fresh § 7a work (bucket 2/3 additions):**
- **M1.3** the role table promises "GPU promotion; newton() wrap" at
  the relaxation site; `_build_mf_at` applies neither — a
  `scf_soscf=true` vibration run relaxes under DIIS.  Honor or amend.
- **M1.1** `init_guess = "chkfile"` (the continuation read) is a
  sanctioned-but-unstated second spelling of an SCF_SECTION knob —
  name it in § 7a or route it through the dresser.
- **M1.2** the DFT trio (xc / grids.level / disp) has two spellings —
  layout's for optimization, hand-constants for vibration; the same
  drift class § 7a closed for SCF, one section over.  A DFT dresser is
  the symmetric fix.
- **M1.4** two GPU-consumption mechanisms in one engine (promotion
  helper vs import-time class selection), the promotion helper emitted
  DEAD in vibration decks.

**Simplification/dead (bucket 5 additions):** `scf_setup.
density_fit_line` has zero callers (added 2026-08-21, dead on
arrival); the emitted artifact carries TWO `_mb_outfile` definitions
(the dead first being the known-wrong `resolve()` form); the IR-only
block nests a complete per-mode loop inside its own per-mode loop
(N² idempotent work + a redundant second kernel per displaced point);
the retry loop and geomeTRIC kwarg spellings are two-home copies;
`frozen_elements`/`frozen_residue_names` are pinned `[]` so every deck
carries inert union machinery; `emit_solvent_lines`' `mf_var`
parameter is never varied; an unused constant import rides
vibration_emitters with a no-longer-true justifying comment.

**Doc drift (bucket 4 additions):** the emitted header still teaches
`python <job>.spectra.py` + the retired layout/topics vocabulary; the
deck's IR banner and its constants comment contradict each other on
validation status; the Methods fragment overclaims (rot/trans
projection unconditional — false on the partial-Hessian path;
"analytically" for FD polarizability derivatives); `_LiftView`'s
"dissolves at P3" note predates P3 landing; siesta.md § 5's spin
contract names a deleted field and a measured-false premise;
pyscf.md's intro/§ 2 staged-loop and `cfg.stage` claims; template.md's
"three items carry allocation" (four), "twelve vibration items" (14),
§ 12.2's overtaken PySCFConfig.stages claims; script-preparation's
four-writers counts (SIESTA 48 not 44; the Spectra row's W2/artifact
✅ overstate what a one-Block layout can enforce); the catalogue's
solvent `mf.SMD()` engine_key (no SMD path), log_file's forbidden
`mol.stdout` pattern, auxbasis's stale anchor; the vibration items'
`expands` carrying prose where the key means engine keywords; the
unsuffixed vibration `.log` vs its own rung-naming comment.

**Verified clean by the reader:** the one-spelling rule holds for every
SCF_SECTION knob, density_fit, and PCM; the shared-emitter set
(threading, GPU probe, save helper, molwatch, dresser, DF-kw, solvent)
is genuinely one-home.

## THE EXECUTION PLAN — persistent, resumable *(user, 2026-08-21: "make
sure all plan is persistently recorded so we can compact context and
come back to do another full round of review and validation")*

Each unit lists its scope and STATUS.  A fresh session resumes here:
read this section, pick the first non-done unit, verify its items
against the code before acting (claims decay), and after the last unit
run the R×3 protocol again — the user has asked for a second full
review-and-validation round when these land.

- **U0 (DONE, 49a0c15f):** the two same-day regressions — ECP double
  emission (+ compile probes in both gates), _formDirty writers.
- **U1 (DONE, 346b58b7):** consolidate the launcher work
  with the review's findings on the same path —
  (a) a Task-setup UI control + endpoint to (re)write the portable
  `jobset.sh` into an EXISTING described folder (their current dirs
  predate the save-door change; on another machine a workstation-baked
  launcher is wrong and the bootstrap is the fix);
  (b) the next-step notes teach the WHOLE flow via `./jobset.sh`,
  including the bench lane when `task.bench` is non-empty:
  prep bench → submit bench <stage> (grouped) → summarize bench
  (writes `bench-result.json` + the editable `run-config.toml`
  proposal) → read/accept → prep run (applies the verdict pins) →
  submit run;
  (c) E-A1: the two `String(task.engine)` sweepable-poisoning sites
  use the one accessor `_handoverEngine`;
  (d) E-T4/A4: the `--from` hint's token uses full-ladder position,
  not the enabled-filtered index;
  (e) E-J2: the CLI's bench success hint teaches the real grammar
  (`submit bench <stage>`), and prep's hints speak `./jobset.sh`
  (prep just wrote it).
- **U2 (awaiting yes):** the correctness unit — E-M4.7 (HF+Raman
  NameError), E-M3.1/V-3c (charge bypass via `resolve_net_charge`),
  E-M4.6r (vibration `check_rules` = the optimization deck's
  `ast.parse` gate), E-M1.5 (the CPU-fallback lie in help + emitted
  decks), M1.3 (newton at the relax site — honor the role table),
  G-1c (RKS+spin refusal owned by the gate), G-1d (pyscf parity),
  E-M6.3 (`is_dft` on the vibration `line=`), E-M4.1 + E-V4a (the two
  broken `__all__`s), E-J3 (wrapper header None arm), E-B9 (handover
  `_what` sender), E-B10 (notice `level` key), E-J1 (PySCF bench
  refusal by name), G-1a (validate_ladder), G-1b (gate ③ at save).
- **U3 (awaiting ruling):** the double-fire dedup (engine copy
  suppressed where the kind owns the check; one charge resolver;
  `where` ids translated to catalogue names) — and the FROZEN-ATOMS
  SCIENCE DECISION: constrain the vibration relaxation or state why
  not (E-V2e / E-M7.1's Pattern A/B fixes land with it).
- **U4 (awaiting yes):** the documentation reconciliation sweep —
  bucket 4 plus the engine reader's additions (workflow § 7, roadmap
  ×4, web/spectra.md, engines/pyscf.md, form-schema.md,
  job-system.md ×3, handover-procedure, script-preparation counts +
  DeckSpec slots, job-contracts § 6.1 keys, tabs.md, README rows,
  toc.json plans, siesta.md § 5 spin, template.md counts, line
  anchors ~30, the emitted header's retired vocabulary, the Methods
  fragment's two overclaims, the deck-internal IR-banner
  contradiction).
- **U5 (awaiting per-item yes):** retirements/consolidations —
  archive the two delivered plans substance-first; SpectraConfig +
  selection.py dead half; `_shared.py`'s three zero-caller helpers +
  the Pattern-B contract re-home; the two orphaned doors; the JS/state
  dead lists; jobset S1-S7; `density_fit_line`; the doubled
  `_mb_outfile`; the N² IR-only loop + redundant kernel; the DFT
  dresser (M1.2) and one GPU mechanism (M1.4); refs rendering
  (pass `f.refs` through the long-help path) + legend casing;
  T1-T3 test retirements; the auto-detect trio extraction.
- **U7 (2β — DESIGNED 2026-08-21, build awaiting yes):** multi-point
  value axes, integrated across placements under ONE bench.  The
  user's question ("gpu vs cpu separately submitted to different
  clusters — how does bench integrate these under one framework?")
  and the design that answers it, built on three verified hooks
  (environment.json's `Domain` = a reachable (partition, qos) pair
  with limits; `submit._job_wants_gpu` already reading per-job GPU
  placement; the wrapper already choosing its env from the finished
  deck):
  1. value axes enumerate into trials exactly like machine axes; each
     trial's deck carries its point as a PIN through the existing
     override lane (template < declaration < run-config < flags);
  2. a trial's PLACEMENT is derived from its own finished deck — the
     same one door the wrapper uses: a GPU-asking deck needs the gpu
     domain, else the CPU domain.  No new declaration; the GPU fact
     keeps its one home;
  3. `submit bench <stage>` partitions trials by required domain and
     submits ONE GROUPED JOB PER DOMAIN (the grouped machinery
     unchanged, run per group; a `--domain` filter submits one side);
     job-set.json records each trial's group;
  4. caps are validated per trial against ITS domain at PREP
     (Sol: gpu 48 cores vs standard 128, from environment.json) — an
     unschedulable cell (gpu × np128) is refused or dropped BY NAME
     at prep, never discovered in the queue;
  5. `summarize` stays allocation-blind (async by design): it reads
     whichever trials' artifacts have landed → ONE bench-result.json
     spanning the matrix + one run-config proposal;
  6. CROSS-CLUSTER is the same design, because environment.json is
     per-machine and the bundle is one folder (the user's is a git
     repo): each machine preps/submits only the groups whose domain
     it reaches; results ride the folder back; summarize on either
     machine reports unknown/incomplete for the rest — the existing
     honesty, unchanged;
  7. comparability stated, not assumed: CPU trials compare across
     Sol's partitions (same silicon); GPU numbers are their own
     build's; the verdict reports per-combination facts and choosing
     stays the user's (run-config is a proposal).
  On the yes: the contract lands in generator.md § 4.3a (replacing
  the refused-by-name note), then grid enumeration → prep fold →
  per-domain grouping in submit → summarize's matrix report → the
  Task-setup bench table's warning flips to describe the real rule.

- **U6 (after U1-U5):** the second full R×3 review-and-validation
  round the user has asked for, plus a full `none2e` + live E2Es.

## The consolidated priority head (all five readers in)

1. **E-M4.7** HF+Raman NameError (a paid-for Hessian then a crash).
2. **E-M3.1 / V-3c** the charge-heuristic bypass (silent wrong science).
3. **E-M7.1 / V-2e** the frozen-atoms contradiction + the science
   decision (constrain the relaxation or state why not).
4. **E-M4.6r** vibration `check_rules=None` (the shipped-crash class
   stays uncatchable at prep).
5. **E-M1.5** the CPU-fallback lie in form help + emitted decks.
6. **E-A1** the task-setup sweepable poisoning.
7. The rest of bucket 2, then the bucket-3 dedup ruling, then the
   mechanical sweeps.
