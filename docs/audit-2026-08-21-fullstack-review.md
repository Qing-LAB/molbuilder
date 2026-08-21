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

## Pending

The engine-emission reader (script_emit + the two decks + layout +
catalogue vs engines/*.md) had not reported at the time of this
snapshot; its findings extend this record when they land.
