# The spectra migration — the vibrational calculation becomes a described job

**Role:** plan
**Domain:** execution + web
**Started:** 2026-08-20
**Companions:** [`engines/template.md`](?doc=engines/template.md) (*"a template
describes a CALCULATION, not an engine"* — this plan is that sentence's
scheduled work for spectra, named in
[`execution/script-preparation.md`](?doc=execution/script-preparation.md)'s
four-writers table); [`engines/stages.md`](?doc=engines/stages.md) §§ 1.1a,
6.8; [`execution/generator.md`](?doc=execution/generator.md) § 4.3a;
[`web/task-setup.md`](?doc=web/task-setup.md);
[`web/handover-procedure.md`](?doc=web/handover-procedure.md);
[`web/spectra.md`](?doc=web/spectra.md) (the tab's display contract — its
compute half is what migrates); [`engines/overview.md`](?doc=engines/overview.md)
§ 3 (the frozen-atom three-stage contract, kept).

---

## 0. The rulings this plan executes *(user, 2026-08-20)*

1. **New framework first; the old code transitions onto it.** We build a new
   producer on the framework the structure-optimization loop verified, then
   **substitute**, then **remove** the obsolete parts. We do not repatch the
   old path.
2. **Template-driven, engine-specific.** The spectra calculation gets its own
   template — the same `template@2` format the structure-optimization one
   uses, sharing every item that is genuinely shared, **specialized** where
   spectra differs (its own options, its algorithms trimmed to the purpose).
3. **PySCF first, engine-agnostic shape.** The workflow's shape must admit a
   future second engine exactly as the optimization ladder's does.
4. **The UI follows the same principle**: the form is translated from the
   template/catalogue the way the parameter tab already translates it — never
   hand-maintained.
5. **The workstation is the test environment** for this migration.

## 1. What exists today — the facts *(verified 2026-08-20, file:line)*

**The old path is a parallel mini-framework**, predating the JobSet one:

- `spectra/engine_base.py` — its own engine `Protocol` + registry, central
  method `render_script(struct, cfg) -> str` (finished **text**, the exact
  shape the seam gave up on 2026-08-18);
- `spectra/pyscf_script.py` (1557 lines) — the code generator: a
  self-contained `<job>.spectra.py` (SCF at the input geometry → Hessian →
  harmonic analysis, with the frozen-atom partial-Hessian path → optional
  finite-difference Raman + IR projection → per-mode electronic structure),
  writing `.spectra.json` **phase-by-phase** through an inlined atomic
  writer — which is what the tab's live-watch polls;
- `config/spectra.py` — `SpectraConfig`, ~30 fields with **inline** form
  metadata (not catalogue-backed); its own schema endpoint
  (`/api/build/schema/spectra`) and render door (`/api/spectra/render`);
- the run is launched **by hand**: Save writes the script, installs a
  `.run.sh` via the one wrapper renderer, the user runs it. No CLI verb, no
  `prep`/`submit`, no launch record, no ledger.

**What survives untouched on the read side:** `.spectra.json` (schema v4,
`spectra/results.py` + `sidecars/spectra.py`), the Results-tab presenter, the
chart/mode-table/animation stack, and the live-watch — all of them read the
artifact, not the producer.

**The designed hooks already waiting** (the framework anticipated this):

- `task.calculation` — *which kind of calculation this describes*, the key
  into the engine's warm-file vocabulary (`task.py:175`);
- `pyscf/warm-files.toml` already declares the **`[vibration]`** section
  (empty = base rules only — *"an empty section IS a statement"*);
- the `procedure` category exists in the template's six;
- the PySCF seam (`spec_for`, `.py` decks) is live and E2E-proven.

**The collision to resolve:** the framework *also* has a second frequency
path — catalogue items `compute_frequencies` / `temperature_K` /
`pressure_atm` make the optimization deck emit `mf.Hessian()` +
`thermo.thermo()` → `<JOB>.thermo.txt` (`pyscf/input.py:1302`). Two homes for
"the Hessian on the framework" would be drift by construction.

## 2. The target — one described calculation

A vibrational spectrum becomes an ordinary described job:

```
describe (--calculation vibration)   →  task.json + <label>.template.toml
prep run freq                        →  the deck: <label>_freq.py
submit --mode direct  (workstation)  →  runs; writes <label>.spectra.json
Results / Spectra tab                →  reads it, unchanged
```

- **`task.calculation = "vibration"`** — the word the warm-file section
  already declares. (The *tab* keeps its name, Spectra; the calculation kind
  is the physics.)
- **Two stages: `opt` → `freq`, and `opt` is skippable** *(user ruling,
  2026-08-20 — reversing this plan's first draft)*: a harmonic analysis is
  only valid at a stationary point, so relaxation belongs INSIDE the
  calculation, not in a premise about its input. The framework makes the
  ruling cheap: the ladder is just stages; **the skip flag is the existing
  per-stage enable** (the Task-setup stage table's toggle — "I already
  relaxed this" = disable `opt`); the optimized geometry reaches `freq`
  through the same warm-file carry every continuing stage uses; and the
  frozen set flows to BOTH stages from the one structure-side source —
  constraining the relaxation and selecting the partial-Hessian subspace,
  one fact, two consumers. When `opt` is skipped, the freq deck CHECKS the
  gradient at the input geometry and **warns, never refuses** (names the
  max force, says frequencies may be unreliable off a stationary point) —
  skipping is a deliberate choice the user is entitled to make.
- **The template**: same `template@2` file, generated for the vibration kind:
  - **shared items stay shared** — method/functional/basis/charge/spin/ecp/
    dispersion/density-fit, SCF knobs, grid, and the whole execution
    category (threads, memory, GPU, `mpi_np` where meaningful) are the
    *existing* PySCF catalogue rows; the spectra template selects them with
    vibration-appropriate defaults;
  - **new catalogue rows for what is genuinely spectra's** (kind `engine`,
    `engines=["pyscf"]`, spelled from `SpectraConfig`'s fields, help text
    carried over): `compute_raman`, `compute_ir`,
    `displacement_amplitude_ang`, the `es_*` mode-selection family,
    `freq_min_cm1`/`freq_max_cm1` — category `procedure`/`profile` as each
    warrants. **IR and Raman are one calculation, one UI, two independent
    toggles** *(user, 2026-08-20)*: one Hessian, one mode set, one shared
    displaced-geometry loop computing whichever properties are ticked.
    The old `compute_ir`-requires-`compute_raman` refusal
    (`pyscf_script.py:96`) was an implementation artifact — and backwards
    on cost, since IR alone (a dipole read per displacement) is far
    cheaper than Raman (a response calculation per displacement) — so the
    lifted emitters gain the dipole-only loop mode and the coupling
    retires. With IR first-class, its NOT-VALIDATED intensity prefactor
    (roadmap § 5) stops being carried: **P1's E2E validates IR
    intensities against a water reference** and resolves the flag.
    The frozen-atom *selectors* stay structure-side (the sidecar
    three-stage contract, `engines/overview.md` § 3) — frozen atoms are a
    fact about the structure, not a template value;
  - **which-kind is template-visible**: the vibration template exists
    *because* `describe --calculation vibration` chose the vibration preset
    (stage set + item selection + defaults) — the same mechanism
    `default_pyscf_stages(strategy)` uses today, extended per kind.
- **`spec_for` stays ONE per engine.** The PySCF emitter learns the
  vibration deck form, and the existing `pyscf_script.py` block emitters are
  **lifted in** as its emission library — the old code transitioning onto
  the new framework, exactly as ruled. The deck keeps the phase-writing
  atomic `.spectra.json` writer, so live-watch works unchanged.
- **The collision resolves to one door**: the in-deck `compute_frequencies`
  → `thermo.txt` path **retires**. Thermochemistry (RRHO,
  `temperature_K`/`pressure_atm`) becomes items of the vibration template
  (writing into `.spectra.json`'s `engine_metadata`/a `thermo` block —
  schema v5 if a new field is warranted). One Hessian door on the framework.
- **The artifact gate** learns the vibration deck's expected outputs
  (`.spectra.json` present + schema-valid), the same shape every stage has.

## 3. The UI — the same translation, the same hand-over

The tab's **display half does not change**. Its compute half substitutes:

- steps "Set the parameters" + "Generate/Save" are replaced by the pattern
  the structure-optimization flow uses: the form rendered **from the
  catalogue** for (engine=pyscf, calculation=vibration) — the same
  form-schema renderer, dropdowns/tri-selects and all (`form-schema.md`) —
  and **"Send to Task setup"** writing the four hand-over files
  (`handover-procedure.md` § 2). Task setup preps; submit runs direct on the
  workstation; Results shows the artifact.
- **kept, because they are structure-side, not producer-side**: the
  read-only MolView card; auto-detect
  (`/api/structure/analyze` filling charge/spin/method); the frozen-atom
  sidecar pre-fill (the reference instance of the three-stage contract).
- the old `/api/spectra/render`, `/api/build/schema/spectra`, and the
  save-script/install-wrapper flow retire **with** the substitution — the
  user's order: substitute, then remove; no long-lived dual path.

## 4. Phases *(each lands reviewed, pinned, mutation-checked; workstation E2E is the bar)*

| phase | delivers | done when |
|---|---|---|
| **P0 · contracts** | this plan reviewed; the vibration rows added to the catalogue; `stages.md` § 6.8 note for the kind; `generator.md`/`job-system.md` phase entries updated; the thermo-retirement decision recorded in the catalogue items' help | the doc gates green; the user's yes on § 6's decisions |
| **P1 · the producer** | vibration stage preset; `spec_for`'s vibration form (lifting `pyscf_script` emitters); artifact gate; `describe --calculation vibration` | **CLI E2E on this workstation**: water → describe → prep → submit direct → a schema-valid `.spectra.json` with modes → the Results tab loads it |
| **P2 · the tab substitutes** | the form from the catalogue; Send-to-Task-setup; the compute cards rewired | the same water run driven **from the browser**, ending on the Results tab; the four hand-over files byte-compatible with the CLI's |
| **P3 · removal** | `render_script`/registry path, the two old routes, `SpectraConfig`'s form metadata (fields → catalogue), the in-deck `compute_frequencies` trio | the old doors gone; `script-preparation.md`'s table shows Spectra ✅ across; every remaining `spectra/` module serves the **artifact** (results/selection/methods/parse), none the producer |

**Test-pin shape:** the P1 E2E pinned end-to-end (deck contains the Hessian
block; `.spectra.json` schema-valid, phases complete); a byte-faith pin that
the migrated deck's science blocks match the old generator's for one fixed
config (the lift is a move, not a rewrite); the § 3 hand-over compared
against the CLI's files; removal pinned by the four-writers table test.

## 5. Deliberately untouched

VibrationView and the 3Dmol seal (task #104); the SpectrumChart component
plan; transport (the *branching* kind — its own workstream); the IR-intensity
validation debt (roadmap § 5 — carried into the new item's help text
verbatim, still flagged NOT VALIDATED); Sol/SLURM (D7's cluster half — this
migration is workstation-scoped by ruling and touches no submit-layer code).

## 6. Decisions for review *(each blocks its phase, none blocks P0's start)*

- **D1 — the kind word**: **SETTLED (user, 2026-08-20): `vibration`** —
  the warm-file section's word; the tab keeps its user-facing name.
- **D2 — thermo subsumption**: retire `compute_frequencies`/`thermo.txt` in
  favor of the vibration kind (this plan's default: one Hessian door), with
  RRHO as vibration-template items — or keep the cheap in-deck check?
- **D3 — the compound ladder**: **SETTLED (user, 2026-08-20): IN v1** —
  `opt` → `freq` with `opt` skippable via the per-stage enable; frozen
  atoms constrain both stages from the structure's one declaration; a
  skipped `opt` earns a gradient warning in the freq deck, never a
  refusal. (The first draft deferred this on the old tab's
  "assumed pre-relaxed" premise; the physics says otherwise, and the
  framework's existing stages/warm-carry/enable machinery makes the right
  design the cheap one.)
- **D4 — `.spectra.json` v5**: only if thermo output needs a first-class
  block; otherwise `engine_metadata` carries it and v4 stands.
