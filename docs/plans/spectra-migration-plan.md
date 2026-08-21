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
- **One stage (`freq`), and relaxation is its mandatory PRECONDITION**
  *(user rulings, 2026-08-20 — twice refined: first from "deferred" on the
  physics — a harmonic analysis is only valid at a stationary point — and
  then from "a skippable stage" to this, because optimization is not a
  peer rung you toggle among others; without it the measurement means
  nothing)*: the vibration deck performs the relaxation as its first act,
  then the Hessian on the result, **in one process** — geomeTRIC straight
  into `mf.Hessian()`, the standard PySCF pattern, no cross-stage
  geometry hand-off at all. The ONLY way relaxation does not run is the
  user's explicit statement: the template item **`already_relaxed`**
  (bool, default **false**, `procedure` category), whose help text names
  the responsibility being assumed. When it is set, the deck still checks
  the gradient at the input geometry and **warns with the numbers, never
  refuses** — the statement is the user's to make. The frozen set
  constrains BOTH the in-deck relaxation and the partial-Hessian subspace
  from the one structure-side declaration; the relaxation's convergence
  knobs are the existing geometry items (`geom_gmax` family) selected
  into the vibration template, **defaulting to the tight tier** — a
  frequency deserves a properly converged stationary point.
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
  atomic `.spectra.json` writer, so live-watch works unchanged — **and the
  relaxation precondition becomes a TRACKED phase** *(user, 2026-08-20:
  the viewer tracks all the steps — a silent gap while geomeTRIC works,
  likely the longest part of the run, would betray that)*:
  `phase_relaxation` written like the others, carrying the step count and
  current max force so the indicator shows convergence live; under
  `already_relaxed = true` it reads complete-by-assertion with the
  gradient-check number attached — the warning made visible where the
  phases live, not only in a log.
- **The collision resolves to one door**: the in-deck `compute_frequencies`
  → `thermo.txt` path **retires**. Thermochemistry (RRHO,
  `temperature_K`/`pressure_atm`) becomes items of the vibration template
  (writing into `.spectra.json`'s `engine_metadata`/a `thermo` block —
  schema v5 if a new field is warranted). One Hessian door on the framework.
- **The artifact gate** learns the vibration deck's expected outputs
  (`.spectra.json` present + schema-valid), the same shape every stage has.

## 2b. The thermo presentation — the viewer gains plots *(user, 2026-08-20)*

The deck computes, the viewer draws — never the reverse. Because a single
(T, P) point is not plottable, the deck evaluates the RRHO functions over a
**temperature grid** (arithmetic over frequencies it already holds; the
grid is a documented presentation default, not a scientific knob) and
writes the arrays into the v5 `thermo` block beside the headline numbers
at the requested (T, P). The viewer then adds, following the existing
chart conventions:

- **G(T), H(T), S(T) curves** over the grid, the requested T marked;
- **the free-energy decomposition bar** — electronic + ZPE + thermal
  enthalpy − T·S → G — showing where the number comes from;
- the **regime note on the plot itself** (full RRHO vs vibrational-only
  for frozen systems), not buried in a log.

Mode-resolved entropy contributions (which soft modes dominate S — telling
for anchored systems) are recorded as the optional follow-up, not v1.

## 3. The UI — the same translation, the same hand-over

The tab's **display half does not change** — with one additive
exception: the phase indicator gains the relaxation chip (§ 2, D4). Its compute half substitutes:

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
| **P1 · the producer** | vibration stage preset; `spec_for`'s vibration form (lifting `pyscf_script` emitters); artifact gate; `describe --calculation vibration` | ✅ **DELIVERED 2026-08-20** — the bar ran live TWICE (`tests/test_vibration_e2e.py`, 34 s total): water relaxes in 3 tracked geomeTRIC steps (max force 2.3e-5), the Hessian gives 1639/3791/3886 cm⁻¹, thermo lands full-RRHO with ZPE ≈ 13.3 kcal/mol + the 30-point grid, the artifact is schema-5 **in the attempt dir** and parses through the Results door; the DECOUPLED IR-only run lands in water's literature windows (bend 55.5 > asym 27.0 > sym 4.9 km/mol), resolving the prefactor flag at the band level. The live E2E's own catch: the lifted `_mb_outfile` resolved `__file__` through the attempt's symlink and wrote the artifact a level up — outputs now follow the cwd, where the run ran |
| **P2 · the tab substitutes** | the form from the catalogue; Send-to-Task-setup; the compute cards rewired | ✅ **code + pins DELIVERED 2026-08-20** — the tab renders `GET /api/build/schema/pyscf?calculation=vibration` through the shared renderer (the ES lock is schema-driven now — the old hardcoded ids had **never matched**, so the fade first works here); Generate/Save/methods/prior cards replaced by Send; the sender extracted to `lib/task-handover.js` (one door, two tabs — `/structure-optimization` delegates); auto-detect repointed and green through a real chromium (`net_charge`/`spin`/`method` land by name); the **byte-compat bar is pinned**: `test_the_browser_hand_over_writes_the_cli_s_files` proves the browser's template + structure pair byte-identical with `describe`'s and the description heads agreeing on id/kind/structure; the relaxation chip (data-driven fourth dot) + the § 2b thermo tab (G/H/T·S curves + decomposition bar, deck computes / viewer draws) landed with JS pins. **Open half of the bar:** the live browser water drive on the workstation ending on the Results tab |
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
- **D2 — thermo subsumption**: **SETTLED (user, 2026-08-20): retire** the
  in-deck `compute_frequencies` → `thermo.txt` path; the vibration kind is
  the one Hessian door. The CONTENT survives and grows: ZPE and G(T, P)
  are standard, publishable numbers that are ~free once the Hessian
  exists, so `temperature_K`/`pressure_atm` re-home to the vibration
  template and the results land in the v5 `thermo` block — with the
  honesty note on regime: full RRHO for a free molecule; **vibrational
  contributions only** (stated, not refused) when atoms are frozen, since
  a molecule anchored to an electrode does not rotate.
- **D3 — where relaxation lives**: **SETTLED (user, 2026-08-20, twice
  refined)** — relaxation is the freq deck's mandatory precondition,
  in-process, NOT a stage: a stage toggle would present "skip it" as an
  ordinary choice when it is an assertion. The user's explicit
  `already_relaxed = true` is the one skip, and it earns a gradient
  warning, never a refusal. (Draft one deferred relaxation on the old
  tab's premise; draft two made it a skippable stage; this is the final
  form.)
- **D4 — the artifact goes to v5, additive**: **SETTLED (2026-08-20,
  by the user's tracking point)** — `phase_relaxation` is a new field, so
  the schema bumps, under the readable-set rule the molstruct sidecar
  established: the reader accepts {4, 5}, a v4 file lacks the relaxation
  phase and reads whole. If D2 confirms, the `thermo` block rides the
  same bump — one additive version, both new facts. The Results/Spectra
  display gains exactly one thing: the relaxation chip on the phase
  indicator (additive, its own pin).
