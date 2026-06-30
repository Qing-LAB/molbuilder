# Spec — SIESTA `.fdf` emitter

**Module**: `molbuilder/siesta/input.py` (`SiestaConfig` lives at `molbuilder/config/siesta.py`) &nbsp;·&nbsp; **Tests**:
`tests/test_smiles_and_siesta.py`, `tests/test_review_fixes.py`,
`tests/test_pyscf.py` (cross-engine charge handling),
`tests/test_molwatch_preview.py` (sibling `.molwatch.log` output)

## Sibling outputs

Alongside the `<basename>.fdf` file, `convert(...)` also writes
`<basename>.molwatch.log` by default (`cfg.write_molwatch_log =
True`).  That sibling log carries one *initial-state preview block*
(step 0) containing the molecule's coordinates, with no energy /
forces / SCF data, and a `kind: initial_preview` marker line.

Purpose: molwatch can render the structure the moment the user
loads it, before SIESTA has produced any of its own output.  The
preview file is static (one block, never updated); for live updates
during a SIESTA run the user points Watch at the run **directory**
which discovers the right log via the [job-layout v1
protocol](../protocols/job-layout.md), or at the SIESTA `.out` file directly.

Set `cfg.write_molwatch_log = False` to suppress the sibling file.

**Stage-aware filename** (job-layout v1): when `cfg.stage` is set
to 1/2/3 (the Structure-optimization tab's "Relaxation stage" preset), the sibling
log filename becomes `<basename>-stage<N>.molwatch.log` so multiple
stages accumulate in one directory without collisions.  The
`SystemLabel` itself stays unsuffixed across stages so SIESTA's
`.XV` / `.DM` / `.CG` restart files transfer cleanly.  Filename
rule is centralised in
`molbuilder.trajectory_log.format.molwatch_log_basename`.

**"Run with:" verbose-comment block.**  When `cfg.verbose_comments =
True` (default), the generated FDF header carries a hint block
recommending the canonical invocation:

```fdf
# === Run with (job-layout v1) ===
# Run from this directory -- all outputs share the SystemLabel basename below.
#     mpirun -np 4 siesta < <basename>[-stage<N>].fdf > <basename>[-stage<N>].out
# Watch the run live: open the Results tab and point it at this directory
# (the loader resolves it to <basename>[-stage<N>].molwatch.log).
```

Following the suggested stdout redirect (`> <basename>.out`) keeps
the run directory canonically named per job-layout v1; the Watch
tab's directory discovery chain picks up `<basename>.out` as a
fallback when no `.molwatch.log` is present.

The format spec for `.molwatch.log` is documented in
[`pyscf.md`](pyscf.md) (the format itself is engine-agnostic;
the `# engine:` header line distinguishes who wrote it).

## Staged optimization (#542)

SIESTA mirrors the PySCF staged-opt design: a single CLI invocation
emits one `.fdf` per enabled stage plus one `<basename>.run.sh`
runner that walks the ladder.  Three surfaces drive it.

**Data model — `SiestaStageSpec`** (`molbuilder/config/siesta.py`).
Each stage is a dataclass with:

* `name` (regex `^[A-Za-z0-9_]+$` — surfaces as the filename suffix
  `<basename>_<name>.fdf` and the runner's `STAGES=(...)` array),
* `enabled: bool`,
* `relax_type: {"CG","Broyden","FIRE"}`,
* `relax_steps: int` (1..10000 — emitted as `MD.NumCGsteps`, the
  universal step keyword per decision-log 2026-06-23),
* `relax_force_tol: float` (eV/Å — emitted as `MD.MaxForceTol`),
* `relax_max_displ: float` (Bohr — emitted as `MD.MaxCGDispl`),
* `on_nonconvergence: {"proceed","continue","halt"}` (runner
  policy: proceed to next stage, continue THIS stage with
  `continue_retries` more passes, or halt the ladder),
* `continue_retries: int` (≥1).

`SiestaConfig.stages: List[SiestaStageSpec]` is the source of truth;
defaults come from `_default_siesta_stages()` — the 3-stage ladder
established by decision-log 2026-06-23 (stage1 CG warm-up 0.05 eV/Å,
stage2 Broyden publishable 0.04 eV/Å, stage3 Broyden crystal-tight
0.01 eV/Å).

**Structural validation** (`validate_siesta_stages`, wired into
`validation/siesta.py::_validate_siesta` — parity with PySCF's
`validate_stages` via `validation/pyscf.py`, 2026-06-29).  When a stage
bundle is requested, the same `validate()` pipeline the Build tab + CLI
run rejects the structural errors the generator can't recover from, as
clean `"error"` issues (NOT a render-time crash or a silent dropped
stage):

* empty `stages` list, or no enabled stage (would emit a no-op / unbound
  geometry);
* **duplicate stage `name`** — fatal: per-stage fdfs are keyed
  `<basename>_<name>.fdf`, so a collision silently overwrites a stage in
  `render_siesta_stage_fdfs`;
* bad `name` chars, `relax_type` outside {CG,Broyden,FIRE}, non-positive
  `relax_steps`/`relax_force_tol`/`relax_max_displ`, bad
  `on_nonconvergence`, or `continue_retries` outside [1,5].

This is the same contract PySCF enforces (config.md cross-engine
equivalence); SIESTA stages are no longer the unvalidated outlier.

**Execution on a scheduler** — how the ladder runs as a *job-set*
(per-stage directories, symlink-shared package, per-stage resources, a
dependency chain with `.XV` carry-forward) is specified in
[`../protocols/staged-execution.md`](../protocols/staged-execution.md).
The monolithic single-job runner below is its `direct`-mode fallback.

**Strategy presets** (`SIESTA_STAGE_STRATEGY_PRESETS`).  Overlay
enable-flag patterns onto whatever stage knobs are in `cfg.stages`:

| Preset        | Enables                | Use case |
|---|---|---|
| `publishable` | stage1 + stage2        | default; one warm-up + one publishable relax |
| `loose-only`  | stage1                 | quick preopt; geometry sanity only |
| `vib-quality` | stage1 + stage2 + stage3 | tight final for vibrational / IR work |

The same preset names and enable masks live in
`molbuilder/config/pyscf.py::STAGE_STRATEGY_PRESETS` and in
`molbuilder/web/static/lib/form-schema.js::STAGE_STRATEGY_PRESETS`.
All three are kept in lock-step by
`tests/test_siesta_stage_strategy_presets_drift.py` — a regex-parse
gate over the JS file, mirroring the PySCF #534 drift gate.

**CLI surface** (`molbuilder fdf ...`):

* `--stage-strategy {publishable,loose-only,vib-quality}` — overlay
  the preset's enable mask onto `cfg.stages` and emit a stage bundle.
* `--stages-json '<literal JSON>'` *or* `--stages-json path/to.json`
  — replace `cfg.stages` wholesale with a list of
  `SiestaStageSpec`-shaped dicts.  Combinable with `--stage-strategy`
  (knob values from JSON, enable mask from preset).
* Mutually exclusive with the single-stage `--stage {1,2,3}` overlay
  inherited from decision-log 2026-06-23 (that overlay remains as
  the minimum-viable single-fdf path).
* Bad JSON → clean Click error (`"--stages-json: not valid JSON"`);
  missing path → `"--stages-json: file not found"`.

Without either flag the CLI takes the single-fdf path; the
`<basename>.run.sh` runner and per-stage fdfs are not emitted.

**Form-schema stage-table widget**.  `SiestaConfig.stages` is
annotated `List[SiestaStageSpec]`, so the type-driven schema helper
(`molbuilder/web/blueprints/_shared.py::dataclass_to_form_schema`)
emits `{kind: "stage-table"}` automatically — no SIESTA-specific
form code.  The widget surfaces one row per stage with
`name`/`enabled`/`relax_type`/`relax_steps`/`relax_force_tol`/
`relax_max_displ`/`on_nonconvergence`/`continue_retries` columns;
the JS round-trips a list-of-dicts payload back through
`coerce_to_field_type` into `List[SiestaStageSpec]`.  Shape pinned by
`tests/test_siesta_form_schema_stage_table.py` (15 cases covering
field set, choice lists, defaults, section assignment, round-trip).

**Stage-bundle output layout** (per `molbuilder fdf JOB.fdf
--stage-strategy publishable`):

```
JOB_stage1.fdf              SystemLabel JOB  (one per enabled stage)
JOB_stage2.fdf              SystemLabel JOB  (same label across stages
                                              so .XV / .DM / .CG warm-restart
                                              between stages with no
                                              user copy)
JOB.run.sh                  STAGES=(stage1 stage2)  ON_NONCONV=(...)
JOB-stage1.molwatch.log     # stage: stage1   # convergence.max_force_ev_per_ang: 0.05
JOB-stage2.molwatch.log     # stage: stage2   # convergence.max_force_ev_per_ang: 0.04
```

**Per-stage molwatch headers**.  Each per-stage `.molwatch.log`
carries `# stage: <name>` plus `# convergence.<key>: <value>`
headers (currently `max_force_ev_per_ang` + `max_steps`) so the
Watch / Results inspector renders the right horizontal threshold +
progress indicator for the stage that's currently running.  Format
pinned by `tests/test_trajectory_log_stage_targets.py`.  Header
emission is in `molbuilder/trajectory_log/format.py::write_initial_preview`
(`stage_name` + `convergence_targets` kwargs); whitespace in a
convergence-target key raises `ValueError` at write time rather
than producing a malformed log.

**Runner contract**.  `JOB.run.sh` is bash, executable, passes
`bash -n` syntax check, and:

* declares `BASENAME='<stem>'`, `STAGES=(...)`, `ON_NONCONV=(...)`,
* applies the "force-halt-last" rule (the final stage's policy is
  always rewritten to `halt` so the ladder never overshoots its
  last fdf),
* loops over stages, calling `mpirun -np <N> siesta <
  ${BASENAME}_${STAGE}.fdf > ${BASENAME}_${STAGE}.out` per stage,
* respects `MB_NP` / `SLURM_NTASKS` / `PBS_NP` for process count
  (per `docs/config.md` v2 wrapper-independence contract).

CLI surface pinned by `tests/test_cli_siesta_stages.py`; runner
behaviour by the same module (`test_runner_*`).

**Cross-engine equivalence**.  See `docs/protocols/script-execution.md`
§ "Cross-engine equivalence table" — the SIESTA staged-opt surface
(`StageSpec` data model, `_emit_*_multi_stage` generator,
`convergence_targets` molwatch header, `STAGE_STRATEGY_PRESETS`
preset table, `--stages-json` / `--stage-strategy` CLI flags, form
stage-table widget) is the engine-by-engine parallel to the PySCF
implementation shipped under #534.

The emitter takes a `Structure` (or an XYZ/PDB file path) and writes
a SIESTA-runnable `.fdf` text.  It also optionally copies matching
`<Element>.psml` files into the FDF's directory.

## Public API

```python
@dataclass class SiestaConfig: ...
Config = SiestaConfig                       # backwards-compat alias

render_fdf(struct, config=None, *, cell=None) -> str
convert(input_path, fdf_path, config=None) -> dict
copy_pseudopotentials(species, lib, dest_dir) -> List[str]    # missing
```

## Output sections (in order)

1. **Header**: `SystemName`, `SystemLabel`, atom + species counts.
2. **Lattice**: 3×3 in Å.  Either user-supplied (`cell=` kwarg) or
   auto-generated as an orthorhombic vacuum box of `extent + 2 *
   cell_padding` per axis with the molecule centred.
3. **Species table**: `%block ChemicalSpeciesLabel` listing each
   unique element with its atomic number, ordered by atomic number.
4. **Atomic coordinates**: `%block AtomicCoordinatesAndAtomicSpecies`
   in Å, one atom per line, last column is the species index.
5. **Basis & grid**: `MeshCutoff`, `PAO.BasisSize`, `PAO.EnergyShift`.
6. **XC**: `XC.functional`, `XC.authors`.
7. **SCF**: `SolutionMethod`, `DM.MixingWeight`, `DM.NumberPulay`,
   `DM.Tolerance`, `DM.Energy.Tolerance`, `MaxSCFIterations`,
   `ElectronicTemperature`, optional `DM.UseSaveDM`.
8. **Spin** (only when `cfg.spin_polarized=True`): `Spin polarized`
   plus, when `cfg.spin_total is not None`, the two-line
   `Spin.Fix .true.` / `Spin.Total <v>` constraint pair (see "Spin
   contract" below).
9. **NetCharge**: emitted iff resolved charge != 0 (see "Charge
   contract" below).
10. **k-grid**: Monkhorst-Pack mesh from `cfg.kgrid`.
11. **Geometry optimisation / dynamics**: `MD.TypeOfRun` plus the
    universal step-count keyword `MD.NumCGsteps` for every
    relaxation mode (CG, Broyden, AND FIRE) in SIESTA 5.4.2 — the
    `MD.NumBroydenSteps` / `MD.NumFIRESteps` per-type aliases listed
    in some older references are NOT recognised by 5.4.2 and are
    silently dropped (see decision-log 2026-06-23 in `design.md`).
    Verlet/Nose dynamics use `MD.FinalTimeStep` instead.
    Relaxation modes (CG/Broyden/FIRE) also emit `MD.MaxForceTol`
    and the universal displacement cap `MD.MaxCGDispl` (despite the
    CG-prefixed name).  Dynamics modes (Verlet/Nose) emit
    `MD.InitialTemperature` and `MD.LengthTimeStep`; Nose-Hoover NVT
    additionally emits `MD.TargetTemperature` (defaulting to
    `md_initial_temperature` when `md_target_temperature is None`)
    — without it SIESTA's thermostat target falls back to 0 K and
    the run quenches instead of equilibrating.  All modes
    optionally emit `MD.UseSaveCG` / `MD.UseSaveXV`.  Skipped
    entirely when `cfg.relax_type.lower() == "none"`.
12. **Output flags**: `WriteForces`, `WriteCoorStep`, `WriteCoorXmol`,
    `WriteMDhistory`, and `SaveHS` (always emitted; the older
    `WriteHS` keyword is silently dropped by SIESTA 5.4.2 and was
    replaced 2026-06-23).
13. **Diagonalizer — solver choice and the optional GPU accelerator**
    (contract rewritten 2026-06-29; supersedes the old "ELPA only when
    GPU" model, which wrongly denied CPU runs ELPA).

    Two **orthogonal** decisions:

    a. **`Diag.Algorithm` is the eigensolver choice, independent of
       hardware** — `ScaLAPACK` (SIESTA's default Divide-and-Conquer) /
       `ELPA-1STAGE` / `ELPA-2STAGE`.  **ELPA runs on CPU *and* GPU.**
       Performance *affinity* (a hint, not a restriction): GPU favors
       **1STAGE**, CPU favors **2STAGE** (so the affinity-aware default
       is 1STAGE when GPU is on, 2STAGE when off; the user may override
       either way).

    b. **`enable_gpu` is an optional accelerator on top of an ELPA
       choice.**  ON ⇒ the ELPA solve runs on the GPU
       (`Diag.ELPA.GPU .true.`) — **GPU-only, no silent CPU fallback**.
       OFF with an ELPA algorithm ⇒ **CPU-ELPA** (`Diag.ELPA.GPU
       .false.`).  `enable_gpu` is only meaningful when an ELPA
       algorithm is chosen (GPU + ScaLAPACK is rejected at the UI).

    **Emission (`render_fdf`):**
    - ScaLAPACK → emit **nothing** (SIESTA's built-in default).
    - ELPA-* → **always** emit `Diag.Algorithm <choice>`, plus
      `Diag.ELPA.GPU .true.` (GPU on) or **`Diag.ELPA.GPU .false.`
      (CPU)**.  The explicit `.false.` is load-bearing: the
      source-built ELPA defaults to the GPU codepath, so an *omitted*
      flag makes a CPU-ELPA job initialize CUDA and crash
      (`cudaGetLastError: unknown error`; verified on Sol job 57852378
      — see [`../job-case-analysis/ANALYSIS-G1K1C4.md`](../job-case-analysis/ANALYSIS-G1K1C4.md)).
      `Diag.ELPA.GPU` *alone* (no ELPA `Diag.Algorithm`) is silently
      ignored — both keywords are required for the GPU path
      (Src/diag_option.F90:213-225).

    **Env routing — keyed on "needs ELPA", not on GPU:**
    - **ScaLAPACK** → `molbuilder-siesta` (the precompiled conda-forge
      build, no ELPA — perfectly fine for DnC).
    - **ELPA (CPU *or* GPU)** → `molbuilder-siesta-gpu`, the only
      ELPA-linked build (see [`siesta-gpu.md`](siesta-gpu.md)).  This is
      why CPU-ELPA still routes to the "gpu" env — that env is the ELPA
      build, GPU usage is separate.
    - The router (`runwrap`) detects **either** an ELPA `Diag.Algorithm`
      **or** `Diag.ELPA.GPU` true and bumps `siesta → siesta-gpu`;
      `molbuilder.json` `envs.{siesta,siesta-gpu}` overrides the concrete
      env name (the per-machine selector — there is no build-tab env
      picker, the env *follows* the algorithm choice).

    The web UI's `enable_gpu` toggle is the script-input contract — it
    never queries env presence.  `runwrap.write_run_wrapper` gates env
    presence at script-generation time: if the resolved target env isn't
    installed, `WrapperError` fires with the install hint instead of
    letting `source activate` fail cryptically at run time.

    Consistent with [`../protocols/scientific-validation.md`](../protocols/scientific-validation.md)
    (ELPA-CPU vs ELPA-GPU agree to ~1e-6 eV) and the CPU-vs-GPU benchmark,
    whose CPU baseline *is* ELPA-1STAGE on CPU.
14. **Troubleshooting block** (when `cfg.verbose_comments=True`):
    inline tuning hints for SCF / forces / speed, plus relaxation
    hints when an MD block is present.

## Verbose comments contract

When `cfg.verbose_comments=True` (default), every numeric parameter
above is preceded by a `# ...` block describing:

* what the parameter controls (one sentence),
* a sensible range,
* what to do when it misbehaves (one or two example tweaks).

Removing or substantially changing one of those comments is a spec
change and triggers a test update.

## Charge contract

Resolved charge is computed once per `render_fdf` call:

* If `cfg.net_charge is not None`: use it as-is (including 0, which
  disables auto-detection).
* Otherwise: `formal_charge_from_phosphates(struct)`.

If the resolved charge is non-zero, an explicit `NetCharge ±N` line
is emitted, with verbose-mode comments explaining the source
("user-specified" or "auto (phosphate protonation)") and what SIESTA
will do with it.

## Spin contract

SIESTA's default is spin-restricted (no `Spin` block emitted →
closed-shell DFT).  Open-shell systems (radicals, transition
metals, triplets) **silently produce wrong electronic structure**
without spin polarisation.

Targeted SIESTA version range: 4.1 -- 5.x.  v5 introduced a
unified single-line `Spin <option>` keyword (recognised options:
`non-polarized`, `polarized`, `non-collinear`, `spin-orbit`) that
supersedes the older multi-line `SpinPolarized true` form.  v4.1+
back-compat-accepts both spellings; the v5 manual marks the older
form deprecated.  The generator emits the v5 form.

The total-spin pin requires TWO lines, not one:

* `Spin.Fix .true.`  enables the constraint.  Without it,
  `Spin.Total` below is silently ignored.
* `Spin.Total <value>`  target total spin moment in mu_B
  (= number of unpaired electrons).

* `cfg.spin_polarized=False` (default): no `Spin` block.
* `cfg.spin_polarized=True`: emit `Spin polarized`.
* `cfg.spin_total` (float, optional): when set together with
  `spin_polarized=True`, emit the
  `Spin.Fix .true.` / `Spin.Total <value>` pair so SIESTA's
  initial guess targets the right multiplicity.  When set with
  `spin_polarized=False`, the value is ignored (no `Spin.Fix` /
  `Spin.Total` lines) — `Spin.Total` is meaningless without
  polarisation.

There is no equivalent of PySCF's "method=RKS validation": SIESTA
can be told `Spin polarized` regardless of basis or method, so
the only correctness rule is that the user MUST set it for any
open-shell system.  Document this loudly in the FDF when the user
passes `--spin-polarized`.

## Cell-padding auto-bump (charged systems)

When `cell is None` (auto-vacuum mode) AND the resolved charge is
non-zero AND `cfg.cell_padding < 25.0`:

* `effective_padding` is silently bumped to 25 Å.
* The `# (auto-generated orthorhombic vacuum cell ...)` comment in
  the FDF says so explicitly: "padding auto-bumped from X → 25 A
  because NetCharge != 0".

Reason: SIESTA's compensating-background-charge correction needs ≥25 Å
between periodic images for image-image Coulomb to drop below ~1
meV.  A neutral molecule doesn't need this.

## PDB serial / chain id width (via Structure.to_pdb)

This is a Structure-side spec; called out here because the FDF
emitter feeds Structure into PSML copy logic.

## `convert(input_path, fdf_path, config)`

* Auto-detects format from extension (`.xyz` or `.pdb`).
* Returns a summary dict: `{"fdf", "n_atoms", "species",
  "missing_psml"}`.
* If `cfg.psml_lib` is set and `cfg.copy_psml=True`: copies matching
  `<Element>.psml` files into `fdf_path`'s directory.  Missing
  pseudopotentials are listed in the summary; the calling CLI exits
  with code 2 in that case.

## Forbidden patterns

The emitter must NOT:

1. Emit the `MD.TypeOfRun` block when `cfg.relax_type == "none"`.
   The `none` value means single-point only; emitting CG would force
   relaxation.
2. Truncate atom-coordinate lines.  All atoms in `Structure` go into
   the `%block AtomicCoordinatesAndAtomicSpecies`.
3. Emit invalid SIESTA syntax for any standard config.  Every
   variant tested in `tests/` must `convert()` end-to-end without
   raising.

## Test reference

* `test_smiles_and_siesta.py` — render_fdf for a built DNA + a
  convert() round-trip via XYZ.
* `test_review_fixes.py` — net_charge override (S2), cell-padding
  auto-bump (D3), Config alias (D1).
