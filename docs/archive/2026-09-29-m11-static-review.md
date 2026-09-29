# The M11 static review — five tracks, 2026-09-29

*ARCHIVED — the evidence record. This is the record the plan's § 5w works from, not a plan: the five static
full-text reviews W50 asked for (`plans/plan.md` § 0a M11), each read by an agent
at commit `03c52cfa` with nothing changed and nothing run, then every defect
re-read against the code and the engine's own source by the main session. The
findings' IDs are the reviewers' own, prefixed by track in the plan (SO- Structure
optimization on SIESTA, PO- on PySCF, SS- Spectrum on SIESTA, PS- on PySCF, T-
Transport). Line numbers are as of `03c52cfa` — re-derive before acting (plan § 5a).
The reviewers could not write their own files (the harness returns a subagent's
findings as text), so each report below is the main session's condensation of the
reviewer's hand-back; the verification tables are the main session's.*

---

## Review — Structure optimization on SIESTA

Code at 03c52cfa; SIESTA 5.4.2 source
(local tree + gitlab raw 5.4.2). Real runs: claude-au-bdt-au/optimization/aubdtau-relax
(pre-M6 decks), claude-validate h2-flat-siesta / h2-m2d, Sol Au bench.

Ordinary path correct: every keyword spelled as 5.4.2 reads it, units accepted, presets
match tuning.md § 4.

### C. Findings (as reported; each to be verified before acting)

- **C1 defect** — execution items have several homes (template, stage column, execution,
  one-point bench row); pins applied after stage overrides (resolve.py 394 vs 483) so a
  calculation-wide value beats a rung's column; generator.md 371-373/433 ladder has no stage
  rung. GPU: run_uses_device (prep_inputs 424-452) reads execution + template, never a stage
  override → `del out["gres"]` (520-522) while the deck writes Diag.ELPA.GPU .true.;
  bench_inputs same (856-859); --from hint reads only ov.restart (viewer.js 2210).
- **C2 defect** — system_label, psml_lib, species_order shared only for transport → offered as
  optimization columns; species_order per rung → .XV species indices overwrite (ioxv.F 94-95);
  system_label per rung → SystemLabel vs warm-file list (prep.py 1241 vs 2420); psml_lib
  ignored once pseudos staged (prep.py 596-600). Fix: shared for optimization + vibration.
- **C3 defect** — GPU-ELPA BlockSize realign rule (tuning.md 457, 475-477; template.md
  1628-1629) not built; SIESTA snaps down to a power of two (diag_option.F90 447-471); four
  texts wrong (cat 830, config/siesta.py 1016-1017, bench/result.py 228, manual).
- **C4 defect (UI text)** — index.html 167-170 "Start from … clean — the default, none of the
  three flags is written, so SIESTA never looks": all false; 137-144 stale Job-name paragraph.
- **C5 defect** — false execution labels in every .template.toml: omp_threads "auto: physical
  cores" (CPU default 1), mpi_np "(single-process)", max_memory_mb "machine's maximum resolved
  at prep" + "SIESTA emits a SystemMemory hint" (no such keyword); ulimit -v per process.
- **C6 defect (help)** — scf_energy_converge "on by default" (read_options.F90 702: false).
- **C7 defect (help)** — pao_energy_shift "SIESTA's own default is 0.02 Ry" (atom.F 57: 0.01).
- **C8 defect (record)** — omp_threads never reaches the deck (Resources.cpus_per_task);
  PROVENANCE says `auto` while the wrapper ran 1; gpu_count/gres same gap.
- **C9 defect, low** — copy_psml read by nothing.
- **C10 risk** — automatic ParallelOverK for any non-Γ mesh (input.py 535) against SIESTA's
  default and manual; Au run-0: 16 ranks, 5 k-points, 11 idle; deviation not stated.
- **C11 risk** — ELPA decks write ParallelOverK .true., SIESTA forces it off; layout.py
  156-161 note says the opposite; 123-125 "BlockSize third state 0".
- **C12 risk** — find_psml glob `S*.psml` can stage Si/Sb/Se… as S (input.py 289-294).
- **C13 risk** — transiesta offered for an optimization; dies after the queue wait.
- **C14 doc-drift (engine)** — tuning.md 311 MaxSCFIterations; tuning.md 132-135 MD.MaxDispl;
  "one-step MD" reason (template.md 2465-2468, input.py 621-630); MD.UseSaveCG written for
  Broyden/FIRE (inert; config/siesta.py 1492-1495 says CG only); runwrap 1924-1927 "packaged
  build has no ELPA"; config/siesta.py 1107-1109.
- **C15 doc-drift** — stages.md 313-318 counts; template.md 1624; form-schema.md/index.html/
  config docstring "form from SiestaConfig"; template.md § 9.1 vs § 12.1, § 10a, 2386-2388;
  config/siesta.py 1-27, 1541-1542; resolve.py 395-397; runwrap 4889-4890; task_setup.html
  35-38; input.py module docstring; runwrap retry comments 3905-3907, 4031-4032;
  continue_retries help omits the SCF_NOT_CONV retry.
- **N1–N14 nits** — relax_steps tiers; troubleshooting hints ignore the deck's values; bench
  pins outside ranges (44 warnings on disk); one range two severities; blank number → raw
  TypeError; psml_lib caption; compatibility locks; mixing_weight "density"; relax_force_tol
  "largest force" (per component); `coarse` default stage runs tight values; md_target help;
  bench card "independent"; effective-parameters block lists every kind (§ 5t.5, doubt);
  auto_ranks n_atoms shim.

Known (not re-reported): W37 → M2h; W38 F4/F5 → M2i; W38 F6 → M2k; W36 ⑦ → M2k; W20 → M2e;
W30; W33 / § 5q; § 5s.4; § 5v.

---

## Review — Structure optimization on PySCF

Code at 03c52cfa; PySCF 2.14, geomeTRIC
1.1.1, gpu4pyscf 1.8.1. Reference deck: projects/claude-validate/optimization/h2-pyscf-m2b/
H2_01_coarse.py (09de947a, pre-M6).

### C. Findings (as reported; each to be verified before acting)

| # | sev | finding |
|---|---|---|
| C1 | defect | on_nonconvergence wired to assert_convergence (one step's SCF); PySCF swallows GeomOptNotConvergedError; halt writes _optimized.xyz + exits 0; continue retries only an SCF failure from the input geometry; proceed turns off the per-step SCF guard. input.py 1276-1332; relax_policy 54-70. Contract pyscf.md 262-280, tuning.md 913-917, cat 2093-2096. (= Spectrum-PySCF C1) |
| C2 | defect | GPU + unrestricted: deck calls gpu4pyscf `stability` = NotImplemented → TypeError, deck catches NotImplementedError/AttributeError only (input.py 1180, 1212) |
| C3 | defect | optimizer=berny offered: criteria passed under names berny ignores; pyberny absent; pip-into-conda advice (input 328-330, 408-410); no trajectory/log/held atoms; vibration already refuses berny |
| C4 | defect | `<label>_<NN>_<stage>_geom.log` promised (input 314-316, pyscf.md 129, runfiles 868-874, parser) never written: PySCF substitutes its console-only log.ini (geometric_solver 147-150); 0 such files in projects/ |
| C5 | defect | layout.py 212, 216 `{value:.0e}` (= Spectrum-PySCF C9) |
| C6 | defect | PySCF GPU under a scheduler: no --gres (runwrap 4938-4940, 5003-5006) — KNOWN W38 F6 → M2k; add template.md § 6.1 sentence |
| C7 | risk | blank max_memory_mb = PySCF's 4000 MB; help/deck comment/footer say machine / "no cap" (= Spectrum-PySCF C6a) |
| C8 | risk | sbatch sizes OpenMP-only job as MPI ranks (-N 1 -n W; + -c T → W×T); admission None; `--np` hint for PySCF — static only, confirm on Sol |
| C9 | risk | "PySCF auto-loads a Stuttgart ECP" (chemistry 234-235), cat 1215 (= Spectrum-PySCF C5) |
| C10 | risk | ECP set lacking an element → stderr "not found", element runs all-electron; deck says "applied to"; no read-back (mole.py 706-716) |
| C11 | risk | wB97X-D in help; PySCF 2.14 refuses (= Spectrum-PySCF C18) |
| C12 | risk | `*-v` functional + default d3bj double-counts dispersion |
| C13 | risk | box checks (cell.check errors) run for PySCF gas-phase decks (validation/__init__ 254-256) |
| C14 | risk | R3 ladder-loosening check knows only SIESTA fields (validation/stages.py 54-67) |
| C15 | nit | "snapshot the converged density" comment false (mf never moved) |
| C16 | doc-drift | trajectory carry `<label>_geom_optim.xyz` never matches `<label>_<NN>_<stage>_geom_optim.xyz` |
| C17 | doc-drift | deck tier comment tight gmax 1.5e-5 vs preset 2e-4 |
| C18 | doc-drift | tuning.md "no direct cap / line search" vs trust radius |
| C19 | doc-drift | pyscf.md `convergence_set` "available" |
| C20 | nit | banner prints literal BLAS=1 |
| C21 | doc-drift | 18 stale statements (a–r in the hand-back): staged-opt loop, proceed default, stages.md § 1.3 counts, template.md threads allocation, form-schema dataclass, catalogue CLI flags, config/pyscf docstrings + molwatch_log shim, task_setup.html READ-ONLY, presets source, solvent eps=? header, tabs.md, --np hint, berny footer, geom.log name, continue help, tuning.md § 5 resume |

Decisions named by the reviewer: C1 (how continue re-enters), C2 (CPU stability before GPU
or skip with reason), C3 (refuse or retire berny), C4 (write the log or drop the promise),
C7 (what a blank memory budget means), C16 (carry by token or drop the row).
D-1: the default one-rung description is named `coarse` but runs the template's publishable
values — not a defect by contract; the tab should say so.

---

## Review — Spectrum tab on SIESTA (vibration: relax → force constants → the job's finish)

Code read at 77731fa6; SIESTA 5.4.2 source (git e486d12).

### C. Findings (as reported; each to be verified before acting)

1. **C1 · defect** — `write_forces` / `write_coor_step` offered on the vibration form (no
   `calculations` key: catalogue :656-673, :674-690), written into the FC deck, unguarded.
   The finish needs both: `parse/engines/siesta_fc.py:178-183` requires forces+coords at FC
   step 0; SIESTA prints force rows only if `Write.Forces` (write_subs.F:781-784,
   read_options.F90:1858) and per-step coords only if `WriteCoorStep` (state_init.F:288,
   outcoor.f:88, read_options.F90:1906). Unticking either → the finish fails after the whole
   FC run (5.1 h here) with "has not written its first step". Fix: fix both `.true.` in the FC
   deck and take them off the vibration form, or refuse `false` by name in
   `siesta_vibration_checks`.
2. **C2 · defect** — `spectra/vibrational_analysis.py:286-298`: a non-stationary reference
   says "untick `already_relaxed`…"; `ladder_relaxation` (:203-205) ignored. Contract
   vibration.md § 5.8 (:1444-1453) and prep (`validation/sidecar.py:418-422`): continue the
   `relax` stage. Reachable: relax stopped at 0.009949 constrained; FC step 0 got 0.009929
   against 0.01. Fix: with `ladder_relaxation` present, give the V1.36 remedy.
3. **C3 · risk** — FC wrapper retries SCF_NOT_CONV as a "warm resume" (`runwrap.py:3334-3337`,
   :3995-4008; `continue_retries` 1, catalogue :528). FC cannot resume: DM written only at
   step 1 (save_density_matrix.F90:123-126), every step re-reads it (m_new_dm.F90:136-144),
   the run restarts at the first atom (siesta_init.F:693-695). A retry repeats steps 0…k from
   the same start. Fix: no SCF retry on an FC deck, or describe it truthfully.
4. **C4 · risk** — `relax` accepts Verlet/Nose/none (catalogue :408-419; config/siesta.py
   :623-638; siesta/input.py:607-619): MD or a single point instead of a relaxation; only a
   note (validation/siesta.py:584-593). Fix: refuse all but CG/Broyden/FIRE on a vibration's
   `relax`.
5. **C5 · risk** — `fc_displacement = 0` (ofc.f90:97 divides by dx) and `relax_steps = 0`
   (MD.Steps 0 = single point) only warn (validation/metadata.py:100-107; template.py:1340
   "range stays advisory"; form min/max not enforced, form-schema.js:681-768). Fix: refuse
   `fc_displacement ≤ 0` and `relax_steps < 1` by name.
6. **C6 · risk, DESIGN DECISION** — Task setup gives both steps the same 12 `stage`-group
   columns and the tier presets (viewer.js:3593-3606, :236-254; build.py:2389-2417;
   config/siesta.py:1546-1565). On `freq`: relax_type/steps/max_displ do nothing; on `relax`:
   fc_displacement does nothing; `relax_force_tol` on `freq` becomes the finish's test
   (prep.py:748); a preset splits it ("coarse" relax at 0.05, freq judges at 0.01 → warning
   loop). Choice: route each step its own items (as transport does per role), or mark inert
   cells.
7. **C7 · doc-drift (engine claim)** — "each later displacement starts from the previous
   displacement's density" (vibration.md :1179-1181, :1201, :1421-1422;
   siesta/vibration_deck.py:112-115; input.py:1365-1367) is false: every displacement
   re-reads the step-0 density (m_new_dm.F90:136-144); real outputs log it.
8. **C8 · doc-drift (engine claim)** — "SIESTA reads these files unless a deck says .false."
   (input.py:1379-1380, :1389-1390; vibration_deck.py:121-124; vibration.md :1185-1186;
   config/siesta.py:24-26): true for `.DM` only (DM.UseSaveDM default true,
   read_options.F90:824-829); `MD.UseSaveXV`/`MD.UseSaveCG` default to UseSaveData = false
   (struct_init.F:84-85; read_options.F90:1284-1286).
9. **C9 · doc-drift** — FC deck banner (input.py:1168-1174), wrapper header
   (runwrap.py:4183-4186), wrapper help, flat mode line say the stage reads .XV/.CG; the same
   deck writes `MD.UseSaveXV .false.`. Fix: key the three texts on what the deck writes.
10. **C10 · doc-drift** — run log's parameter rows: `runwrap.py:2081`
    `_sc.declarations(engine="siesta")` with no calculation/stage (declarations supports both,
    script_emit.py:931-951): lists tbt_*, md_*, restart… Contracts disagree (parse.md
    :1561-1562; job-contracts.md :949).
11. **C11 · nit** — deck header prints `jobset launch run 02_freq` (input.py:1118-1120);
    `identity.resolve_stage_ref` (:374-397) accepts only `freq` or `#2`. Every staged SIESTA
    deck.
12. **C12 · nit** — FC deck carries relaxation text keyed on `relax_type`: tips
    (input.py:1661-1669), a false `MD.Steps` bench row (_bench_marks_for 444-452), "after a
    successful relaxation" (:1679), "Without this block SIESTA relaxes every atom" (:1289),
    "drive SIESTA yourself… Perfectly good" (:1147-1155), WriteCoorXmol "FINAL coordinates"
    (the last displacement).
13. **C13 · nit** — thermochemistry note always says whole-body motions were removed
    (vibrational_analysis.py:187-199); here 0 were (methods.py:213-216 already conditional).
14. **C14 · nit** — relax deck's start-state text ("One field in the description decides…",
    input.py:1384-1391) — a vibration has no `restart` field; carry keyed on
    task.calculation (prep.py:2420-2421) drops `.CG` for a continued relax; flat layout's
    re-run relax reads the FC run's displaced `.XV`.
15. **C15 · nit** — `pyscf/stages.py:148` `stage_name == "relax"` case-sensitive vs
    stages.md :562 (names case-insensitive); `resolve_stage_ref` exact (identity.py:390-392).
16. **C16 · nit** — empty number box → `null` → `_shared.py:1190` `float(None)` → HTTP 400
    raw TypeError naming no field (build.py:1067-1071); comment at _shared.py:1316-1319 says
    such failures surface as an error Issue.
17. **C17 · nit** — hand-over files 0600 (`/api/files/write` mkstemp+replace,
    files.py:1669-1688): template.toml, source.xyz, source.molstruct.json — same class as
    V1.37, a second writer V1.37 does not name.
18. **C18 · doc-drift nits** — form-schema.md :292, § 5 :382-390; task-handover.js:11;
    core.js:19-23; web/blueprints/spectra.py:14-16, :72-74, :79-84 ("alias preserved");
    spectra.html:5 title "Raman / IR"; task_setup.html:33-42 "READ-ONLY … Saving is NOT
    wired"; config/siesta.py:13-24 (old numbers); relax_force_tol / relax_max_displ help
    defaults (0.02 / 0.05 vs this form's 0.01 / 0.02).

Known (not counted): W48 (already_relaxed / temperature_K kind "deck" vs "produce"); V1.33;
W41 (⚠ beside in-tolerance sentence; held indices in sorted order; FC .out titled
"optimization"; whole-output parse); V1.37; V1.38.

### A. Coverage (summary)
Read in full: vibration.md, siesta.md, spectra.md, form-schema.md, handover-procedure.md,
task-setup.md, normal-modes.md, spectra.html, spectra/viewer.js, form-schema.js,
task-handover.js, build.py, blueprints/spectra.py, task_setup.html, catalogue, template.py,
config/siesta.py, resolve.py, pyscf/stages.py, siesta/vibration_deck.py, siesta/input.py,
siesta/layout.py, siesta/stages.py, warm-files.toml, deck_record.py,
spectra/siesta_vibration.py, parse/engines/siesta_fc.py, spectra/vibrational_analysis.py,
spectra/normal_modes.py, atom_permutation.py, transport/sort.py, validation/spectra.py,
validation/siesta.py, validation/metadata.py, spectra/displacement_sweep.py.
Read in part: lib/spectra/core.js, _shared.py, jobset/prep.py (non-transport), script_emit.py,
siesta_reader.py, validation/__init__.py, runwrap.py (ranges), task-setup/viewer.js (ranges),
identity.py, materialize.py, sidecar.py, contract.py, parse/fdf.py, methods.py, results.py,
pyscf/vibration_deck.py, _cli.py, submit.py, summarize.py, agreement.py, chemistry.py,
files.py, parse.md, job-contracts.md.

### B. Parameter trace (45 template items + the kind's own lines) — see the hand-back for the
full table; verdicts: all "ok" except #23 relax_type (C4, C6), #24 relax_steps (C5, C12),
#27 fc_displacement (C5, C6), #29 write_forces (C1), #30 write_coor_step (C1),
#37 continue_retries (C3); FC start-state lines right but their stated reasons C7/C8/C9;
bench-marks MD.Steps row false (C12).

### D. Checked and found correct (summary)
FC keywords/units/defaults; the .FC format and its reader; declining .XV; held atoms (sort,
permutation, constraints, all-held refusal); whole-body motions removed (0/3/6); normal modes
and thermochemistry; geometry carry and relaxation record; prep refusals; displacement sweep;
wrapper `loads` check and finish ordering; form recommendations and template writing; every
shared keyword present in SIESTA 5.4.2 with its unit; several help claims confirmed.

### E. Verification by the main session (2026-09-29, code at 03c52cfa, SIESTA 5.4.2 e486d12)

| # | verdict | read |
|---|---|---|
| C1 | CONFIRMED defect | siesta_fc.reference_frame_of needs forces+coords; write_subs.F:782-784 prints rows only if writeF (read_options.F90:1858 `Write.Forces`, default outlng); outcoor.f:88 returns without writec; no check names either flag in validation/ |
| C2 | CONFIRMED defect | _stationarity: remedy text fixed at "untick `already_relaxed`" whatever `ladder_relaxation`; § 5.8 + sidecar.py:418-422 give "continue the `relax` stage" |
| C3 | CONFIRMED risk | save_density_matrix.F90: idyn 6 writes DM only at istp 1; m_new_dm.F90:136-144 forces DM_init (read the undisplaced .DM) every FC step; siesta_init.F: idyn 6 inicoor=0 -- a retry re-walks every step from the same start |
| C4 | CONFIRMED risk | relax_type calculations=[optimization, vibration], choices incl. Verlet/Nose/none; no vibration check on it |
| C5 | CONFIRMED risk | ofc.f90: tmp = Ang**2/eV/dx; ranges advisory (metadata.py warn); relax_steps 0 -> MD.Steps 0 |
| C6 | CONFIRMED premise -- DESIGN DECISION (user) | viewer.js seeds every row from the `stage` group; prep._vibration_block takes relax_force_tol of the FC stage as the finish's criterion |
| C7 | CONFIRMED | m_new_dm.F90 comment: FC re-reads the un-displaced DM; read_DM = UseSaveDM |
| C8 | CONFIRMED | read_options.F90:824-829 UseSaveDM default .true.; struct_init.F:84 UseSaveXV default UseSaveData (.false.); read_options.F90 UseSaveCG default usesaveddata |
| C9 | CONFIRMED | real deck aubdtauvib_02_freq.fdf:17 "CONTINUES: SIESTA reads the .XV / .DM" vs :494 `MD.UseSaveXV .false.`; runwrap header keyed on UseSaveDM alone |
| C10 | CONFIRMED | runwrap.py:2081 declarations(engine=) only; parse.md § 5d.3a says per engine, calculation and stage |
| C11 | CONFIRMED | deck header `jobset launch run 02_freq`; resolve_stage_ref: exact name or #N; every launch path goes through it |
| C12 | CONFIRMED | real FC deck: :140 "Without this block SIESTA relaxes every atom", :743 MD.TypeOfRun tip, :747 "after a successful relaxation", :812 MD.Steps bench row, :13 "Perfectly good" (skips the finish) |
| C13 | CONFIRMED | note unconditional; n_rigid in scope |
| C14 | CONFIRMED | warm-files [vibration] has no .CG; relax deck text "One field in the description decides" on a kind with no `restart` item |
| C15 | CONFIRMED | stages.md: names compared case-insensitively everywhere; vibration_render_kind and resolve_stage_ref compare exactly |
| C16 | CONFIRMED | form-schema.js:698/705 "" -> null; _shared.py float(None) TypeError; build.py returns 400 with the raw text; _shared.py:1316-1319 comment says otherwise |
| C17 | CONFIRMED | files.py:1671 mkstemp + os.replace keeps 0600; persist.py already owns an atomic writer that widens the mode |
| C18 | CONFIRMED (sampled) | task_setup.html "READ-ONLY ... Saving is NOT wired"; spectra.html title "Raman / IR"; config/siesta.py docstring 0.02/0.05 + "SIESTA silently ignores ... all on" |

---

## Review — Spectrum tab on PySCF (vibration: Phase 0 relax → Hessian → IR/Raman → thermo)

Code at 03c52cfa;
PySCF 2.14.0, geomeTRIC 1.1.1, pyscf-properties 0.1.0 in molbuilder-pySCF.
Reference script: projects/claude-au-bdt-au/spectrum/aubdtau-pyscf/aubdtau_01_freq.py.

### C. Findings (as reported; each to be verified before acting)

Defects
1. **C1 critical** — geomeTRIC out of steps recorded `converged: true`: PySCF catches
   GeomOptNotConvergedError (geometric_solver.py 161-169), `optimize()` drops the flag
   (191-192); `assert_convergence` covers one step's SCF only (92-93). vibration_deck.py
   379-384 sets converged True; halt never halts; `continue` (relax_policy.py 62-64) retries
   only a step-SCF failure and from the input geometry; the optimization deck shares it
   (input.py 1276-1332). Docs: pyscf.md 265-271, 744; vibration.md 516-518; catalogue
   2096, 2114; tuning.md 913-916; normal-modes.md 835; validation/spectra.py 514-516.
2. **C2** — UKS/UHF: mo_energies_eh written 2-D (emitters 1007, 1019), reader needs 1-D
   (results.py 781-785) → /api/spectra/load 400; _mo_window slices the spin axis
   (1695-1708, 1751-1753); pyscf.md 634-636 promises stability() the deck never calls.
3. **C3** — PCM: held-atom Hessian (emitters 514-522), IR-only analytic block (558-576) and
   Raman α (1493-1498) omit with_solvent.hess / equilibrium_solvation; preflight says the
   opposite (validation/spectra.py 415-419; scf_setup.py 166-170).
4. **C4** — (762ca298) Methods count and R7 note use struct.axis_kind (methods.py 455-460;
   validation/spectra.py 200-203); the deck uses isolated (emitters 1221-1223): periodic
   structure with 0-2 held → wrong stated count.
5. **C5** — def2 ECP false belief left in text: chemistry.py 234-235, catalogue basis help
   1215 "ECPs bundled to Rn", pyscf.md 520-521.

Risks
6. **C6** — blank max_memory_mb → 4000 (VibrationConfigView 149-157) not "machine max at
   prep" (catalogue 881); overrides PYSCF_MAX_MEMORY; XC-derivative arrays (~36+12 GB here)
   unstated; R8 advisory overstates.
7. **C7** — SOSCF partial: relaxation's DIIS SCF before newton() (deck 243-252); displaced
   SCFs DIIS only.
8. **C8** — IR help "nearly free" (catalogue 1788, 1807) false with atoms held (6·N_free).
9. **C9** — layout.py 212, 216 `{value:.0e}` rounds conv_tol / conv_tol_grad to one digit.
10. **C10** — `_optimized.xyz` pair copies the input's SIESTA info.relaxation/calculation
    (input.py 1513-1528, 1623-1625).
11. **C11** — Methods omits ECP and PCM; auxbasis sentence ignores a set auxbasis; fixed grid
    sentence; LYP uncited.
12. **C12** — Task setup hover shows the general default as "Recommended" on a vibration
    folder (build.py 1797; viewer.js 1778-1779).
13. **C13** — explicit mode numbers not bounded at prep; recorded before filtering; three
    false comments (emitters 1710-1715; selection.py 82-88, 95).

Doc drift
14. **C14** — core.js 836-838, web/spectra.md 477-479 "PySCF's gate refuses" periodic.
15. **C15** — web/spectra.md 252-253 display eigenvector by per-atom length vs largest
    Cartesian component.
16. **C16** — normal-modes.md 92-95, 1513-1515; emitters docstring 25-27, 51-53 still name
    harmonic_analysis (retired 2026-09-23).
17. **C17** — handover-procedure.md 92, 97 say `level`; code uses `severity`.

Nits
18. C18 wB97X-D named in help, refused by PySCF 2.14. 19. C19 dipole origin comment.
20. C20 build() without dump_input=False. 21. C21 wasted work (alpha_eq, redundant SCF,
double rebuild, "atom k/N_ATOMS"). 22. C22 T=0 / P=0 warn only → inf/nan kills the writer.
23. C23 wording/stale text (list in the hand-back).

Known (not counted): V1.9, V1.10, V1.11, V1.12, V1.14 + A1.16 U7, V1.17, V1.31, V1.33,
V1.37, W41's ir_fd_step_ang.

### E. Probes proposed for the e2e step
C1 water geom_max_steps=3; C2 a radical to the Results tab; C3 water in PCM (held / IR);
C4 CO₂ marked periodic; C9 scf_conv_tol 2.5e-9.

---

## Review — Transport, the five rungs (seed → leads → device → transmission → record)

Code at 03c52cfa;
SIESTA 5.4.2 source. Reference: projects/claude-au-bdt-au/transport/aubdtau-T (TD8).

### C. Findings (as reported; each to be verified before acting)

Defects
- **F20 severe** — spin kinds the engines refuse pass every check: CAPABILITY["siesta"] all
  four for every kind incl. transport (electronic_state.py 81-87); ES4 passes; Spin.Fix
  written on every rung (deck.py 186, 223, 415, 539). Engine: read_options.F90 938-940 and
  m_transiesta.F90 119-120 die on nspin>2; m_tbt_hs.F90 244-245 same; ts_options 1116-1120
  die on FixSpin in TSmode (leads only warn 1121-1123). chemistry-correctness.md 536-542 ✅.
- **F30 severe** — gather check compares an upstream attempt's deck with the stage folder's
  last-rendered deck (prep.py 2089, 2126-2130), not a render from the current template;
  re-pointed citation / edited shared value + prep device only → stale .TSHS copied, and
  .gathered-from says consistent. transport.md 1375-1376, 2655, 2659-2662.
- **F25** — citation with no deck/record: panel sends SZ / LDA / PBE and kgrid 0 0 0 as
  choices (blueprints/transport.py 639-640 default None; form-schema.js 170-182, 241;
  core.js 753-762; citation_defaults 216, 249-250); contract: blank = not chosen
  (transport.md 2057-2062; core.js 743-746).
- **F1** — tbt_k_grid: rung tab shows [1,1,1] (catalogue default), template/deck run [3,3,1]
  (citation_defaults 103-104); an explicit 1 1 1 cannot be sent; a shared kgrid change after
  _apply_kgrid does not move TBT.k.
- **F24** — scf_must_converge, block_size, parallel_over_k (optional, no default) never sent
  from a rung tab (form-schema.js 896-905; core.js 643-656).
- **F3** — Task setup run card offers execution items as pins; _prep_transport refuses any
  pin (prep.py 1744-1750), incl. use_gpu which task-setup.md 409-410 says is set there;
  build.py 1698 no kind filter (restart part known).
- **F2** — kgrid_displacement on the shared panel; deck.py 471-473, 707-710 hard-code 0.0.
- **F27 + F13** — bias scan attempts live in v*/ folders; record (record.py 193-194, device),
  Task setup attempt count (build.py 2217) and "already under way" (prep.py 2715) look only
  in the stage folder.

Risks
- **F26** extra `*-electrode` label → three-electrode deck that dies at the device.
- **F4/F34** Task setup rename/add rungs, shape switch; renamed seed silently drops the seed DM
  (prep.py 2075-2076); task-setup.md 148 says stages and shape fixed.
- **F5** device override of bias_voltage_v honoured in a single-bias run (resolve.py 364-392).
- **F15** bias list: duplicates and out-of-range pass (task.py 491-501).
- **F35** no check that the T(E) window covers ±V/2 + a few kT.
- **F33** foreign deck's omitted keywords get catalogue defaults, not SIESTA's.
- **F14** "ladder loosens" advisory on transport's per-rung dm_tolerance.

Doc drift
- F21 invented 2 V threshold (validation/__init__.py 432-442 vs transport.md 834-835).
- F31 deck.py 316-318 says DM.UseSaveDM default false (engine: true).
- F29 T(E) relative to the lead's E_F in inspectors/transport.js 139-142, 288-289,
  transport.md 1022 (engine: the device's Ef, m_tbt_hs.F90 417).
- F28 device record lacks E_F, iterations, poles; energy is `ts: Total`, not comparable to
  the seed's `siesta: Total`.
- F8 tabs.md 214, 235-238; F18 write_hs help (catalogue 763); F32 species-order rationale;
  F7 "seconds" for the transmission (973 s measured).

Nits: F6, F9, F16, F17, F36 new; known F10, F22 (§ 5u.4), F11, F12 (step 3), F23 (§ 5v),
stale "no template" sentences (+3: task-handover.js 15-16, prep_inputs.py 449-450, 716-719),
G(E_F) at finite bias (TD7), bias input not saved (step 8). F19 → TD12's header should name
Spin and SystemLabel too.

### D. Checked and found correct (summary)
Poles 123; TS.Voltage 0; electrode positions 1-27 / 58-84; semi-inf −A3/+A3; mu binding;
tbtrans outputs; TBT.k list; eta 1 meV / 0.1 meV; TS.Elecs.Bulk read by both; 10 K floor;
buffer block; window line/mid-rule; .gathered-from; neutral junction; lead extraction (I6,
z-period 7.20187 = span + spacing 2.4006); I12 per lead; seed DM read by the device; kz=1 on
the device; record energies relative to E_F; role items skipped; foreign_overrides; resolve
refusals; pole check; no transport presets; hand-over and bench refused for transport.

---

## Verification by the main session — the other four tracks *(2026-09-29, at `03c52cfa`)*

Each row was re-read in the code and, where the finding is about the engine, in
the engine's own source (SIESTA 5.4.2 `e486d12`; PySCF 2.14.0, geomeTRIC 1.1.1,
gpu4pyscf 1.8.1 in `molbuilder-pySCF`). A finding not in these tables was
reported and is re-read when it is fixed.

**Structure optimization on SIESTA**

| # | verdict | read |
|---|---|---|
| C1 | CONFIRMED | `run_uses_device` reads the condition and the template, never a stage override; `use_gpu` (kind deck, group staging, not allocation) passes `_column_items`' membership rule, so it can be a column; `del out["gres"]` when the answer is no |
| C2 | CONFIRMED | `system_label`, `psml_lib`, `species_order` carry `shared = ["transport"]` only, so `_column_items` offers them on an optimization; `ioxv.F` reads each atom's species index (`isa`) back from `.XV` |
| C3 | CONFIRMED | no code realigns `BlockSize` (`realign` / `pow2` found only in the template reader and BENCH-MARKS); `diag_option.F90` `elpa_gpu_block_size` snaps DOWN to a power of two |
| C4 | CONFIRMED | `index.html`: "leave Start from on clean — the default, and none of the three flags is written"; `SiestaConfig.restart` defaults to `continue`, and both states are written |
| C6 | CONFIRMED | `read_options.F90`: `converge_EDM = fdf_get('SCF.EDM.Converge', .false.)` against the help's "on by default" |
| C7 | CONFIRMED | `atom.F`: `eshift_default = 0.01_dp` against the help's "SIESTA's own default is 0.02 Ry" |
| C9 | CONFIRMED | `copy_psml` is read by no code (a `SiestaConfig` field, a comment in `template.py`, a docstring in `validation/metadata.py`) |

**Structure optimization on PySCF**

| # | verdict | read |
|---|---|---|
| C1 | CONFIRMED | `geometric_solver.kernel` catches `GeomOptNotConvergedError` (`conv = False`) and `optimize` returns `kernel(...)[1]`; geomeTRIC `optimize.py` sets FAILED at `maxiter` and raises; `assert_convergence` guards one step's SCF (l.92-93); the deck calls `optimize` under every policy (`input.py` `_mb_run_optimization`) |
| C2 | CONFIRMED | gpu4pyscf `scf/uhf.py`, `scf/hf.py`, `scf/rohf.py`: `stability = NotImplemented`; the deck catches `(NotImplementedError, AttributeError)` and promotes to GPU before the check |
| C3 | CONFIRMED | no `pyberny` in the env's site-packages; `berny_solver` hands kwargs to `Berny(...)`, whose criteria are `gradientmax` / `gradientrms` / `stepmax` / `steprms`; `validation/pyscf.py` has no refusal (the vibration kind's is in `validation/spectra.py`) |
| C4 | CONFIRMED | geomeTRIC 1.1.1 keeps `config/log.ini` (with the file handler), not `log.ini` beside `optimize.py`, so `geometric_solver` substitutes its console-only `log.ini`; no `*_geom.log` under `projects/` |
| C5 | CONFIRMED | `pyscf/layout.py`: `f"mf.conv_tol  = {value:.0e}"` and the same for `conv_tol_grad` |

**Spectrum on PySCF**

| # | verdict | read |
|---|---|---|
| C1 | CONFIRMED | as Structure optimization on PySCF C1; the vibration deck binds `optimize as _geom_opt` and writes `converged = True` after it under `halt` and `continue` |
| C2 | CONFIRMED | `CAPABILITY["pyscf"]["vibration"]` includes `unrestricted`; `MO_ENERGIES_EQ = mf.mo_energy` (2-D for UKS) is written whole; `SpectraResults` refuses a non-1-D array; `_mo_window` slices `_mos[_lo:_hi]` on the spin axis |
| C3 | CONFIRMED | PySCF's PCM Hessian `kernel` solves under `equilibrium_solvation=True` and adds `with_solvent.hess(dm)`; the held-atom route calls `hess_elec(atmlst=)` + `hess_nuc` (neither); the IR route's `proc_hessian_` restores dispersion only |
| C4 | CONFIRMED | `methods.py` and `validation/spectra.py` pass the structure's `axis_kind` to `rigid_motions`; the deck passes `('isolated',)*3` |
| C5 | CONFIRMED | `validation/chemistry.py`: "PySCF auto-loads a Stuttgart ECP … when basis='def2-SVP'"; catalogue basis help "ECPs bundled to Rn"; `pyscf.md` "built-in ECPs up to Rn" |
| C9 | CONFIRMED | as Structure optimization on PySCF C5 |

**Transport**

| # | verdict | read |
|---|---|---|
| F20 | CONFIRMED | `read_options.F90` and `m_transiesta.F90`: TranSIESTA dies on `nspin > 2`; `m_tbt_hs.F90`: tbtrans the same; `m_ts_options.F90`: `Fixing spin is not possible in TranSiesta`; `CAPABILITY["siesta"]` grants all four treatments to every kind, transport included, and no check refuses a fixed count there |
| F30 | CONFIRMED | `prep.py` gather: `current_deck = up_dir / <stage deck>` — the upstream stage folder's last render, compared by `same_calculation`; only the rung asked for is rendered |
| F25 | CONFIRMED | the shared route sets `default = None` on an unanswered cited row; `makeSelect` adds a blank option only for `optional` / `null_option`, so a required enum shows and returns its first choice; `makeTriple` falls back to `[0, 0, 0]`; `_sharedValues` sends every field |
| F1 | CONFIRMED | the rung surface is built with no citation, so `tbt_k_grid` shows the catalogue's `[1, 1, 1]` while the template holds the cited `[3, 3, 1]`; `diffFromDefaults` sends only a difference from `[1, 1, 1]`; the person's shared `kgrid` is laid over after `tbt_k_grid` was derived |
| F24 | CONFIRMED | `scf_must_converge`, `block_size`, `parallel_over_k` are `optional` with no default; `diffFromDefaults` skips a field with no default |
| F3 | CONFIRMED | `run_inputs` turns every non-machine run-card item into a pin; `_prep_transport` refuses any pin |
| F2 | CONFIRMED | both transport k-grid blocks write `0.0` in the displacement column; no transport module reads `kgrid_displacement` |
| F27 / F13 | CONFIRMED | the record looks in the per-point folders for the transmission only; Task setup's count and prep's "already under way" read the stage folder |
| F21 | CONFIRMED | `validation/__init__.py`: "above the ~2 V linear-response limit" |
| F29 | CONFIRMED | `inspectors/transport.js`: "T(E) is measured relative to this" (the leads') and "which is the LEAD's" |
| F31 | CONFIRMED | `transport/deck.py`: "SIESTA's default for this keyword is false"; `read_options.F90`: `DM.UseSaveDM` defaults to `.true.` |
