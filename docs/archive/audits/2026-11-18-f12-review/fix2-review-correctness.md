*Verdict: closed in 2.77.5.* The correctness review of F12 fix release 2, read before its release: every finding was fixed or measured in 2.77.5 (its changelog row). Kept as the record of what the review found.

# Correctness review — F12 fix release 2 ("the deck writes the value the person set")

Reviewer lens: does each fix do what the person set, on every road that reaches it.
Read whole: `pseudos.py` (check_coverage and its severities), `validation/siesta.py`
(the pseudopotential check), `cli.py pseudo check`, `siesta/layout.py`,
`script_emit.render_deck`, `template.py` (shared / why_shared / why_unread),
`periodicity_gate.py` with the Cell page (`modify/periodicity.js`,
`lib/molview/model-jobs.js`), `parse/engines/pyscf.py`, the PySCF decks
(`pyscf/input.py`, `pyscf/vibration_deck.py`, `scf_setup.py`), `siesta/seam.py`,
`siesta/makov_payne.py`, `siesta/vibration_phases.py`, `spectra/siesta_vibration.py`
(the finish, the by-hand finish, the optical record), `spectra/mode_frames.py`,
`spectra/raman.py` (`optical_data_of`), `jobset/prep.py` (the launched-stage gate),
`pyscf/script_input.py` (the same-calculation reading), the plan and its archive.
Measured with the molbuilder env's Python on decks prepared in memory; scratch
scripts in `scratch-correctness/`. Nothing under the checkout was written.

9 findings. The first five are wrong results or a new stop on a road the person
uses; the last four are smaller.

---

## 1. A PySCF relaxation's trajectory now gives its last frame — and, near convergence, its last few frames — another evaluation's SCF history

- **Where:** `molbuilder/parse/engines/pyscf.py:221-227` (`_scf_by_frame`), with the
  deck's own SCF after the relaxation, `molbuilder/pyscf/input.py:1166-1167`.
- **What is wrong:**
  ```python
  for i in range(len(energies) - 1, -1, -1):
      ...
      k = next((k for k in range(end - 1, -1, -1)
                if runs[k] and abs(runs[k][-1]["energy"] - energies[i])
                <= _SAME_ENERGY_EV), None)
  ```
  Each frame takes the *latest* run of PySCF's log that ends within 1 µHa of its
  energy. But every PySCF relaxation deck runs one more SCF after geomeTRIC returns,
  at the relaxed geometry, from a cold or stale density:
  ```python
  "    mf.reset(mol_eq)"
  "    mf.kernel(dm0=dm_prev)"
  ```
  That run is in the same log and ends at the last frame's energy, so the last frame
  takes it. The frame before then searches only runs before it and takes geomeTRIC's
  *last* evaluation whenever the two energies are within 1 µHa — and that is what
  convergence makes them: `geom_etol` ships at 1e-6 Ha. The converged tail shifts by
  one, which is the P-D2 symptom again, at the other end of the trajectory.
- **Scenario:** a person relaxes any molecule on PySCF, closed shell or open, and opens
  its trajectory. The last frame shows the 10–12 cycles of the deck's cold
  re-convergence instead of the 3–4 warm cycles geomeTRIC's last step took. When the
  last steps changed the energy by less than 1 µHa, each of those frames shows the
  next step's SCF. For a closed-shell run this is a regression: the old pairing by
  position gave the last frame its own run.
- **Measured** on `_scf_by_frame` itself, with runs laid out as the deck logs them:
  - three frames plus the trailing re-convergence →
    `['eval0', 'eval1', 'final-reconverge']`;
  - five frames whose last three energies are within 1 µHa, plus the
    re-convergence → `['eval0', 'eval1', 'eval3', 'eval4', 'final-reconverge']`;
  - the same frames without the trailing run (a wall-stopped relaxation) → right.

  That the re-convergence lands in the same log is traced, not run: it is the same
  `mf`, reset onto geomeTRIC's molecule, which writes to the same output. The new e2e
  test (`test_each_frame_is_given_its_own_steps_scf`) checks only that the energies
  agree, so it passes either way.
- **The doc** `model/parse.md` § 5a ("the log's run that ends at the step's own
  energy") has the same gap: two runs end at that energy.
- **Smallest fix:** match forward. Frame *i* ≥ 1 takes the *earliest* matching run after
  the run frame *i−1* took. Frame 0 takes the latest matching run before frame 1's run,
  which still skips the stability check's runs. A trailing run after the last
  evaluation is then never chosen, and near-equal energies resolve in log order.

## 2. G-D1 is not fixed on the road it was found on: the Cell page sends the box rounded, and the gate compares it at 1e-9 Å

- **Where:** `molbuilder/periodicity_gate.py:380, 385-397` (`_same_box`, `atol=1e-9`),
  with `molbuilder/web/static/modify/periodicity.js:64, 494, 596`.
- **What is wrong:** the page fills the cell boxes rounded to 6 decimals, and Apply sends
  those boxes back as typed:
  ```js
  function round(n) { return Math.round(Number(n) * 1e6) / 1e6; }
  setIdle(cellInputs[r * 3 + col], m ? round(m[r][col]) : "");
  cell: explicit ? stagedCell() : null,
  ```
  The gate compares the result with the structure held at full precision:
  `np.allclose(..., rtol=0.0, atol=1e-9)`.
- **Scenario:** this is the review's own scenario. A person saves a structure from a
  finished charged SIESTA run (`xv2xyz --from-run` or the Results tab). Its cell is the
  `.XV`'s, written in Bohr to 9 decimals and read back in Å, for example
  `13.000768160976383`. They open the Cell page and press Apply without changing
  anything. The cell goes back as `13.000768`, the box "moved" by 1.6e-7 Å, and the
  structure is marked modified. A later blank charge again drops the recorded −1. The
  browser also adopts the rounded cell, so the box really has been rewritten.
- **Measured:** `apply_edit` on a structure with an `.XV`-style cell and a recorded
  `info.calculation`:
  - payload rounded the way the page rounds it → `structure_modified: True`;
  - full-precision payload → `None`.

  The new test row `block-unchanged` (`_its_own_box`) sends full precision, although
  its docstring says "each field as the page fills it". `web/molview.md`'s new
  sentence, "an Apply that changed nothing keeps the run's record", does not hold for
  such a structure.
- **Smallest fix:** compare at the precision the page sends (|Δ| ≤ 5e-7 Å for the cell,
  origin and vacuum). Or have the page send the structure's own value for a field
  nobody typed in.

## 3. The by-hand `finish` writes the optical data file into the run's root and cites it from a result that is not there — its Raman cannot be opened

- **Where:** `molbuilder/spectra/siesta_vibration.py:582-588` (`_finished`) and
  `:1119-1120` (`_write_optical`); the reader is `molbuilder/spectra/raman.py:459`.
- **What is wrong:**
  ```python
  res = _with_raman_phase(res, run, _write_optical(here, index, res), PHASE_COMPLETE)
  out = Path(deck).parent / compose(run.label, ROLE_SPECTRA)
  ```
  The optical data file goes to `here.run_dir / compose(here.label, ROLE_OPTICAL)`, the
  run's root. The result goes beside the force-constant call's deck, in the call's
  folder below the run. The result cites the file by name only (`data_file`), and the
  one reader looks for it beside the result:
  ```python
  data = path.parent / result.optical["data_file"]
  if not data.is_file(): raise RamanError(f"... cites {data.name}, which is not beside it")
  ```
- **Scenario:** a SIESTA vibration with Compute Raman on has its finish
  fail, for example because the job env lacks netCDF4. The person follows § 5.5 and runs
  `mb_vibration.pyz finish <deck> <output>` in the force-constant call's folder. The
  result it writes now says Raman is complete, which is F-D1's fix. Opening its Raman,
  through `postprocess` / `raman_view_of` or the spectra viewer on that file, is refused:
  "cites t.optical.nc, which is not beside it".

  The by-hand step has also rewritten the run's own root `<label>.optical.nc`.
  `engines/vibration.md` § 5.5 (line 1565) says this step writes beside the call's deck
  and that "its root holds no result". The `main()` docstring (`:1307`) still says the
  verb works "for anything holding a finished force-constant call". It now refuses a
  folder that is not a call of a run (`_here`).
- **Traced** (writer, reader and both paths read). Not run: it needs a finished SIESTA
  Raman run. The new e2e test compares only `phase_relaxation` / `phase_raman`.
- **Smallest fix:** in the by-hand verb, write the optical payload beside the result it
  cites, in the call's folder: give `_write_optical` the folder to write into. Leave the
  run's root alone, as the contract says.

## 4. The D10 change makes `prep` refuse every PySCF stage launched before this release ("not the calculation its runs ran")

- **Where:** `molbuilder/pyscf/input.py:696-697` (`emit_save_call(..., converged="_GEOM_CONVERGED")`),
  read by `molbuilder/pyscf/script_input.py:227` (`script_differences`) at the
  launched-stage gate, `molbuilder/jobset/prep.py:2932-2937`.
- **What is wrong:** `save_optimized_xyz` ships `true`, so every PySCF optimization deck
  carries the save call. Its third argument, which is only the file's title words,
  changed shape. The same-calculation reading compares every statement that does not
  only report, and a `_save_structure(...)` call is compared.
- **Scenario:** a person launched a PySCF relaxation under 2.77.4 and it stopped at its
  wall. They update molbuilder, then prep the stage again to give it a longer wall,
  which `job-request.md` calls the one thing a launched stage is prepared again for.
  `prep` refuses: "stage 'coarse' was launched, and its deck rendered now is not the
  calculation its runs ran". The way out it names redoes the stage, losing the run's
  continuation, although nothing it computes changed.
- **Measured:** `script_differences(old, new)` on a deck rendered now and the same deck
  with the save line as 2.77.4 wrote it gives one difference: the run had
  `_save_structure(mol_eq, …, _MB_WORDS['optimized_geometry'])`, it renders
  `… _MB_WORDS['optimized_geometry'] if _GEOM_CONVERGED else _MB_WORDS['stopped_geometry'])`
  now. D12's new stability lines do the same to every open-shell deck, but those do
  compute something new (one more stability analysis).
- **Smallest fix:** choose the words in a statement the reading sets aside, an
  assignment into `_MB_WORDS` (in `REPORTS`). For example,
  `_MB_WORDS['optimized_geometry'] = (… if _GEOM_CONVERGED else …)` just before the call.
  The call line then stays exactly as 2.77.4 wrote it.

## 5. D10 is fixed on the optimization deck only: a PySCF vibration's relaxation still titles an unconverged geometry "Optimized"

- **Where:** `molbuilder/pyscf/vibration_deck.py:343-344`.
- **What is wrong:**
  ```python
  if getattr(cfg, "save_optimized_xyz", False):
      out.append(emit_save_call("mol", PYSCF_OPTIMIZED, names, indent="    "))
  ```
  There is no `converged=`, although `_geometric_converged` is in hand three lines above.
- **Scenario:** a PySCF vibration, `already_relaxed` unticked,
  `on_nonconvergence = proceed`, whose relaxation misses geomeTRIC's criteria. The
  result's relaxation warning says "did not meet geomeTRIC's criteria". The saved
  `<job>_optimized.xyz` is titled "Optimized geometry (PySCF)", which is D10's exact
  defect, on the other PySCF deck.
- **Measured:** the deck rendered with `on_nonconvergence='proceed'` writes
  `_save_structure(mol, _mb_outfile(JOB + '_optimized.xyz'), _MB_WORDS['optimized_geometry'])`
  with no condition.
- **Smallest fix:** `emit_save_call("mol", PYSCF_OPTIMIZED, names, indent="    ", converged="_geometric_converged")`,
  or the reported-statement form of finding 4.

## 6. Where no family is compared, any author difference is called a "minor mismatch, ~1–2 kcal/mol": every species of every vdW-DF run is warned falsely

- **Where:** `molbuilder/pseudos.py:845-856` (the authors compare), reached from
  `validation/siesta.py:132-141` and `cli.py:277-281`.
- **What is wrong:** D1 now leaves the family uncompared for `VDW`, as the contract says:
  "it is run on GGA ones". The next check still compares authors and calls any
  difference same-family and minor:
  ```python
  if expected_xc_authors and info.xc_authors != "unknown" \
          and info.xc_authors.lower() != expected_xc_authors.lower():
      ... "minor mismatch; results may differ from publication-grade by ~1-2 kcal/mol."
  ```
- **Scenario A:** a person prepares the vdW-DF run D1 unblocked (`VDW` / `DRSLL`) on
  PseudoDojo PBE files, the set C2 says vdW-DF runs on. Every species is warned
  "generated for PBE; calc requests DRSLL -- minor mismatch …". An LDA file in that run
  is warned the same "minor" way.
- **Scenario B:** a CI script uses the invocation the chapter documented until now,
  `pseudo check <dir> --xc PBE`, without the new `--xc-functional`, on an LDA set. It is
  now told "minor mismatch ~1-2 kcal/mol", where it was told "XC-FAMILY mismatch: bond
  lengths and energies will be silently wrong".
- **Measured** with `check_coverage` on `tests/fixtures/psml/pseudodojo-v0.4`:
  - `VDW` / `DRSLL` → `xc_mismatch` on S, N and Au, with that sentence;
  - `LDA` / `CA` → `xc_family_mismatch`, correct.
- **Smallest fix:** compare authors only when the family was compared and matched, that
  is when `expected_xc_family` is set and equals the file's.

## 7. F-D2 reads "stated relaxed" from the calculation's description as it stands now, not from the run

- **Where:** `molbuilder/spectra/mode_frames.py:106-109` (`_relaxation_record`).
- **What is wrong:**
  ```python
  calc = open_calculation(run.place.root)
  cfg = stage_config(calc.task, command_stage(run.stage), calc.template_text)
  if not getattr(cfg, "already_relaxed", False):
  ```
  The run's own statement exists on both engines: its result's
  `relaxation.already_relaxed` (`engines/vibration.md`, line 484: "in both results").
  For SIESTA, the code before this release read the run's own deck block.
  `stage_config` returns `None` for a stage the description no longer holds, which is
  then read as "not stated".
- **Scenario:** a person ran a SIESTA vibration with `already_relaxed` ticked. Later
  they untick it or rename the stage in the description, or the description does not
  currently load. They then write a frame set from that run with
  `molbuilder frameset from-mode`. Frame 0 then cites the vibration's own deck record
  as the relaxation's, for example a 1×1×1 k-grid where the relaxation used 5×5×1.
  That is the transport mis-citation § 5.10 ③ was written to stop. If the description
  does not load, the command fails on a finished run.
- **Plausible:** traced. Whether the description of a launched calculation can change
  without a restore was not established.
- **Smallest fix:** read `res.relaxation["already_relaxed"]` from the run's result,
  which the caller (`mode_frames.py:303`) already holds, on either engine.

## 8. D12's new final stability verdict uses the test the loop's own comment rejects for degenerate shells

- **Where:** `molbuilder/pyscf/input.py:1028-1031`.
- **What is wrong:** after the restarts run out, the state is asked once more and judged
  by
  `_MB_STABLE = bool(_mb_np.allclose(_mb_host(_internal), _mb_host(mf.mo_coeff)))`.
  The loop above says why that is not the criterion: "Degenerate shells … are rotated
  freely within the degenerate space, so a coefficient comparison flags them forever …
  The energy is the criterion; the coefficients are not."
- **Scenario:** an O2-like open shell whose three restarts each lowered the energy.
  The last check hands back rotated degenerate orbitals, and the log says "WARNING --
  still internally unstable after 3 restarts … treat the geometry and any frequencies
  as provisional", although no lower state was found. It is a warning only; the run
  goes on.
- **Plausible, rare:** the code is read. The path was not run.
- **Smallest fix:** in the `else:`, use the loop's own rule: one more restart, and
  "stable" when the energy does not fall by `_MB_STABILITY_ETOL`. Or say the last state
  was not confirmed rather than calling it unstable.

## 9. The plan states "No check or warning stops a build or a calculation" as in force; the code still has such stops, and the plan lists none of them as open

- **Where:** `docs/plans/plan.md:60-62`.
- **What is wrong:** the sentence reads as true of the code today. Prep still refuses
  scientific judgements. Two examples:
  - `validation/geometry.py:44-48` — atoms closer than 0.3 Å, "SCF will diverge",
    `error`;
  - `validation/spectra.py:383-397` — a PySCF vibration with a PCM solvent and IR on is
    refused. Measured: prep of such a deck raises `ValidationError` naming
    `config.solvent`.

  None of these is a round-1 defect in the module queue, so no release on the plan
  owns turning them into warnings.
- **Scenario:** a reader of the plan, or the next release's author, takes the rule as
  done, and the remaining stops are never listed or worked.
- **Traced and measured (one case).** Low: a plan sentence.
- **Smallest fix:** say it is the rule the fix releases apply, and list the error-level
  checks still standing, or name the release that sweeps them.

---

## Checked and holding

- **D2:** the deck reads `DM.Tolerance 1.25e-04`, `SCF.H.Tolerance 5.0e-07 eV` and
  `DM.EnergyTolerance 6.25e-05 eV`, and `parse/fdf.py` reads them back exactly.
  `_exact` spells 0.00625 → `0.00625`, 5e-7 → `0.0000005` and 1e-4 → `1.0e-04`.
  The transport gather's same-calculation reading compares numbers as numbers, so an
  old `0.000100` equals a new `0.0001`. Only values the old rounding really changed
  differ, as they should.
- **D3:** `job_name` is shared for optimization and vibration (PySCF's only kinds) and
  states its `why_shared`. The override refusal (`template.py:2752`) and the Task
  setup's shared and rung surfaces read `shared_by_every_stage`. No `IDENTITY_ITEMS`
  is left.
- **D4:** with `verbose_comments` off, the SIESTA water deck drops from 1008 to 190
  lines and every emitted line is still present for the check gate. The transport
  gather now renders as prep does; comments are outside the same-calculation reading.
- **D5, D8:** the help now matches `atom.F`. The tips read
  "raise SCF.Mixer.History above its 8".
- **D9:** `solvent=' Water '` gives an optimization deck that compiles, with header
  `water (IEF-PCM, eps=78.3553)` and `mf.with_solvent.eps = 78.3553`. Every reader asks
  `solvent_of`.
- **D13:** the PySCF vibration's held roles are `force_constants` only, and the refusal
  wording follows.
- **D6:** the phase role reaches the seam (`prep.py:814`). The force-constant run's step
  0 is its reference (§ 5.5's "FC step 0"), and relaxation and optical keep −1.
  Benchmark trials (role `None`) keep −1, which is moot because their energies are
  capped SCFs.
- **D1's blocking:** what stops prep is `missing` / `misnamed` alone, the same set on
  every surface (`validation/siesta.py`, `cli.py`). The vdW-DF run now prepares.
- **P-D2's tolerance against PySCF's extra cycle:** the final energy and the last
  `cycle=` line differ by at most 1.4e-7 Ha even at `conv_tol` 1e-4 (water, RHF and RKS,
  measured). The 1 µHa match is not lost to it.
