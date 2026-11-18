# F12 review, round 1 — slice D: the template, the stages, the engines' decks, configuration, tuning

Checkout `the checkout` at f578482db (2.77.2). Read-only throughout.

**Chapters read end to end:** `engines/template.md`, `engines/stages.md`,
`engines/overview.md`, `engines/tuning.md`, `engines/optimization-postprocessing.md`,
`configuration.md`; `engines/siesta.md` and `engines/pyscf.md` were read whole by
two sub-reviewers (see Method).

**Code read whole:** `template.py`, `template_file.py`, `kind_table.py`,
`engine_table.py`, `resolve.py`, `deck_record.py`, `pseudos.py`, `persist.py`,
`config_dir.py`, `config/__init__.py`, `config/stages.py`, `optimization/`,
`bench/`. The SIESTA sub-reviewer read `siesta/` (every module) and
`config/siesta.py`. The PySCF sub-reviewer read `pyscf/` (all but `vibration_*`) and
`config/pyscf.py`. Callers and callees outside the slice were followed only to
trace a claim: `validation/stages.py`, `validation/siesta.py`, `jobset/prep.py`,
`jobset/ladder.py`, `jobset/model.py`, `jobset/prep_inputs.py`, `script_emit.py`,
`task.py`, the data tables, and the local SIESTA 5.4.2 tree.

**Method.** The catalogue claims (counts, markers, defaults, presets) were checked
by read-only Python that imports `molbuilder` with `-B` (no bytecode written),
using the in-memory catalogue, `accepts` / `why_not_in`, `check_coverage` on
`tests/fixtures/psml`, and `check_ladder_does_not_loosen` on in-memory configs.
The two sub-reviewers rendered decks in memory. I re-read the code lines behind
every sub-reviewer finding below, and the line numbers here are the corrected ones.
- One process breach: the PySCF sub-reviewer briefly wrote a helper script at
  `/tmp/f12pyscf/render.py`, outside the checkout, and deleted it. Nothing was
  written under `the checkout`, `~/.config` or `~/.local/state`.

**Result: 13 defects, 10 untrue sentences.**

---

## Defects

### D1. A vdW-DF SIESTA calculation can never pass prep: every PSML file is refused as an "XC-family mismatch"

- **Where:** `pseudos.py:608-612` (`_XC_FAMILIES`) and `:305-323` (`_LIBXC_MAP`).
  The check is in `check_coverage`, `:831-850`, read by the settings gate at
  `validation/siesta.py:130-155`.
- **Rule** (`science/pseudopotentials.md` C2, which this module serves): "Family
  mismatch (GGA vs LDA) → `xc_family_mismatch` → ERROR. Never correct … **False
  positives:** low, bounded by the libxc id table."
- **What the code does:**
  - `_XC_FAMILIES` maps the catalogue's offered `xc_authors` DRSLL and LMKLL
    (`xc_functional = VDW`) to the family `"VDW"`:
    `"VDW": ("drsll", "lmkll")`.
  - `_LIBXC_MAP` has no id that yields `"VDW"`. A PSML file's family is
    therefore always `GGA`, `LDA` or `unknown`.
  - So `expected_xc_family("DRSLL") == "VDW"` never equals a recognised file's
    family. `check_coverage` returns `xc_family_mismatch`, which is in
    `ERROR_STATUSES`, and the gate refuses the deck.
- **Measured** on `tests/fixtures/psml/pseudodojo-v0.4/N.psml`:
  - `xc_authors = PBE` → `ok`;
  - `DRSLL` and `LMKLL` → `xc_family_mismatch`, "N.psml was generated for GGA
    (PBE); calc requests VDW (DRSLL) -- XC-FAMILY mismatch: bond lengths and
    energies will be silently wrong".
- **Scenario:** a person picks *VDW* / *DRSLL* on the SIESTA form (both are
  offered choices) with the PseudoDojo PBE set. Prep refuses every species with
  an ERROR saying the run would be "silently wrong". No PSML set reaches any
  other answer, so an offered choice can never be prepared. vdW-DF is ordinarily
  run on GGA pseudopotentials.
- **Certainty:** read and traced, and measured.

### D2. The SIESTA deck rounds a value the person set, so the deck computes something else

- **Where:** `siesta/layout.py:222-227` (`_FMT`), applied at `:377-378`:
  `shown = format(param.value, fmt) if fmt else f"{param.value}"`.
  - `.6f` for `negf_neq_eta_ev`, `ts_elecs_eta_ev`, `tbt_elecs_eta_ev`,
    `tbt_contours_eta_ev`;
  - `.4f` for `bias_voltage_v`, `negf_eq_pole_ev`;
  - `.1e` for the three SCF tolerances.
- **Rule:** `template.md` § 5.4, "the deck says what it was given … write the
  value the job holds". § 6.6 obligation 4: "written into that rung's deck with
  the value that will apply".
- **What the code does** (the SIESTA sub-reviewer rendered these through
  `layout.line`):
  - `ts_elecs_eta_ev = 5e-7` is inside its range [0, 1] and draws no warning,
    yet is written `TS.Elecs.Eta 0.000000 eV`. The same holds for
    `negf_neq_eta_ev = 4e-7`. `template.md` § 3 says a written 0 means zero
    broadening, so this is another calculation.
  - A bias point of 0.00625 V is written `TS.Voltage 0.0063 eV`, in a folder
    named `v0.00625`.
  - Two points, 5e-5 V and 6e-5 V, get two folders but one `TS.Voltage 0.0001 eV`.
    The repeated-point refusal works on folder names, so it does not catch this.
- **Not a defect for the tolerances:** `siesta.md` § 3 row 8 documents two
  significant figures for them. There it is a disagreement between chapters
  (§ 5.4 against § 3), not a bug.
- **Certainty:** read and traced (format), and rendered. What TranSIESTA then does
  with the written 0 is not traced.

### D3. PySCF's identity setting `job_name` may be overridden per stage, which breaks the warm chain between rungs

- **Rule:**
  - `stages.md` § 1.1a item 2: "The `JOB` literal stays unsuffixed, exactly as
    `SystemLabel` does and for the same reason: the engine finds the previous
    rung's files by that name".
  - `pyscf.md` § 2: "`job_name` stays unsuffixed so `.chk`/`_optimized.xyz`
    transfer across stages".
  - `template.md` § 5, row `shared`: the identity binds every rung "because each
    rung reads what the one before it wrote under that label".
- **What the code does:**
  - The catalogue's `job_name` carries no `shared` (`shared = ()`).
  - `template.IDENTITY_ITEMS = ("system_label", "species_order", "psml_lib")`
    (`template.py:2190`) holds SIESTA's identity only.
  - So `why_not_in` (`template.py:2748-2756`) has no refusal.
  - Measured: `accepts(job_name, "other", "override", engine="pyscf",
    kind="optimization", stage="medium")` → `None`. The same call for SIESTA's
    `system_label` is refused, "shared by every stage of this calculation: a
    calculation has one name …".
  - `settings_in("override", engine="pyscf", …)` lists `job_name`, so it is
    offered as a stage-table column.
  - The deck's JOB literal is `cfg.job_name` (`pyscf/input.py:203`). The launch
    carries warm files named by the label, `compose(label, r.suffix)`
    (`jobset/model.py:521-536`).
- **Scenario:** in a PySCF ladder, `"overrides": {"job_name": "h2b"}` on `medium`:
  - prep accepts it;
  - `medium`'s deck reads `h2b.chk` and `h2b_optimized.xyz`, while the launch
    copied `<label>.chk` and `<label>_optimized.xyz`, so it starts cold although
    `restart = continue`;
  - `tight` then finds no `<label>.chk` in `medium`'s run.
- **Certainty:** read and traced. The cold start is inferred from the deck's
  JOB-named reads; nothing was run.

### D4. "Verbose inline comments" off gives no bare deck on either engine, and on PySCF it drops the wrong lines

- **Where:** `jobset/prep.py:797` calls `_sc.prepare_deck(…)` with no `verbose`.
  `script_emit.py:1519` defaults it to `True`. The catalogue notes are gated only
  on that argument: `script_emit.py:1145-1150` and `:1181-1182`. No caller passes
  `cfg.verbose_comments`; `transport/gather.py:442` passes `verbose=True`.
- **Rule:**
  - The item's help, which the form shows (`catalogue.template.toml`,
    `verbose_comments`): "Turn it off for a bare deck -- only the comments go".
  - `siesta.md` § 3: "Verbose comments (default on). Every numeric parameter is
    preceded by a `# …` block".
- **Scenario:**
  - With `verbose_comments = false`, a SIESTA deck still carries every catalogue
    note (the 20-line MeshCutoff note, the DM.Tolerance tier ladder, …).
  - A PySCF deck keeps the same notes but loses the lines that are behind
    `if v:` (`pyscf/input.py:369-386`): the `# charge = +0 (detected …)` and
    `# spin (2S) = …` lines saying where the charge and spin came from.
- **Certainty:** read and traced. Both sub-reviewers rendered it.

### D5. The `pao_energy_shift` help tells every person a false deviation from SIESTA's default

- **Where:** `data/catalogue.template.toml`, `[item.pao_energy_shift]` help: "SIESTA's
  own default is 0.02 Ry. This catalogue starts at 0.01 because …". The same
  item's `engine_default` says `value = 0.01` (`atom.F:57 … eshift_default =
  0.01_dp`). The local SIESTA 5.4.2 manual says `PAO!EnergyShift … <0.01 Ry>`.
- **Rule:** `template.md` § 5.2: "Where this project ships a value the engine would
  not, the item's `help` states that it is a deviation, what the engine's own value
  is". § 10b: "A `help` string is a claim about the engine".
- **Scenario:** the form calls 0.01 "SIESTA's own default" (through
  `template.default_words`) and, in the same field's help, says 0.01 departs from
  SIESTA's 0.02. Every SIESTA deck's PAO.EnergyShift note repeats the false
  sentence. It is already recorded as C7 in `archive/2026-09-29-m11-static-review.md`
  and is not fixed.
- **Certainty:** read and traced.

### D6. In a charged SIESTA vibration, the Makov–Payne script in the force-constant folder prints a nudged geometry's energy as the total energy

- **Where:** `siesta/seam.py:37-42` writes the script beside every deck of a
  charged finite system, a vibration's `fc` phase included. The script reads
  `steps[-1]["energy"]` (`siesta/makov_payne.py:297-301`) and prints it as
  `E_total (raw/corrected, eV)`. In an FC run the last committed step is the last
  free atom's −z displacement (`parse/engines/siesta_reader.py`, `Begin FC step`).
- **Rule:** `siesta.md` § 4 puts the script beside "each phase folder of a
  vibration's run". § 9 says it reads "its last energy". The script itself says it
  "Reads the converged total energy".
- **Scenario:** an anion vibration (net_charge −1). Running
  `python3 makov_payne_correction.py` in the `fc` folder prints the energy at a
  displaced geometry as the molecule's corrected total energy. The ΔE_MP term is
  right; the totals are not.
- **Certainty:** read and traced (SIESTA sub-reviewer, confirmed against
  `makov_payne.py`).

### D7. The ladder-loosening warning misses a tenfold loosening of a small tolerance

- **Where:** `validation/stages.py:147-150`:
  `if abs(cur - prev) <= _REL_TOL * max(abs(prev), abs(cur), 1.0): return False`,
  with `_REL_TOL = 1e-9` at `:44`. The `1.0` floor makes the tolerance an absolute
  1e-9 for every value below 1.
- **Rule:** `stages.md` § 4 R3: "A ladder that *loosens* — stage 2 coarser than stage
  1 — … So a description is also read across its stages", on every item with
  `tightens`.
- **Measured:**
  - `check_ladder_does_not_loosen([("tight", scf_conv_tol=1e-10), ("extra", 1e-9)])`
    → `[]`, no warning;
  - 1e-10 → 1e-8 is warned.
- **Scenario:** a PySCF ladder whose tight rung ships `scf_conv_tol = 1e-10`, then a
  person's added stage at 1e-9: no "loosens" warning. The same happens for
  `geom_etol`.
- **Certainty:** read and traced, and measured.

### D8. The SIESTA troubleshooting tips contradict the deck's own SCF.Mixer.History

- **Where:** `siesta/input.py:1403`, "increase SCF.Mixer.History to 5-8", and
  `:1420`, "raise SCF.Mixer.History to 6". The shipped value is 8
  (`config/siesta.py`; catalogue default 8). The item's own note says raise it to
  12-20, and calls 2-6 too low.
- **Rule:** `template.md` § 2.1a: "Two homes for one fact drift … a class copy of a
  catalogue fact is a defect".
- **Scenario:** a person whose SCF fails follows the deck's tip and *lowers* the
  history from 8.
- **Certainty:** read and traced.

### D9. A PySCF deck's header can show `eps=?` for a solvent the body applies

- **Where:** `pyscf/input.py:269-271`, `_SOLVENTS.get(cfg.solvent, '?')` on the raw
  value. `scf_setup.emit_solvent_lines` (`:176`) and the gate lower-case it.
- **Rule:** `pyscf.md` § 3: the header states the run's choices.
- **Scenario:** Solvent typed `Water` gives the header
  `Solvent : Water (IEF-PCM, eps=?)` while the body writes
  `mf.with_solvent.eps = 78.3553`.
- **Certainty:** read and traced, and rendered.

### D10. An unconverged PySCF geometry is saved labelled "Optimized geometry (PySCF)"

- **Where:** `pyscf/relax_policy.py:165` (wall stop) and `:177` (`proceed`) return
  `(mol, False)`. `pyscf/input.py:691-693` then always runs
  `emit_save_call("mol_eq", PYSCF_OPTIMIZED, …)` with the words
  `"Optimized geometry (PySCF)"` (`:1589-1591`). This overwrites the per-step label
  "Geometry reached (PySCF, relaxing)".
- **Rule:** `pyscf.md` § 3 / § 3.9: the reached geometry is kept as each step's is,
  and "the converged geometry is written over it at the end".
- **Scenario:** a rung stopped at its wall, or a `proceed` rung short of
  geomeTRIC's criteria, leaves `<job>_optimized.xyz` titled "Optimized geometry"
  while its own record says not relaxed.
- **Certainty:** read and traced.

### D11. A PySCF deck with density fitting off or dispersion `none` says nothing about either

- **Where:** `pyscf/layout.py:183-197`: `if not value: return None` for
  `density_fit`, and `if value == "none": return None` for `dispersion`. Nothing is
  written in their place (`script_emit._render_sections`, `:1141-1143`).
- **Rule:** `template.md` § 6.6 obligation 4: "Every item of a rung's layout is
  written into that rung's deck with the value that will apply … No value reaches
  the engine by omission."
- **Scenario:** with `dispersion = "none"` or `density_fit = false`, the theory
  section is silent about both. The values do appear in the deck's `_MB_PARAMS`
  record rows.
- **Certainty:** read and traced.

### D12. PySCF's stability analysis can print "still internally unstable" for a state it never tested

- **Where:** `pyscf/input.py:989-1021`. If all three restarts lower the energy, the
  `for … else` branch sets `_MB_STABLE = False` and prints the WARNING without a
  fourth `stability()` call.
- **Rule:** `pyscf.md` § 7.4: the WARNING row means "checked; **not** repaired".
- **Scenario:** three successive genuine repairs produce a warning about the third
  repaired state, which was never checked.
- **Certainty:** read and traced; rare.

### D13. A refusal tells a PySCF vibration to set `restart` on a relaxation stage it cannot have

- **Where:** `template.py:2308-2326` (`why_unread`), reached through
  `why_not_in(..., "override")` at `:2757-2761`.
  - `restart` declares `stages = ["relaxation"]`.
  - A PySCF vibration's stages play only `force_constants`
    (`kind_table.toml`, `[vibration.on.pyscf] stage_roles`).
  - The clause still names the relaxation stage and ends "set it there".
- **Rule:** `template.md` § 6.4: the refusal names "the rungs that read" the item,
  with the way out.
- **Measured:** `accepts(restart, "clean", "override", engine="pyscf",
  kind="vibration", stage="freq")` → ", which only the relaxation stage of a
  vibration reads, not 'freq' (its force_constants stage) -- set it there". A
  PySCF vibration has no stage of that role.
- **Certainty:** read and traced; the message is untrue advice, nothing is run.

---

## Untrue sentences

### U1. `template.md` § 6.4, lines 1632-1636

- **Says:** "the transport kind's electronic contract — … electronic temperature,
  **the SCF mixer and its tolerances**, spin treatment and count — must be **one**
  value across the five rungs."
- **The code:** `mixing_weight`, `pulay_history`, `dm_tolerance`, `h_tolerance`,
  `dm_energy_tolerance` and `scf_energy_converge` carry `citation = ["transport"]`
  but no `shared`. Measured: `accepts(mixing_weight, …, "override", kind="transport",
  stage="device")` → `None`, while `basis_size` is refused as shared. § 5's own
  `citation` row agrees with the code ("each SCF stage's starting value, not
  shared").
- Read and traced.

### U2. `template.md` § 6.3, line 1405

- **Says:** "**Four items are merged**: `use_gpu`, `net_charge`, `spin_treatment`
  and `unpaired_electrons`."
- **The catalogue** also holds these as one row for both engines: `restart`,
  `wall_slack`, `gpu_count`, `max_memory_mb`, `verbose_comments`,
  `write_molwatch_log`, `temperature_K`, `already_relaxed`, `compute_raman`,
  `compute_ir`, `raman_modes`, `raman_freq_min_cm1`, `raman_freq_max_cm1`. Several
  of them are in the table just above the sentence.
- Read and traced.

### U3. `template.md` § 6.4, lines 1780-1782

- **Says:** "a kind with roles that carries it must find one of its own there —
  both refused when the catalogue is read".
- **The code:** `Item.__post_init__` (`template.py:767-772`) checks only the kinds
  named in the item's `calculations`. An item with no `calculations` (carried by
  every kind) whose `stages` names none of a role-bearing kind's roles is not
  refused. No current catalogue row is in that state.
- Read and traced.

### U4. `template.md` § 6.6, line 1949

- **Says:** quotes `resolve`'s layer words as "… this stage's override · **the
  rung** · …".
- **The code:** the word is `SAID["role"] = "the stage's role"` (`template.py:2650`),
  as § 5.4's table says.
- Read and traced; wording only.

### U5. `siesta.md` § 1.1, line 133, row 3

- **Says**, in the present tense: "Only `kind` in `{engine, deck}` reaches the deck
  writer; `wrapper` and `produce` items belong to other layers".
- **The code:** the writer reads wrapper and produce items off the config:
  - `siesta/input.py:924-933`, `mpirun -np {cfg.mpi_np}`;
  - `:1011-1016`, `cfg.omp_threads` and `cfg.max_memory_mb`;
  - `:322-323`, `mpi_np` in BENCH-MARKS;
  - `cfg.species_order` and `cfg.verbose_comments` throughout.
  Nothing filters by kind.
- Read and traced.

### U6. `pyscf.md` § 1, line 75

- **Says:** "The public surface is `spec_for` and `PySCFConfig`, which
  `molbuilder.pyscf` exports."
- **The code:** `pyscf/__init__.py` is a docstring: "It re-exports nothing".
- Read and traced.

### U7. `pyscf.md` § 2, lines 111 and 136

- **Says:** "A successful run produces exactly these" / "The script's header
  `Outputs:` block lists **exactly** this set".
- **The code:** the table leaves out `<job>.constraints.txt`
  (`runfiles.ROLE_CONSTRAINTS`). A relaxation with held atoms writes it
  (`pyscf/input.py:646-658`) and lists it in Outputs.
- Read and traced.

### U8. `pyscf.md` § 2a, line 185

- **Says:** "All three failure paths … **exit**, with a message naming the cause
  and the two ways out (run where the GPU is, or set `use_gpu` to false on its run
  card)."
- **The code:** the failed-promotion path (`runtime_info.py:304-311`) names no way
  out, only the cause and "There is no CPU fallback".
- Read and traced.

### U9. `pyscf.md` § 7a, line 853

- **Says:** "`_mb_configure_theory(mf)` … It is emitted on **every** deck and every
  construction calls it".
- **The code:** only the vibration deck emits it (`pyscf/vibration_emitters.py:632`,
  `:1009`). The optimization deck applies `THEORY_SECTION` inline and has no such
  function.
- Read and traced.

### U10. `pyscf.md` § 3, line 406

- **Says:** "The deck writes each value beside its source."
- **The code:** only while `verbose_comments` is on (`pyscf/input.py:369-386`; see
  D4).
- Read and traced.

---

## Checked and found true (for the record, not findings)

- **`template.md`, catalogue facts.** § 4.0a: 78 of 89 SIESTA help strings break
  lines. § 4.2 and § 6.0: the example items are as the catalogue states them. § 5.3:
  the `above` / `at_least` table. § 5: the `citation` (16 rows), `role`, `shared`,
  `why_fixed`, `per_point_from`, `limit_applied_by` and `launch` rows. "eighteen
  rows" on `stages = "scf"`. The five `allocation` items. `read_by` only on
  `use_gpu`. `required` declared by none.
- **`template.md`, the template code.** The reader's refusals (schema major,
  `written_by`, unknown `kind`/`category`/`read_by`, conditional keys). The retired
  and renamed sentences. The 2.1.0 zero rule at prep. `template_with_values` writes
  no run setting and no catalogue-only key. `config_from_template` judges by
  today's catalogue type.
- **`template.md`, the run-time path.** The precedence in § 10a.2 is `resolve`'s
  order. `_check_fits` refuses and never clamps. Role answers are laid last.
  `template_file.of` refuses a run setting naming `jobset migrate`.
- **`stages.md`.** The § 1.3 group counts for both engines (2/20/35/3/17/12 and
  1/32/12/1/10/6). The five engine-less items. The PySCF stage items and their
  `engine_key`s. Stage-name rule and case-folded duplicates. `shape` refusal.
  `check_identical_stages`. `engine_ladder` / `fresh_stages` / strategies.
- **`tuning.md`.** Every preset value in § 2.2–2.5, § 2.10 and § 4 against
  `SIESTA_STAGE_PRESETS` / `PYSCF_STAGE_PRESETS`. The shipped defaults in § 2.5,
  § 2.6, § 2.8 and § 4. The vibration recommendations equal the tight rung.
  `bench/result.py`'s asked-versus-ran comparison (block size recorded, not
  compared). The `gpu_count` refusals in `prep_inputs`.
- **`overview.md` § 2 and § 3a.** The block emitters exist as named. `use_gpu` is
  `deck`, `execution`, default off, expands `Diag.ELPA.GPU`/`gpu4pyscf`, in
  `staging`.
- **`configuration.md`.** `config_dir` resolution (§ 2.1c, empty value not set, no
  append to the override; § 2.1d runtime fallback; § 3.1 resolvers).
  `persist.write_bytes` `mode` / `exclusive` (§ 2.3). The only TOML writer is
  `template.py`'s (M-5).
- **`optimization-postprocessing.md` § 6.** The export names
  `<label>.optimization.nc` and `<label>.optimization-csv`.
- **SIESTA sub-reviewer.** The `template.md` claims I passed it hold: the not-set
  line, the PROVENANCE rows, BENCH-MARKS, `unit_word`, the role note, the
  relax-type refusal, `write_md_xmol`, ENGINE-OFFSET, the VIBRATION block.
- **PySCF sub-reviewer.** The RUN-SETTINGS block and its five names.
  `MAX_MEMORY_MB = None` reaching `gto.M`. `limit_applied_by` pyscf = deck. The
  not-set lines. Restart gating. No BENCH-MARKS. The merged items' spellings.
