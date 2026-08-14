# The SIESTA catalogue — all 44 fields in six categories (T1)

**Role:** plan
**Domain:** execution
**Companions:** [`execution/template-unification-plan.md`](?doc=execution/template-unification-plan.md)
(the plan this unit serves) · [`engines/template.md`](?doc=engines/template.md)
§ 6.2 (the category vocabulary — **the authority**) ·
[`engines/siesta.md`](?doc=engines/siesta.md).

> **This is a working document for review, not a contract.** It records where
> every `SiestaConfig` field lands and — more usefully — **which placements were
> hard and why**. Once the placements are agreed it becomes the input to T2, and
> the arguments here move into the contract or are dropped.

---

## 1. The placement

44 fields. `wf` is the existing `workflow_group`; it is orthogonal to `category`
and is shown to make clear that the two axes disagree, which is the point of
having both.

### 1 · `system` — what am I calculating? (4)

| field | keyword | wf | note |
|---|---|---|---|
| `net_charge` | `NetCharge` | profile | |
| `spin_polarized` | `SpinPolarized` | profile | |
| `spin_total` | `Spin.Fix` + `Spin.Total` | profile | one item, two keywords (`expands`) |
| `species_order` | *(ChemicalSpeciesLabel ordering)* | — | section-less today; membership is total, so it is an item |

### 2 · `method` — at what level of theory? (5)

| field | keyword | wf | note |
|---|---|---|---|
| `basis_size` | `PAO.BasisSize` | stage | **Q1** — the dominant accuracy knob, placed where it is *reported* |
| `pao_energy_shift` | `PAO.EnergyShift` | stage | ⚠ **M1** — see § 2 |
| `xc_functional` | `XC.functional` | profile | |
| `xc_authors` | `XC.authors` | profile | pairs with the above; two keywords, one decision |
| `psml_lib` | *(stages `.psml` files)* | profile | the pseudopotentials ARE part of the theory |

### 3 · `accuracy` — how precisely are the equations solved? (6)

| field | keyword | wf | note |
|---|---|---|---|
| `mesh_cutoff` | `MeshCutoff` | stage | |
| `kgrid` | `%block kgrid_Monkhorst_Pack` | stage | |
| `dm_tolerance` | `DM.Tolerance` | stage | |
| `dm_energy_tolerance` | `DM.Energy.Tolerance` | stage | |
| `relax_force_tol` | `MD.MaxForceTol` | stage | the geometry criterion — what answer you accept |
| `electronic_temperature` | `ElectronicTemperature` | profile | ⚠ **Q2 / M2** — see § 2 |

### 4 · `convergence` — how do I reach it when it fights? (5)

| field | keyword | wf | note |
|---|---|---|---|
| `mixing_weight` | `DM.MixingWeight` | profile | the classic SCF knob |
| `pulay_history` | `DM.NumberPulay` | profile | SIESTA's `diis_space` |
| `max_scf_iter` | `MaxSCFIterations` | budget | patience, not the criterion |
| `solution_method` | `SolutionMethod` | profile | ⚠ **M3** — see § 2 |
| `restart` | *(expands to `DM.UseSaveDM`)* | stage | ⚠ **M4** — see § 2 |

### 5 · `outputs` — what do I want produced? (16)

| field | keyword | wf | note |
|---|---|---|---|
| `relax_type` | `MD.TypeOfRun` | stage | decides **what runs** — CG / Verlet / MD |
| `relax_steps` | `MD.NumCGsteps` | budget | |
| `relax_max_displ` | `MD.MaxCGDispl` | stage | ⚠ **M5** — see § 2 |
| `md_initial_temperature` | `MD.InitialTemperature` | profile | |
| `md_target_temperature` | `MD.TargetTemperature` | profile | |
| `md_length_timestep` | `MD.LengthTimeStep` | profile | |
| `system_label` | `SystemLabel` | profile | ⚠ **M6** — see § 2 |
| `wrap_into_cell` | *(pre-emission positioning)* | profile | |
| `verbose_comments` | *(comment-block control)* | profile | |
| `write_forces` | `WriteForces` | — | |
| `write_coor_step` | `WriteCoorStep` | — | |
| `write_coor_xmol` | `WriteCoorXmol` | profile | |
| `write_md_history` | `WriteMDhistory` | profile | |
| `write_hs` | `SaveHS` | profile | |
| `write_molwatch_log` | *(writes `.molwatch.log`)* | — | |
| `copy_psml` | *(triggers `.psml` staging)* | — | |

### 6 · `execution` — how does it run on this machine? (8)

**This is the sweepable set** (§ 6.2). Three carry `allocation: True` and are
therefore **valueless with a resolver** (§ 6.4).

| field | keyword | value | resolver |
|---|---|---|---|
| `diag_algorithm` | `Diag.Algorithm` | set | — (`read_by = ["wrapper"]`, § 6.1) |
| `parallel_block_size` | `BlockSize` | **unset** | `block_size` |
| `parallel_over_k` | `Diag.ParallelOverK` | set | — |
| `enable_gpu` | `Diag.ELPA.GPU` | set | — |
| `continue_retries` | *(baked into the wrapper)* | set | — |
| `mpi_np` | *(`mpirun -np`)* | **unset** | `rank_count` |
| `omp_threads` | *(`OMP_NUM_THREADS`)* | **unset** | `omp_threads` |
| `max_memory_mb` | *(`ulimit -v`)* | **unset** | `node_memory` |

---

## 2. The six hard placements

Each is a real judgement call. My position, then the case against it.

**M1 · `pao_energy_shift` → `method` (not `accuracy`).** It sets the orbital
confinement radii, so it is genuinely a basis-quality knob and belongs beside
`mesh_cutoff` by function. Placed with `basis_size` because a user changes them
together and reports them together — *"DZP with a 50 meV energy shift"* is one
sentence. **Against:** it is `workflow_group = "stage"`, i.e. it tightens down the
ladder, which is the signature of an accuracy knob. Weakest of my six.

**M2 · `electronic_temperature` → `accuracy` (Q2).** For a molecule it is
smearing that aids convergence. **Against:** it changes occupations, so it changes
the answer — and for a genuinely finite-temperature calculation it is a `system`
fact, not a numerical one. It may need to be context-dependent, which no other
item is; if so, `accuracy` is the safer default because it warns rather than
reassures.

**M3 · `solution_method` → `convergence` (not `method`).** `diagon` vs `OMM` vs
`OrderN` is how you reach the solution. **Against:** `OrderN` is *approximate* —
it changes the answer, not just the route — so by the § 6.2 test it is a `method`
choice. **I think this one is genuinely wrong as placed** and would move it to
`method`, except that then a user hunting a stubborn SCF will not find it. It
argues for a `read_by`-style cross-reference rather than a second category.

**M4 · `restart` → `convergence`.** It expands to `DM.UseSaveDM`: reuse the
previous density matrix, which is an *initial-guess* choice — PySCF's
`scf_init_guess = "chkfile"` is the same idea and sits in `convergence`.
**Against:** it reads as run plumbing, so `execution` is the intuitive guess. The
cross-engine parallel decided it.

**M5 · `relax_max_displ` → `outputs`.** It caps the optimizer's step. **Against:**
it is a convergence aid for a *geometry* rather than an SCF, so `convergence` is
defensible. Placed with the other `relax_*` items so the relaxation reads as one
panel; splitting one knob off would be worse than a slightly loose category.

**M6 · `system_label` → `outputs`.** `SystemLabel` prefixes every output file, so
in practice it is the naming of results. **Against:** it names the *system*, which
is what category 1 is called. If we want a `identity` concept it would live with
`job_name`, which is where the same question will arise for PySCF.

---

## 3. What the placement exposes

**`outputs` holds 16 of 44 — over a third, and the largest by far.** That is a
signal, not a coincidence: it is currently absorbing three different questions.

1. *what run is this?* — `relax_type`, `relax_steps`, the three `md_*`
2. *what files do I want?* — the six `write_*`, `copy_psml`
3. *how should results be presented?* — `wrap_into_cell`, `verbose_comments`

**This is the strongest argument for Q3's seventh category.** If `task`
(what run) splits out, `outputs` falls to ~10 and each panel answers one question.
Deferred in the plan; the count is the evidence for reopening it.

**`workflow_group` and `category` genuinely disagree**, which confirms both axes
are needed: `execution` spans `budget` (`mpi_np`) and unset (`diag_algorithm`);
`accuracy` spans `stage` and `profile`. Neither is derivable from the other.

**Five items are section-less today** (`write_forces`, `write_coor_step`,
`species_order`, `write_molwatch_log`, `copy_psml`) and all get a category —
which is § 7's total membership doing its job, and a small demonstration that
`category` covers ground `section` did not.

---

## 4. What T2 needs from this

1. Agreement on the six placements in § 2 — or corrections.
2. A decision on **Q3** (a `task` category), which § 3 now has evidence for.
3. Then `declaration_for` emits `category`, `engines` and `resolver`;
   `read_template` requires `category`; `section` retires.
