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
| `mixing_weight` | `SCF.Mixer.Weight` | profile | the classic SCF knob |
| `pulay_history` | `SCF.Mixer.History` | profile | SIESTA's `diis_space` |
| `max_scf_iter` | `MaxSCFIterations` | budget | patience, not the criterion |
| `solution_method` | `SolutionMethod` | profile | ⚠ **M3** — see § 2 |
| `restart` | *(expands to `DM.UseSaveDM`)* | stage | ⚠ **M4** — see § 2 |

### 5 · `procedure` — what does the run carry out, and what does it leave behind? (16)

| field | keyword | wf | note |
|---|---|---|---|
| `relax_type` | `MD.TypeOfRun` | stage | decides **what runs** — CG / Verlet / MD |
| `relax_steps` | `MD.Steps` | budget | |
| `relax_max_displ` | `MD.MaxDispl` | stage | ⚠ **M5** — see § 2 |
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
| `block_size` | `BlockSize` | **unset** | `block_size` |
| `parallel_over_k` | `Diag.ParallelOverK` | set | — |
| `enable_gpu` | `Diag.ELPA.GPU` | set | — |
| `continue_retries` | *(baked into the wrapper)* | set | — |
| `mpi_np` | *(`mpirun -np`)* | **unset** | `rank_count` |
| `omp_threads` | *(`OMP_NUM_THREADS`)* | **unset** | `omp_threads` |
| `max_memory_mb` | *(`ulimit -v`)* | **unset** | `node_memory` |

---

## 2. The six hard placements — dissolved by multi-tagging

**User decision, 2026-08-13: `category` is a list, and it does not affect the
script** (`template.md` § 6.2). The first entry is the panel; the rest make the
item findable where a user would look. That retires all six agonised calls — each
was a symptom of forcing one answer where the parameter genuinely has two.

| was | now | why the second tag |
|---|---|---|
| **M1** `pao_energy_shift` | `["method", "accuracy"]` | reported beside the basis; tightens like an accuracy knob |
| **M2** `electronic_temperature` | `["accuracy", "system"]` | smearing for a molecule; a real temperature for a finite-T run |
| **M3** `solution_method` | `["method", "convergence"]` | **the one I called wrong.** `OrderN` is approximate, so it panels under `method`; a user hunting a stubborn SCF still finds it under `convergence` |
| **M4** `restart` | `["convergence", "execution"]` | an initial-guess choice that reads as run plumbing |
| **M5** `relax_max_displ` | `["procedure", "convergence"]` | a geometry convergence aid, shown with its siblings |
| **M6** `system_label` | `["procedure", "system"]` | names the system; prefixes every output file |

**M3 was the load-bearing one and it is now simply correct.** I had placed it in
`convergence` while arguing it belonged in `method` — the multi-tag gives both,
and the earlier note that it *"argues for a cross-reference rather than a second
category"* is exactly this mechanism.

**The nuance moves to `help`, where a user reads it.** *"`OrderN` scales linearly
but is approximate — it changes the answer, not just the route"* is a sentence.
It was never a taxonomy problem.

**One placement is NOT a free choice.** `execution` is the benchmark's sweepable
set, so it is a claim that the knob changes speed and not the answer. The eight
items in § 1.6 each meet that test; `mesh_cutoff` deliberately does not, though it
also costs time.

---

## 3. What the placement exposes

**`procedure` holds 16 of 44 — over a third, and the largest by far.** It spans
three questions:

1. *what run is this?* — `relax_type`, `relax_steps`, the three `md_*`
2. *what files do I want?* — the six `write_*`, `copy_psml`
3. *how should results be presented?* — `wrap_into_cell`, `verbose_comments`

**Decided 2026-08-13 (user): no split — RENAMED.** All three are *the job in
general*, and the group is engine-specific bookkeeping rather than science, so a
seventh entry in a closed vocabulary would buy nothing a heading inside one panel
cannot. The name changed from `outputs` (which described only groups 2 and 3) to
`procedure` — what the run carries out and what it leaves behind.

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
