# Vibration — the calculation, on PySCF and on SIESTA

**Role:** contract — **the master document for the `vibration` calculation
kind.** What the calculation is, how each engine computes it, what the script
and the deck contain, what the result file holds, and which rules the code is
checked against. Where this document and the code disagree, the code is wrong
or this document is out of date — and the second is fixed first.
**Domain:** engines — with its science half in
[`science/normal-modes.md`](?doc=science/normal-modes.md) and its web half in
[`web/spectra.md`](?doc=web/spectra.md).
**Companions:** [`science/normal-modes.md`](?doc=science/normal-modes.md) (the
derivations, the rules R1–R8, the acceptance test);
[`web/spectra.md`](?doc=web/spectra.md) (the Spectrum tab and the Results-tab
viewer), [`web/spectrumchart.md`](?doc=web/spectrumchart.md) and
[`web/vibrationview.md`](?doc=web/vibrationview.md) (the chart and the
animation); [`engines/pyscf.md`](?doc=engines/pyscf.md) and
[`engines/siesta.md`](?doc=engines/siesta.md) (the two emitters this kind
rides); [`engines/stages.md`](?doc=engines/stages.md) (the description on
disk); [`engines/template.md`](?doc=engines/template.md) § 6.3 (how the
catalogue is narrowed to a kind);
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a and § 6.1
(warm files, the artifact registry);
[`model/overview.md`](?doc=model/overview.md) § 2.2 (what a reorder may do);
[`science/validation.md`](?doc=science/validation.md) (the gate the kind's
checks run in). **Open items:** row **V1** of
[`plans/plan.md`](?doc=plans/plan.md), and nowhere else.

> **Consolidated 2026-09-24** from the design of 2026-09-21, the follow-up
> audit of 2026-09-22, the UI walk of 2026-09-23 and the implementation
> sections that had accumulated in the science and web contracts. The design
> and the audit are archived (`archive/2026-09-24-normal-mode-unification-design.md`,
> `archive/2026-09-24-vibration-audit.md`) as the record of how the decisions
> were reached; every decision, measurement and open item they held is here
> or in the plan. Nothing decided there is re-decided here. The structure-API
> audit (`plans/2026-09-22-unification-audit.md`, still live) holds the UI walk
> of 2026-09-23 whose spectrum rows are registered under V1. One ordering rule
> of the design was knowingly broken: its step 4 (the API shape) was to land
> before the SIESTA arm, and part of it landed after — the cost is the second
> pass V1.0 and V1.6 name.

---

## 0. How to read this document

Three documents share this subject, and each owns one kind of statement:

| document | owns | read it when you want to know |
|---|---|---|
| [`science/normal-modes.md`](?doc=science/normal-modes.md) | **why** — the derivations, the rank rule for the motions that are not vibrations, stationarity, masses, the rules R1–R8 and the acceptance test | why a held-atom run reports fewer modes than `3·N_free`, and why that number is a rank and not a table; and, in its § 4b, both engines' workflows step by step against the discussion this work started from — what the tool does at each step, where it differs, and what is owed |
| **this document** | **what and how** — what the calculation computes on each engine, what the script and the deck contain, what the result file holds, what the code must keep true, what is built and what is owed | how a run works end to end, where a number in the file comes from, what to check the code against |
| [`web/spectra.md`](?doc=web/spectra.md) | **the tab** — the Spectrum tab, the hand-over to Task setup, the viewer, the chart, the animation, and how an absent number is drawn | what a person sees and clicks |

**A student** reads § 1 (the calculation in one page, with the eight rules
in one line each), § 2 (the road), § 4 and § 5 (the two routes, with
pseudocode and pictures), then § 11 (worked examples), with § 12 (the
glossary) open beside them, and takes the science document for the
derivations. **A developer** reads § 2, § 3 (the parameters), § 6 (the result
file), § 7 (the invariants), § 8 (the pieces), § 9 (validation and tests) and
§ 10 (shipped and owed), and checks the code against § 7. Words used before
they are defined — *kind*, *description*, *the pair*, *deck*, *phase*,
*rung*, *route* — are § 12's.

---

## 1. The calculation in one page

### 1.1 What a vibration calculation is

Near a geometry `R₀`, write every atom's displacement as one long vector `u`
(three numbers per atom). The energy is a quadratic bowl and the force is minus
its gradient — the many-coordinate form of Hooke's law:

```text
    E(u)  ≈  E₀  +  ½ uᵀ H u ,     H_{Iα,Jβ} = ∂²E / ∂R_{Iα} ∂R_{Jβ}      (the Hessian)
    F  =  −∇E  =  −H u             moving one atom pushes on the others: that
                                   coupling IS the off-diagonal Hessian
```

Divide each entry by the square roots of the two atoms' masses and diagonalise:
each eigenvector is a **normal mode** — a pattern of atoms moving together at
one frequency `ω = √λ` — and the spectrum is the list of them. Six of the
eigenvectors of a free molecule are not vibrations (three slides, three turns;
five for a straight molecule), so they are removed before the diagonalisation.
Two more quantities ride on the modes: how strongly each absorbs or scatters
light (the **infrared** and **Raman** strengths, from how the dipole and the
polarizability change along the mode), and the harmonic **thermochemistry**
(zero-point energy, entropy, free energy) summed over them.

### 1.2 What holding atoms means

Hold some atoms — an anchor, a metal slab — and split the coordinates into free
`A` and held `F`. Impose `u_F = 0`:

```text
    ┌ F_A ┐       ┌ H_AA  H_AF ┐ ┌ u_A ┐           F_A = −H_AA u_A
    │     │  = −  │            │ │     │     ⇒
    └ F_F ┘       └ H_FA  H_FF ┘ └  0  ┘           F_F = −H_FA u_A   ≠ 0
```

Four consequences, each a rule in the science document:

1. **What is diagonalised is `H_AA`** — the free–free block of the Hessian of
   the *whole* system, with every held atom present in the energy. A held atom's
   displacement is zero; its interaction is not. This is the partial Hessian
   (PHVA) of Head, Li & Jensen and Besley [Head1997, LiJensen2002, Besley2008].
2. **Some eigenvectors of `H_AA` are still not vibrations.** The free atoms can
   turn about one held atom, or about the line through two, at no cost. How
   many such motions survive is a **rank** computed from the geometry
   ([`science/normal-modes.md`](?doc=science/normal-modes.md) § 3.1 — six, five,
   three, one or zero, with two traps a table gets wrong), and they are
   removed **before** diagonalising (R3). With three held atoms not on one line,
   or any held atom in a periodic cell, nothing survives and nothing is removed.
3. **Stationarity is asked of the free atoms only** (R5): `∇_A E = 0`. The
   held atoms carry the constraint force `−H_FA u_A` by definition.
4. **One mass convention on every route** (§ 3.2 there): isotope-averaged
   masses (H = 1.008, not 1 — whole mass numbers move a stretch by 15 cm⁻¹),
   modes normalised `Σ_k m_k |L_k|² = 1` in amu. Crossing the normalisations
   once put every infrared intensity out by 1823×.

**The eight rules of the science contract, one line each** — cited by number
throughout this document; the full statements and their reasons are
[`science/normal-modes.md`](?doc=science/normal-modes.md) § 6:

| rule | in one line |
|---|---|
| **R1** | the count of surviving whole-body motions is computed in one place, by a rank — never tabulated or branched on |
| **R2** | one formula for the mode count, `3·N_free − n_rigid`, for every system; `3N − 6` is the empty-held-set case |
| **R3** | project the surviving motions out, then diagonalise — on both engines, at the Γ point only |
| **R4** | what is reported is what is left: every mode in the file is a vibration, and no reader needs a filter |
| **R5** | stationarity is judged on the free atoms; held atoms carry the constraint force by definition |
| **R6** | a prediction that cannot match the run is not written: a stated count is R2's or the run's own list |
| **R7** | the person is told how many motions survive the hold and what they are, before paying for the run |
| **R8** | a cost claim is read from the code, and what the code cannot skip is stated beside it |

### 1.3 The two engines, and the one path after them

| | **PySCF** | **SIESTA** |
|---|---|---|
| suits | isolated molecules, small clusters | periodic slabs, surfaces, junctions |
| basis | Gaussian functions | numerical atomic orbitals |
| how it gets `H_AA` | **analytic** second derivatives, for the free atoms (§ 4.4) | **central differences** of its analytic forces: nudge each free atom, read every force (§ 5) |
| frequencies + mode shapes | yes | yes |
| infrared / Raman strengths | yes (§ 4.6) | **not offered** (§ 5.6) |
| thermochemistry | full RRHO (rigid-rotor harmonic-oscillator) for a free molecule; vibrational sums with atoms held | vibrational sums |
| per-mode electronic structure | yes (§ 4.8) | no |
| a gold electrode | must be faked as a finite cluster | its natural home — and the same pseudopotentials, orbitals and k-points as the transport step (§ 5.6) |
| where a typical run lands | `spectrum/` | `frequency/` — a storage vocabulary, not an engine rule (§ 2.4) |

Only *how the block is obtained* differs. Everything after it is one function,
`spectra/normal_modes.py::vibrational_modes`, which both engines hand the same
things — the block, the masses, the geometry with its held set, and which
axes repeat (`axis_kind`, with the cell when one does) — and which returns the
modes with the surviving whole-body motions removed. The
PySCF deck carries that function's source inside the generated script; the
SIESTA read-back calls it on the host. There is no second path (R1–R4).

### 1.4 The smallest example, with real numbers

H₂ from the measured SIESTA fixture (`tests/fixtures/siesta_fc`): the bond's
force constant read from the `.FC` file is `k = 41.713 eV/Å²`, `m = 1.008 amu`.

| | diagonalised | motions removed | modes | ω |
|---|---|---|---|---|
| both atoms free | the 6×6 block | 5 (three slides, two turns; the turn about the bond moves nothing) | 1 | `√(2k/m)` = **4744 cm⁻¹** |
| one atom held | the free atom's 3×3 block, `[k]` along the bond | 2 (the free atom swinging about the held one) | 1 | `√(k/m)` = **3355 cm⁻¹** |

*(The 2026-09-23 run of § 5.5, on the same unrelaxed bond, reported 3358 cm⁻¹:
the same mode from its own SCF and its own force constant, the fixture's from
a hand-set deck — one quantity, two SCF settings. On the relaxed bond the road
reports 3022, § 9.)* Holding one end lowers the stretch by exactly `√2`, because
the partner no longer recoils: the reduced mass is `m` instead of `m/2`. Both numbers are
right for the question each asks — a constrained Hessian is a different
question, not a worse answer. The two swings of the free atom are the
surviving motions of point 2 above; without R3 they would be reported as two
modes near zero. [`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 4b works the same example from the eigenvectors up.

---

## 2. The road — from a description to a result file

### 2.1 The whole road

```mermaid
flowchart LR
  M["Molbuilder tab<br/>build the structure,<br/>hold atoms in the viewer,<br/>Save to project (the pair)"] --> S["Spectrum tab<br/>load the structure, pick the engine on the strip<br/>(a periodic structure defaults to SIESTA),<br/>set parameters (the catalogue's form per engine),<br/>read the live checks"]
  S -->|"Send to Task setup<br/>= the hand-over"| T["Task setup<br/>shape, machine, the kind's ladder;<br/>Save writes task.json;<br/>the stage's tab prints the commands<br/>(--target when the CLI would refuse to guess)"]
  T -->|"prep run freq"| P["the deck<br/>PySCF: &lt;label&gt;_01_freq.py<br/>SIESTA: &lt;label&gt;_01_freq.fdf (from a sorted copy)"]
  P -->|"launch run freq"| R["the run<br/>PySCF: writes &lt;label&gt;.spectra.json itself<br/>SIESTA: leaves &lt;label&gt;.FC"]
  R -->|"SIESTA only:<br/>summarize run freq"| A["&lt;label&gt;.spectra.json<br/>the one artifact, schema 6"]
  A --> V["Results tab<br/>chart · modes table · animation ·<br/>electronic structure · thermochemistry"]
```

The CLI walks the same road without the browser:

```bash
molbuilder jobset init --structure P/structure/x.xyz --bundle P/frequency/F \
    --engine pyscf|siesta --calculation vibration --name X --shape hierarchical
molbuilder jobset prep run freq --bundle P/frequency/F --target this
molbuilder jobset launch run freq --bundle P/frequency/F --mode direct --yes
molbuilder jobset summarize run freq --bundle P/frequency/F      # SIESTA only
```

`init` refuses `--stage-strategy` for this kind (a *ladder* is a
description's list of stages, each a *rung* with its own parameter set; a
*tier* ladder — coarse, medium, tight — grades an optimisation's convergence,
and a vibration has one stage) and any engine but the two named.

### 2.2 The description: the relaxation is the person's explicit choice

A vibration is an ordinary described job: `task.json` carries
`calculation: "vibration"`, the engine, and its ladder
(`pyscf/stages.py::vibration_stages`; [`engines/stages.md`](?doc=engines/stages.md)).
The parameters travel in `<label>.template.toml`, the catalogue narrowed to
the kind (§ 3) — **including the relaxation's convergence settings on both
engines**, shown and editable exactly as on the Structure-optimization tab,
with the kind's own recommended values, which are tight (§ 3.1).

**A harmonic analysis is only valid at a stationary point**, so the
relaxation is the measurement's precondition, and **whether it runs is the
person's explicit say** *(user rulings 2026-08-20, refined 2026-09-24: "this
is totally user's control and it is explicit")*, made with one box on both
engines, `already_relaxed`:

| the box | what the tool does, on either engine |
|---|---|
| **unticked** (the default) | the calculation **relaxes first**, at the template's convergence settings — recommended tight, a largest remaining force of about 0.01 eV/Å — and then takes the second derivatives at the relaxed geometry. On PySCF that is Phase 0 of the one script (§ 4.2); on SIESTA it is a **`relax` stage before the `freq` stage** (§ 5.2a), the relaxed geometry carried into the force-constant deck as coordinates |
| **ticked** | the person states the structure is already relaxed **at this level of theory**. Nothing is relaxed; the tool measures the forces at the starting geometry and, above the template's force tolerance, **warns in plain words that the frequencies will be off** — never refuses, because the statement is the person's to make. The hint beside the box says the same before the run is paid for |

Why the warning is plain and the default is to relax *(measured 2026-09-24,
§ 9)*: the H₂ fixture at the experimental bond length, sent through the
SIESTA road unrelaxed, reported 3358 cm⁻¹ where the relaxed bond gives 3022 —
a tenth of the frequency — and the two relaxation tolerances 0.02 and
0.001 eV/Å differ by two wavenumbers. An unrelaxed structure is the one error
that looks like a result.

**How the tool answers the ticked box with numbers**: PySCF's deck checks
the gradient at the input geometry (§ 4.3); SIESTA's read-back reads the
forces SIESTA evaluated at its FC step 0 (§ 5.5). Both judge the largest
absolute force component over the **free** atoms (R5) against the template's
own force tolerance — `geom_gmax` on PySCF, `relax_force_tol` on SIESTA — the
one the person set or left at the kind's recommendation; both write the
number and the verdict into the result, and the viewer shows them.

**The structure can carry its own evidence** *(user direction 2026-09-24:
"if the info meta data is present, we should display it and check against
tolerance of calculation; if not present … we will just accept that but with
a warning/hint")*. A pair exported from the Results tab of a finished
relaxation carries `info.relaxation` — the run's engine, its force tolerance,
the largest force left on the atoms it moved, the atoms it held, and a
fingerprint of the geometry it describes
([`model/parse.md` § 5b.1](?doc=model/parse.md)) — beside `info.calculation`,
the level of theory the deck stated (SIESTA today). Both kinds' gates read
them (`validation/sidecar.py::check_relaxation_record`), and every finding
lands on the box's own card, so the record is displayed where the choice is
made; the Metadata pane shows the raw store.

| the box | the record | the finding |
|---|---|---|
| ticked | absent | the plain warning of the table above, and an info line that no record travels with this structure — the statement stands on its own and the read-back measures it |
| ticked | present, for a different geometry (another frame of that run, or edited since — the fingerprint differs) | **warning**: the record does not vouch for these coordinates |
| ticked | present, for these coordinates | the engine, the tolerance, the steps and the largest remaining force are shown; a different engine, a different level of theory (`info.calculation` against this form's basis, functional, mesh cutoff, electronic temperature), a largest force above **this calculation's** tolerance, or a different held set is each a **warning** naming the number or the field; within tolerance at the same level is an info line |
| unticked | present, for these coordinates, within this calculation's tolerance at the same level and held set | an info line: the record already meets this calculation's criterion, so the box may be ticked and the relaxation skipped |
| unticked | anything else | the same facts as information; the ladder relaxes regardless |

Never a refusal: the record informs the person's explicit choice; it does not
make it. What the PySCF deck's own `_optimized.xyz` pair should record is
V1.31 (§ 10).

### 2.3 Held atoms travel with the structure, never as a form field

Which atoms are held is a fact about the **structure**: the `frozen_atoms`
region in its `.molstruct.json` half of the codec pair
([`model/structure-annotations.md`](?doc=model/structure-annotations.md)). Set
it in the viewer or load a structure that carries it; the Send button exports
the model in one read, so what is drawn held is what the calculation holds.
**Frozen means frozen through every phase** *(user, 2026-08-21)*: the PySCF
relaxation holds the same set (geomeTRIC's `$freeze` file, the optimisation
deck's own mechanism) and the Hessian is taken over the free atoms only; the
SIESTA deck writes the set into `Geometry.Constraints` and nudges the free
range only. The old form field (`frozen_indices`, pre-filled from the sidecar)
was retired at phase P2 of the spectra migration of 2026-08-20
([`archive/2026-08-20-spectra-migration-plan.md`](?doc=archive/2026-08-20-spectra-migration-plan.md))
because a form copy of a structure fact could be edited into disagreement
with the structure it described. The
set is never second-guessed: nothing warns you off a choice you made on
purpose. What the run **says** about the choice is the point — the regime, the
motions removed, the Methods sentence (§ 4.10, § 6).

### 2.4 Where results land — `frequency/` and `spectrum/`

`frequency/` and `spectrum/` are two of the nine folder topics a person picks
from ([`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 2.5),
split by *what is computed* — frequencies and thermochemistry against
intensities — and nothing derives a topic from an engine: a PySCF run with both
intensity flags off belongs in `frequency/` by that description. The table in
§ 1.3 says where a typical run of each engine lands, not where a mechanism
puts it. There is no `frequency` calculation kind; the kinds are
`optimization`, `vibration` and `transport`.

---

## 3. The parameters

### 3.1 One catalogue, narrowed to the kind

Every parameter of every calculation lives in the one catalogue
(`molbuilder/data/catalogue.template.toml`;
[`engines/template.md`](?doc=engines/template.md)). An item declares which
engines it applies to (`engines`, absent = all) and which calculation kinds
select it (`calculations`, absent = every kind — § 6.3 there). The vibration
template, the Spectrum tab's form (`GET /api/build/schema/<engine>?calculation=vibration`)
and the deck's own item lines are all the same narrowing of the same file, so
a parameter is defined once and rendered the same everywhere.

**The vibration-only items** *(the catalogue as of 2026-09-24)*:

| item | engine | what it reaches | default | note |
|---|---|---|---|---|
| `already_relaxed` | both | the person's explicit say (§ 2.2). Unticked: PySCF runs Phase 0, SIESTA runs a `relax` stage first. Ticked: nothing is relaxed; the forces at the starting geometry are measured against the template's force tolerance and a plain warning says the frequencies will be off when they fail | `false` | never refused; the hint beside the box carries the same warning |
| `compute_raman` | pyscf | `COMPUTE_RAMAN` — the polarizability sweep (§ 4.6) | `true` | the expensive optional: about `6·N_free` extra SCFs, each with a response calculation |
| `compute_ir` | pyscf | `COMPUTE_IR` — dipole derivatives (§ 4.6) | `false` | nearly free when it is the only strength asked for and no atom is held; otherwise rides the Raman sweep or its own dipole sweep |
| `temperature_K` · `pressure_atm` | pyscf | `THERMO_T_K`, `THERMO_P_ATM` — the headline of the thermochemistry (§ 4.7) | 298.15 K · 1 atm | **owed on SIESTA**: the derivation sums at these defaults and says so (§ 5.5) |
| `displacement_amplitude_ang` | pyscf | `DISPLACEMENT_AMPLITUDE_ANG` — the ± push along a mode for the electronic-structure probe (§ 4.8) | 0.02 Å | window 0.02–0.20 Å: smaller drowns in SCF noise, larger leaves the harmonic region |
| `es_mode_selection` | pyscf | `ES_MODE_SELECTION` — which modes get the probe (§ 4.8) | `skip` | five choices in the catalogue today — `skip` · `all` · `explicit` · `top_n` · `threshold`; the last two are **retired by decision** (2026-09-23) and still offered until V1.6 lands (§ 10) |
| `es_explicit_indices` | pyscf | the list for `explicit` | `""` | 1-based, `"3, 7, 12"` or `"3-7, 12"`; one format, parsed at the emitter |
| `freq_min_cm1` · `freq_max_cm1` | pyscf | the window `all` (and the two retired selectors) filter by | unset | `skip` selects nothing; ignored by `explicit` — naming a mode is saying *that one* |
| `es_n_homo_below` · `es_n_lumo_above` | pyscf | the orbital window recorded per displaced geometry | 5 · 5 | record size, not cost |
| `net_charge` | both | `NetCharge` / `gto.M(charge=)` | auto | shared with every kind; resolved once by `chemistry.resolve_net_charge` (explicit wins, 0 included; unset runs the phosphate rule — one negative charge per nucleic-acid backbone phosphate, [`model/chemistry.md`](?doc=model/chemistry.md)) |
| `fc_displacement` | siesta | `FC.Displacement` — the nudge of the force-constant run (§ 5.3) | 0.04 Bohr | range 0.005–0.2 Bohr; smaller pushes the force difference toward the SCF noise floor, larger picks up anharmonic terms |
| `relax_type` · `relax_steps` · `relax_force_tol` · `relax_max_displ` | siesta | the `relax` stage's driver, step cap, force tolerance and largest step (§ 5.2a) — the same items the optimization kind shows | the kind's **recommended** values are the tight tier: Broyden, 100 steps, **0.01 eV/Å**, 0.02 Å (`recommended = { vibration = … }` in the catalogue, [`engines/template.md`](?doc=engines/template.md) § 6.3a) | editable like every other item; the tolerance is also the yardstick the read-back judges a ticked box by (§ 5.5) |
| `geom_gmax` · `geom_grms` · `geom_dmax` · `geom_drms` · `geom_etol` · `geom_max_steps` | pyscf | Phase 0's geomeTRIC criteria (§ 4.2) — the optimization kind's own items | the kind's **recommended** values are the tight tier: `geom_gmax` **2·10⁻⁴ Eh/Bohr** (0.010 eV/Å), `geom_grms` 1·10⁻⁴, `geom_dmax` 1·10⁻³ Å, `geom_drms` 5·10⁻⁴ Å, `geom_etol` 1·10⁻⁶ Eh, 100 steps | the general default stays the medium tier for an optimization; the vibration kind recommends tight because a frequency deserves a real stationary point |

**Where each item sits on the form** — the card is the item's `group`, the legend
inside it the first `category` ([`engines/template.md`](?doc=engines/template.md)
§ 6.2 and the key table there), and the vocabulary decides: what the run
*computes* is `profile` (`already_relaxed`, `compute_raman`, `compute_ir`,
`temperature_K`, `pressure_atm`, and the probe's five selectors
`es_mode_selection` · `es_explicit_indices` · `freq_min_cm1` · `freq_max_cm1`
and the two retired ones); a numerical step size is `stage` under *accuracy*
(`displacement_amplitude_ang`, `fc_displacement`), the set a staged sequence
may tighten; a record size is `output` (`es_n_homo_below`, `es_n_lumo_above`).
The selectors are not convergence targets and nothing steps them per stage —
they sat on the `stage` card until 2026-09-24, where Task setup also offered
them as *vary per stage*.

**The engine is not an item.** Which program runs a described job is the
description's `engine` (`task.json`), chosen on the Spectrum tab's engine
strip the way the Structure-optimization tab has always chosen it, carried by
the hand-over, refused by `init` for anything but the two engines. A
one-choice `engine` form item stood in for the strip until 2026-09-24 and
retired with it: a parameter of the deck it never was.

**The shared items** — method, functional, basis, spin, dispersion, density
fitting, the implicit solvent (`solvent`, PCM), the SCF machinery, the
geometry-convergence criteria (`geom_gmax` family, whose values for this
kind are the tight tier through the catalogue's `recommended` key — the row
above; V1.20 closed 2026-09-24), the relaxation's workflow knobs
(`on_nonconvergence`, `geom_max_steps`, `geom_continue_retries`, `optimizer`,
`write_trajectory`, `write_molwatch_log`, `save_initial_xyz`,
`save_optimized_xyz` — § 4.2 says what each does here), the execution category
(threads, memory, GPU, `mpi_np`) — are the engines' own rows, selected into the
vibration template with the kind's defaults.

**Why Raman is on by default and infrared off** *(user, 2026-08-20: one
calculation, two independent toggles)*: the tab began as a Raman tool, and with
Raman on the infrared derivative rides its sweep for free (§ 4.6), so the
analytic infrared route only matters to a person who switches Raman off. A
frequencies-only run turns both off. Their meaning is
[`engines/pyscf.md`](?doc=engines/pyscf.md) and
[`engines/siesta.md`](?doc=engines/siesta.md); the SCF knobs reach every mean
field the PySCF deck builds through one generated dresser (§ 4.3).

**Items the kind does NOT show, and why:** `restart` (the SIESTA start state
is the kind's own, § 5.3; the PySCF deck carries no deck-level restart state);
`frozen_indices` (retired, § 2.3); `SpectraConfig` (a 33-field class
nothing constructed, retired 2026-08-22 — the kind's science is the kind's);
and **no "how many layers should move" question** (withdrawn 2026-09-21: the
structure already carries the answer, and comparing two freeze depths is two
structures and two runs, which works today — § 11). The tool computes what it
is given and asks nothing new.

### 3.2 The honesty gate

A form that shows a knob the calculation ignores is a lie *(user, 2026-08-21)*.
`tests/test_vibration_form_honesty.py` renders the deck once per offered
parameter with a non-default value and requires the deck text to change; a
parameter still awaiting its integration sits on an explicit list in that test,
each row naming its plan item, and the list only shrinks. The kind validator
refuses by name what the render cannot honour (`validation/spectra.py`).

### 3.3 What the person is told before paying for the run

The live checks on the Spectrum tab and the settings gate at `prep` run the same
function, `validation.validate(struct, cfg, calculation="vibration")`, which
dispatches on the config class — `spectra_render_checks` for PySCF,
`siesta_vibration_checks` for SIESTA — and **refuses any other class by name**
rather than returning an empty verdict (an empty list reads as *checked,
nothing found* on every surface — the silent skip that finding **F4** of
[`science/validation.md`](?doc=science/validation.md) forbids: derived facts
are derived server-side, inside the one gate). Both say, from the one rank
rule: how many atoms are held, how many whole-body motions survive and will
be removed, and how many modes will be reported (R7). PySCF's adds the
finite-difference and amplitude checks, the parity of the electron count, and
the cost of what was ticked, stated from what the code does (R8); SIESTA's
checks are § 5.8. A structure with a repeating axis is refused at the PySCF
gate rather than computed as a cluster.

---

## 4. The PySCF route — the generated script, phase by phase

The deck is a Python script generated by `pyscf/vibration_deck.py` (the
composer) from blocks in `pyscf/vibration_emitters.py`, through the same seam
every deck uses — `pyscf.input.spec_for(struct, cfg, calculation="vibration")`
→ `script_emit.prepare_deck` ([`execution/script-preparation.md`](?doc=execution/script-preparation.md)
§ 4). It runs in the `molbuilder-pySCF` env, needs no molbuilder at run time,
and writes `<label>.spectra.json` beside itself after every phase.

### 4.1 The script, top to bottom

```text
# constants from the description: JOB, ATOMS, ELEMENTS, N_ATOMS, FROZEN_INDICES_USER
# (the held set as the structure gave it) → FROZEN_ATOM_IDXS / FREE_ATOM_IDXS,
# ALREADY_RELAXED, GEOM_* criteria, THERMO_T_K, THERMO_P_ATM, THERMO_T_GRID (the deck's
# copy of normal_modes.THERMO_GRID_K), COMPUTE_IR, COMPUTE_RAMAN, RAMAN_FD_STEP_ANG,
# DISPLACEMENT_AMPLITUDE_ANG,
# ES_MODE_SELECTION, ..., and the spliced functions (homo_index, dipole_derivatives,
# rigid_motions, vibrational_modes, vibrational_thermo, vibrational_thermo_grid,
# structure_hash_text) — source text copied from their one home, so the deck
# imports nothing of molbuilder.

mol = gto.M(atoms = EVERY atom, held ones included; basis, charge, spin, ...)
_mb_configure_scf / _mb_configure_dft   # the generated dressers (§ 4.3)
state = {schema_version: 6, engine: 'pyscf', phases: empty, ...}; write the artifact

# Phase 0 — relaxation (§ 4.2)
if not ALREADY_RELAXED:
    mol = geomeTRIC(build_mf(mol), freeze = FROZEN_ATOM_IDXS).run()   # the free atoms relax
COORDS_EQ_ANG = mol.atom_coords(unit='Angstrom')                     # the Hessian's geometry

# Phase 1 — the equilibrium SCF (§ 4.3)
mf = build_mf(COORDS_EQ_ANG).run();  E_eq, MO_ENERGIES_EQ, HOMO_IDX
check  max|F_a| over a ∈ FREE  <  GEOM_GMAX        # R5 — warns with the numbers, never refuses

# Phase 2 — the Hessian and the one harmonic path (§ 4.4, § 4.5)
HESS, DMU_DR, IR_ROUTE = dipole_derivatives(mf_for_hess, FREE_ATOM_IDXS, want_analytic_ir)
λ, L, patterns = vibrational_modes(HESS, MASSES_AMU, COORDS_EQ_ANG, FROZEN_ATOM_IDXS, axis_kind)
                                   # cell= is optional and omitted: an isolated molecule
FREQ_CM1 = sign(λ)·√|λ|·5140.487 ;  state['modes'], state['removed_motions']

# thermochemistry (§ 4.7)
state['thermo'] = RRHO (nothing held) | vibrational sums above E_eq (atoms held), headline + grid

# Phase 3 — intensities (§ 4.6)
Raman: for each free Cartesian coordinate, ±RAMAN_FD_STEP_ANG: SCF + analytic polarizability
       (and the dipole, read for free) → dα/dR, dμ/dR → per-mode activities and intensities
IR alone: analytic dμ/dR from the Hessian's own response (no atom held), else the dipole sweep

# Phase 4 — per-mode electronic structure (§ 4.8)
for each selected mode: SCF at q ± A·L_display → the orbital window around HOMO/LUMO

write the artifact; print the summary
```

*(Every name above is explained in § 4.2–§ 4.10; a first reading can skip the
listing and return to it.)* Every construction site of a mean field calls the same generated
`_mb_configure_scf(mf)` (and `_mb_configure_dft(mf)` on a DFT deck) —
[`engines/pyscf.md`](?doc=engines/pyscf.md) § 7a: the framework never spells an
SCF knob twice, and a deck that builds many mean fields (equilibrium,
displaced points, relaxation) inherits one definition with N call sites.

### 4.2 Phase 0 — the relaxation, the measurement's precondition

`geomeTRIC` runs in-process before the equilibrium SCF, on the free atoms,
with the held set written to its `$freeze` constraints file exactly as the
optimisation deck does. It is a **tracked phase**: `phase_relaxation` goes
`empty → running → complete`, and every step writes the artifact with the step
count and the current largest force, so the viewer's chip shows convergence
ticking down *(user, 2026-08-20: a silent gap while geomeTRIC works, likely the
longest part of the run, would betray "the viewer tracks all the steps")*.

What the phase honours from the description: the convergence criteria
(`geom_gmax`, `geom_grms`, `geom_dmax`, `geom_drms`, `geom_etol`,
`geom_max_steps` — the tight tier at the kind's recommendation, § 3.1), `on_nonconvergence`
(**this** is the phase that policy governs — `proceed` takes the partial
geometry and records `converged: null` with a warning; `continue` re-runs the
optimiser with the optimisation deck's retry budget; `halt` raises),
`write_trajectory` (geomeTRIC's streaming XYZ), `write_molwatch_log` (the same
live-watch hooks the optimisation deck emits), `save_initial_xyz` /
`save_optimized_xyz` (`<job>_initial.xyz`, `<job>_optimized.xyz`, written as
**pairs**, geometry plus `.molstruct.json`, through the codec — a bare `.xyz`
carries no labels, no cell and no identity; fixed 2026-09-22 and proven by a
held set `frozen_atoms: [6, 7]` surviving into the output and reading back
through molbuilder's own codec). `optimizer` is geomeTRIC by refusal: pyberny is absent from the
run environment and has no step callback for a tracked phase.

`already_relaxed = true` skips the optimiser and marks the phase
**complete by assertion**; the gradient check of § 4.3 then carries the number
the assertion is judged by. The rule for what to freeze: the set held for the
spectrum should be a subset of the set held for the relaxation that produced
the geometry — holding *more* for the spectrum is always safe, holding *less*
puts the free atoms off a stationary point in the very subspace that is
diagonalised.

**The geometry in the file is the geometry the Hessian is taken at** —
`COORDS_EQ_ANG`, rebound by this phase — so the eigenvectors and the
coordinates in `equilibrium.positions_ang` belong to the same geometry. *(Until
2026-09-24 the file carried the input coordinates: measured 0.0488 Å off on a
0.96 Å bond after a relaxation. Status in § 10.)*

### 4.3 Phase 1 — the equilibrium SCF, and the check on the geometry

One whole-system SCF at `COORDS_EQ_ANG`, the held atoms present, on the mean
field the dresser configured (chkfile written, GPU promotion when the probe of
§ 4.4 allows it, `newton()` when `scf_soscf` is on). It **halts unconditionally**
on non-convergence: this density feeds the Hessian, every intensity and the
thermochemistry, so no policy makes it optional.

It records the **equilibrium block**: `scf_energy_eh` (the SCF's return value,
stored as is), `mo_energies_eh` (non-finite entries dropped by the deck's
`_filter_finite`) and `homo_idx` (derived by the spliced `homo_index`: sum the
two spin channels of a 2-D `mo_occ`, take the highest index with occupancy
above 0.5 — a rule with a branch whose failure would be silent and would
affect only open-shell work, which is why it was lifted out of inline script
text on 2026-09-09 so one implementation runs and is tested), plus `elements`
and `positions_ang`.

**The stationarity check (R5)** takes the SCF's nuclear gradient and judges
the largest absolute force component **over the free atoms** against the
template's own `geom_gmax` — 2·10⁻⁴ Eh/Bohr at the kind's recommendation —
the one rule both routes judge by (§ 2.2; SIESTA's read-back judges the same
quantity against `relax_force_tol`, § 5.5) *(ten times it until 2026-09-24)*,
recording it as `relaxation.max_force_eh_bohr` and the all-atom figure beside
it as `max_force_all_atoms_eh_bohr` — the keys say their unit *(they said
`_a` until 2026-09-24 while the viewer printed "Eh/Å"; the number was always
Eh/Bohr)*. Measured
2026-09-22, why the free-atom rule matters: water with O and one H held,
relaxed to the deck's own criterion, reads 3.35·10⁻² over all atoms against
7.01·10⁻⁵ on the free ones; CO₂ with both O held 7.2·10⁻² against
3.9·10⁻¹⁴. Judging every atom (the old `np.abs(_g0).max()`) warned on four of
five correct constrained minima, and the warning reached the web UI.

**The structure hash** (`structure_hash`, `sha256:…`) is computed by
`sidecars.spectra.structure_hash_text` over `n_atoms`, the label and one line
per input atom, spliced into the deck so the SIESTA derivation and the PySCF
deck cannot hash differently. It is provenance for a reader; nothing enforces
it — its docstring's promise that *the parser can refuse to merge results from
a different molecule* is not kept, because it carries the job name as its
second line and never matches the codec's own pair hash, which is over the
document's bytes, a different scheme. The identity rule of § 10 (V1.9)
replaces it.

### 4.4 Phase 2 — the Hessian over the free atoms

**The rule** (R8): with atoms held, second derivatives are computed **for the
free atoms only**, and the run says so (`hessian_scope = 'free'`,
`n_atoms_in_hessian`). *(The audit of 2026-09-22 had ruled this an opt-in
with analytic infrared as the default; that ruling was **superseded on
2026-09-23 when the route was built**: no option was added, holding atoms is
what asks for the reduced calculation, and the run states which way it went.
The archived design records both.)* The function named `dipole_derivatives`
is the one place this is decided — the name is historical (it first produced
dμ/dR beside the Hessian), and it is the sole producer of the Hessian in the
deck. With nothing held the
Hessian is the whole molecule's and `hessian_scope = 'all'`. The spliced
`dipole_derivatives(mf, free_atom_idxs, want_ir)` is the one place this is
decided:

```text
if some atom is held:
    mf_h  = rebuild(COORDS_EQ_ANG, density_fit = False)      # PySCF's density-fitted Hessian
                                                              # class takes no atom list
    hobj  = mf_h.Hessian()
    H_AA  = hobj.hess_elec(atmlst = FREE) + hobj.hess_nuc(mol, atmlst = FREE)
            + hobj.get_dispersion()[FREE, FREE]               # kernel(atmlst=) adds this term FULL-SIZE
    H = zeros(N, N, 3, 3);  H[FREE, FREE] = H_AA              # the block is numbered by POSITION in the
                                                              # list passed, not by the atom's index, so it
                                                              # is placed back by index; held rows stay
                                                              # zero and are never read
    route = 'finite-difference' if IR is wanted else 'none'   # the analytic dμ/dR route takes no atom list
else:
    IR alone wanted → pyscf.prop.infrared: Hessian + dμ/dR from ONE response solve (§ 4.6)
    otherwise       → mf.Hessian().kernel()
```

Two corrections found by measurement and kept: PySCF 2.14's density-fitted
Hessian class fails on a partial list (`pyscf/df/hessian/rhf.py:216`, a shape
mismatch), so the reduced route rebuilds a plain mean field and records
`hessian_density_fit = false`; and `kernel(atmlst=)` cannot be used because its
dispersion term is full-size, so the three pieces are summed by hand. The
block agrees with compute-everything-and-slice to **1·10⁻⁸ Hartree/Bohr² for
Hartree–Fock** and **7.5·10⁻⁶ for DFT at grid level 4** (1.5·10⁻⁵ at level 3) —
the difference entirely in the coupled-perturbed response part (the static
part agrees to 6·10⁻¹⁴, and the figure is unchanged by the solver's cycle
cap), whose tolerance PySCF scales with the number of atoms in a batch
(`pyscf/hessian/rhf.py:330`), so the full calculation converges its response
more loosely than the partial one; about 0.03 cm⁻¹ on a stretch. Pinned by `tests/test_vibration_e2e.py` with and
without a dispersion correction.

**GPU.** A run-time probe (`_emit_gpu_coverage_probe`) asks whether gpu4pyscf
covers the Hessian for this SCF type; if not, the mean field is rebuilt on the
CPU for this phase — one extra SCF, and the only path when the coverage is
absent. The reduced route with a GPU mean field is untested (§ 10).

**What the reduced calculation skips, read from PySCF's own code**
(`pyscf/hessian/rhf.py`, `rks.py`, version 2.14), not from a timing:

| piece of the analytic Hessian | runs over | 300 atoms, 50 free |
|---|---|---|
| the SCF, and one rebuild without density fitting when atoms are held | every atom, once each | a 300-atom SCF, twice |
| `_partial_hess_ejk`: the `int2e_ipip1` diagonal contraction | every atom, once | unchanged |
| `_partial_hess_ejk`: the `int2e_ip1ip2` / `int2e_ipvip1` contractions, one per atom with `shls_slice` on that atom's shells | **the free atoms** | 50 of 300 — about a sixth |
| `make_h1`: the `int2e_ip1` contraction, one per atom | **the free atoms** | about a sixth |
| `solve_mo1`: the coupled-perturbed equations, `3·len(atmlst)` perturbations in memory-sized batches | **the free atoms** | 150 perturbations instead of 900 |
| `hess_elec`: the final pair loop | free × free | a thirty-sixth |
| `hess_nuc` | every pair, then sliced | trivial |
| **DFT only** — `_get_vxc_diag`; `_get_vxc_deriv2` with `vmat = zeros((natm, 3, 3, nao, nao))` and `for ia in range(mol.natm)`; `_get_vxc_deriv1` with `(natm, 3, nao, nao)` | **every atom** | unchanged — and it is memory: `300 × 9 × nao²` doubles, about 200 GB at 3 000 basis functions |

So the pieces that dominate *time* — the response equations and the
two-electron derivative contractions — scale with the free atoms, and a
300-atom system with 250 held costs those pieces about what a 50-atom system
would. The exchange-correlation derivative matrices do not shrink: PySCF builds
them for every atom whatever list it is given, and at a few hundred atoms their
size is what stops the run. **Consequence:** the PySCF analytic route is for
molecules of tens of atoms, and for anything without a functional
(Hartree–Fock has no such term); a junction of hundreds of atoms goes to the
SIESTA route (§ 5.7). *(The frozen-atom cost claim — `engines/overview.md`'s *"cuts the cost
sharply"* and the pre-run advisory's *"the cost saving is typically large"* —
was false until 2026-09-23: the full Hessian was computed and sliced, measured
10.6 s free against 10.1 s with 2 of 14 held. R8 now requires a cost statement
to be read from the code.)*

### 4.5 The one harmonic path

`spectra/normal_modes.py::vibrational_modes(hessian, masses, positions, held, axis_kind, cell)`
— spliced into the deck as source, called on the host by the SIESTA read-back —
is the only place modes are made. In order: take the free–free block; divide
by `√(m_i m_j)` per 3×3; build the surviving whole-body motions with
`rigid_motions` (three slides, the turns the lattice permits, then only the
combinations that leave every held atom in place, then only those that move a
free atom — the rank of [`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 3.1, tolerance 10⁻³ Å per unit motion); diagonalise in the **complement** of
those motions in the mass-weighted metric; return `3·N_free − n_rigid`
eigenvalues (ascending; negative = imaginary), the modes in the canonical
normalisation `Σ m_k |L_k|² = 1` (amu), and the removed patterns.

Conventions that must not fork: isotope-averaged masses (`atomic_mass`, the
same table PySCF's `thermo` uses); the wavenumber conversion
`√λ × 5140.487…` for a Hessian in Hartree/Bohr² weighted in amu
(`constants.CM1_PER_SQRT_HARTREE_BOHR2_AMU`, derived from the two constants it
is made of); the display form `eigenvector_display` = the canonical vector
rescaled per mode so `max|L_k| = 1`, for the animation only (§ 6.3). Both
conventions were measured on 2026-09-22: the wavenumber constant agrees with
PySCF's own derivation to 1·10⁻⁹ relative, and `Σ m|L|²` comes out
1.00000000 on the free path and on every held system. The
artifact records what was removed (`removed_motions.count`, `patterns`), so a
reader can see the difference between `3·N_free` and `len(modes)` (R7).

The gate this rests on: the rank rule reproduces PySCF's own
`harmonic_analysis` on free molecules — water, CO₂, HF, methane at
RHF/STO-3G: eigenvalues to 1·10⁻⁸ relative, wavenumbers to 1·10⁻⁴ cm⁻¹,
water's and HF's vectors to 1·10⁻⁶ in the mass metric (methane's degenerate
sets compared by frequency only) — before the old two-branch analysis was
deleted (2026-09-23).

### 4.6 Phase 3 — infrared and Raman strengths

**What an intensity is.** An infrared band is strong when the vibration moves
charge: the quantity is the dipole's rate of change along the mode,

```text
    I  =  42.2561 × |dμ/dQ|²     km/mol,   μ in Debye, Q in Å·√amu,
    dμ/dQ_n = Σ_{k,α} (dμ/dR_{k,α}) · L_canonical_{k,α,n}
```

and a Raman band scatters when the mode changes the polarizability: `dα/dQ`,
combined by Placzek's formula `S = 45 a² + 7 γ²` into an activity in Å⁴/amu.
The `42.2561` prefactor is derived for modes normalised in **amu** — the
convention of § 4.5 — which is where the 1823× defect lived.

**Three ways to obtain dμ/dR, and what each costs** (measured NH₃/PBE0/6-31G,
2026-09-11; the two tensors agree to 0.02 %, so the choice is cost, not
accuracy):

| route | how | cost | availability |
|---|---|---|---|
| **analytic** | the same coupled-perturbed response the Hessian solves, contracted with dipole integrals — one solve for both | **+14 %** over the Hessian alone | `pyscf.prop.infrared`, master branch only, never released to PyPI |
| **finite-difference dipoles** | nudge each free coordinate ±0.005 Å, read the converged dipole, difference | **+486 %** — `6·N_free` extra SCFs | always |
| **rides with Raman** | the Raman sweep already runs those SCFs; the dipole at each point is a one-line read on a converged wavefunction | free | whenever Raman is on |

**The rule for which runs:** analytic **only when infrared is the only strength
asked for and no atom is held** — Raman's sweep would make it buy nothing, and
the analytic module takes no atom list. The choice is made **at run time**,
because `pyscf.prop.infrared` is a property of the environment the deck lands
in, not of the machine that wrote it: the deck tries the analytic route, falls
back loudly (the reason printed into the job log), and records which ran as
`ir_route` (`analytic` · `finite-difference` · `none`) with the step as
`ir_fd_step_ang` when a difference ran. **Asking for intensities must not move
the frequencies**, so the analytic route carries two corrections, both
measured: it is handed the SCF's own `mf.Hessian()` (upstream's class is
hardcoded non-DF, a 7.2·10⁻⁵ Hartree/Bohr² mismatch on a density-fitted SCF,
0.11 cm⁻¹) and the dispersion Hessian is added back (upstream drops it:
7.2·10⁻⁴ Hartree/Bohr², **3.7 cm⁻¹ on every frequency** of a B3LYP-D3BJ run).
With them the Hessian matches the no-infrared path to 3.6·10⁻¹² (density
fitting) and 5.7·10⁻¹² (dispersion) — `max|H(analytic route) − H(plain)| =
6.7·10⁻¹⁶` on the corrected deck — **and the corrected route is faster**
(14.8 s against 14.4 s for the Hessian alone: still one solve). The tensor
layout is measured too: against the finite-difference tensor the emitted
dμ/dR differs by 8.4·10⁻⁵, its transpose by 5.2·10⁻¹.

**Raman** has one route: the static polarizability is analytic
(`pyscf.prop.polarizability`, coupled-perturbed; the polarizability points run
**without density fitting** because the module has no DF implementation) at
each displaced geometry, and its **derivative** is a central difference over
the free Cartesian coordinates at ±`RAMAN_FD_STEP_ANG` = 0.005 Å (the step
Gaussian and ORCA use for static polarizability derivatives). PySCF reports the
polarizability in atomic units (Bohr³), so one global factor `(Bohr/Å)⁶ ≈
0.02197` is applied on the final scalar, and the stored value is genuine Å⁴/amu,
comparable to Gaussian's and ORCA's activity columns — without it the
spectrum's shape is right and every absolute activity is about 50× too small. The run records the
route as `raman_route = 'finite-difference'` with `raman_fd_step_ang` (§ 10 for
status), and the Methods text states the method one way: analytic
polarizability, finite-difference derivative.

**Where the numbers are not well defined.** A charged molecule's dipole depends
on the origin, so its infrared intensities carry a bookkeeping term; they are
computed, said so, and treated with suspicion. With atoms held the formulas
consume the free-atom vectors and derivatives, both of which exist, but a held
atom contributes no dipole derivative: for a molecule anchored at one or two
atoms a small correction, for a molecule on a metal that screens and carries
the charge transfer, not small. The number is computed; how much it means is
the reader's judgement.

**What is validated** (§ 9): band-level — water at B3LYP/def2-SVP in the
literature windows with the right ordering; CO₂ reproducing the mutual
exclusion of a centrosymmetric molecule. **Not done**: a mode-by-mode
cross-check against an external code.

### 4.7 Thermochemistry

The harmonic sums run over the reported modes — every one a vibration (R4) —
minus the imaginary ones, which have no partition function and whose count is
**stated** (`thermo.n_imag_excluded`), never dropped in silence; the motions
removed before diagonalising are counted beside them (`n_rigid_removed`).
Two regimes, said out loud in `thermo.regime` and `thermo.note`:

| regime | when | what is summed |
|---|---|---|
| `rrho` | nothing held | PySCF's own `thermo.thermo`: electronic + translational + rotational + vibrational, at the headline (T, P) and at every point of the grid |
| `vibrational-only` | any atom held | the vibrational sums above the electronic energy, `vibrational_thermo` / `vibrational_thermo_grid` from the one home — there is no gas-phase translational or rotational partition function to add, and no `kT` (the ideal gas's `pV`) either |

**One quantity under one label**: the headline numbers and the curves the
viewer draws are the same sum, and the headline temperature is a point of the
grid (`THERMO_GRID_K` = 50–1500 K in 30 points, a documented presentation
default, plus the headline T). *(Until 2026-09-24 the headline was full RRHO and
the grid vibrational-only under one note — measured on free water at a
temperature exactly on the grid: headline G = −74.95926467 Eh against the
grid's −74.94057624 at 300 K, ΔG = +11.73 kcal/mol, ΔS = −45.28 cal/mol/K.
Status in § 10.)* On SIESTA no total energy is reported, so `h_eh` and `g_eh`
there are the vibrational contributions alone, above an electronic minimum
taken as zero; the file's `note` says so. The deck computes, the viewer draws; the viewer
derives nothing but the electronic reference it needs to plot.

### 4.8 Phase 4 — the electronic structure along a mode

For each selected mode the deck pushes the geometry to `q ± A·L_display` —
`A = displacement_amplitude_ang` along the **display** form of the eigenvector
(its largest absolute Cartesian component = `A`) — runs an SCF at each, and records the orbital
window `[HOMO − es_n_homo_below, LUMO + es_n_lumo_above]` and the SCF energy at
both, beside the equilibrium ones: **two SCFs per selected mode**. The viewer
draws the three level stacks joined orbital by orbital, the gap's shift, and
the coupling `ΔE/(2A)` — the electron–vibration coupling that decides how a
mode modulates a junction's transmission [Galperin2007, Frederiksen2007].
**Which coordinate.** `A` is the largest absolute Cartesian component of
the motion along the display eigenvector (not the largest per-atom length —
an atom moving off-axis swings up to `√3·A`), so `ΔE/(2A)` is a slope per Å
of that component. The slope per unit
normal coordinate, `∂ε/∂Q_ν = ΔE/(2A) · max|L_canonical|` (the largest absolute component), and the
coupling per zero-point amplitude, `g_ν = (∂ε/∂Q_ν) · √(ħ/2ω)` — the number
the inelastic-transport literature quotes — follow from the file, since both
eigenvector forms are in it ([`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 4b.5 F), and neither is written today. Two points give the slope and not
its linearity over `A`; the five-point sample, `g_ν`, and a molecule-projected
window for a cluster whose HOMO and LUMO are metal states are V1.27 (§ 10).

Which modes get it is `es_mode_selection` with the frequency window
(`spectra/selection.py`, inlined into the deck so a cluster node needs no
molbuilder): `skip` (none), `all` (every mode inside the window), `explicit`
(the listed indices; the window is ignored — naming a mode is saying *that
one*). `top_n` and `threshold` ranked modes by Raman activity and are
**retired by decision** (2026-09-23): the probe measures `∂ε/∂Q`, which
follows its own selection rule — in a centrosymmetric molecule the
Raman-bright modes are exactly the infrared-dark ones, so the filter kept one
symmetry class and dropped the other every time, and for an engine that
computes no strengths they were undefined rather than empty. The window is the
cost control, and `all` is cheap where it matters (8 SCFs for CO₂). A mode
whose electronic structure is already in the file is skipped on a resume,
whatever the selector says.

### 4.9 The live artifact

The script writes `<label>.spectra.json` at the end of every phase, after
every relaxation step, and after every per-mode SCF, always by **atomic
replace** (a temporary file, then `os.replace`), so a reader polling every two
seconds never sees a torn file. Five phases, four flags — the Hessian and the
harmonic analysis (Phase 2) close `phase_frequencies`, which Phase 1 opened —
are the reader's clock:

```text
phase_relaxation   empty → running (step count, max force ticking) → complete   (complete by assertion under already_relaxed;
                                                                                 `not requested` on SIESTA, whose route relaxes nothing)
phase_frequencies  empty → running → complete
phase_raman        empty → running → complete      (`not requested` from the first write when the description
                                                    asked for no Raman sweep, and on SIESTA, whose route has none)
phase_es           empty → running → complete      (per-mode entries fill in one at a time; `not requested` under
                                                    `es_mode_selection = skip`, and on SIESTA)
```

**`not requested` is a fourth, terminal state** *(built 2026-09-24, V1.5)*: a
phase the description never asked for is `not requested` from the first write
and never changes, so a reader cannot mistake *nothing was asked* for *nothing
has happened yet* (`empty`) or for *done with nothing behind it* (`complete`,
which both writers used to write). The viewer counts it as finished when it
decides whether the run is done, draws its chip as such, and prints it as the
phase's status. Only a phase that was asked for is ever `running`.

The Results-tab viewer polls the file while any phase is running and redraws as
modes arrive ([`web/spectra.md`](?doc=web/spectra.md) § 7).

### 4.10 The Methods paragraph

`spectra/methods.py::render_methods_md` composes one Markdown paragraph from
the config — level of theory, basis, dispersion, the atom clause (how many
free, how many held, how many modes by R2), the selector and window, the
amplitude convention — takes the engine's own fragment as an argument
(`pyscf_methods_fragment`: the free-atom Hessian sentence citing
[Head1997, LiJensen2002, Besley2008], the density-fitting note, the Raman
sentence with its step), and extracts the bibliography keys back out of the
prose it rendered, so what is cited is what was said. It ships in the script's
header and beside the result. **Before the run it is route-neutral** for
infrared — which dμ/dR route runs is settled inside the job — and the load path
adds the route sentence from `ir_route` and `ir_fd_step_ang`
(`with_ir_route`, § 6). The count it states is R2 from the one derivation
(R6); the composer also has a `results=` arm that would interpolate the run's
own list and frequency span, and **nothing in production calls it** — the
only caller is the deck composer, before the run (V1.18). Two defects stand:
the paragraph reads `cfg.functional` and `cfg.dispersion` whatever
`cfg.method` is, so a Hartree–Fock run's write-up names a functional it did
not use while its next sentence names `pyscf.hessian.rhf` (§ 10); and that
unused arm.

---

## 5. The SIESTA route — force constants, a sorted copy, and a read-back

### 5.1 What SIESTA does

SIESTA takes no second derivative. A **force-constant run** nudges each atom of
one contiguous range by a fixed displacement along x, y and z, both ways, runs
an SCF at each displaced geometry, and writes the forces on every atom to
`<SystemLabel>.FC`. The keywords, with the spellings SIESTA 5.4.2 honours
(verified against the manual-derived table in `tests/validation/test_siesta.py`
and `parse/fdf.py::_norm` — never against `strings` on the binary, whose
concatenated keyword table matches any prefix; `FC.Displ` "matched" four
times and is not a keyword):

| what | keyword | note |
|---|---|---|
| ask for a force-constant run | `MD.TypeOfRun FC` | `PHONON` is retired |
| first / last atom to nudge | `FC.First` / `FC.Last` | 1-based, **one contiguous range**; `MD.FCFirst` / `MD.FCLast` are deprecated aliases and refused by the deck gates |
| how far to nudge | `FC.Displacement` | Bohr; SIESTA's default 0.04; `MD.FCDispl` deprecated |
| the result | `<SystemLabel>.FC` (and `.FCC`) | § 5.4 |
| the derivatives of H and S | `FC.Save.dHS` | the manual's option for writing ∂H/∂R and ∂S/∂R, the ingredient of the electron–vibration coupling (§ 5.6); not in the manual-derived keyword table this repo checks against, and not used. *(The design called it "present in the shipped binary" on the evidence of `strings`, the evidence this section rejects; its presence is unverified.)* |

SIESTA has no unrecognised-keyword diagnostic, so a misspelled keyword runs at
the engine's own default without a word — the `MD.NumBroydenSteps` failure
class that cost a 444-atom allocation in June. Every keyword this deck writes
is a catalogue anchor or a line of `siesta/vibration_deck.py`, and both gates
that refuse a deprecated spelling run over it.

**`vibra` is not used.** SIESTA ships a utility that turns `.FC` into modes
(installed in the `molbuilder-siesta` env), and it was tried on the
measurement of 2026-09-23. It has its own, older fdf reader (`recoor`), not
SIESTA's: it requires `SystemLabel`, `NumberOfAtoms`, `LatticeConstant`,
`LatticeVectors` (or `LatticeParameters`), `SuperCell_1/2/3`, `BandLines` +
`BandLinesScale` and `Eigenvectors`, and its coordinate reader accepts exactly
four formats — `NotScaledCartesianBohr`, `NotScaledCartesianAng`,
`ScaledCartesian`, `ScaledByLatticeVectors` — none of them the
`AtomicCoordinatesFormat Ang` molbuilder writes; it refused the deck twice
(*"not enough values in Coords line"* after the format was changed to one it
names). So the *one deck, two binaries* trick that serves `tbtrans` is not
available: tbtrans shares SIESTA's reader, `vibra` does not. Nothing it
computes lies outside the one harmonic path, so the file is read on the host
instead (§ 5.5) and the deck carries none of `vibra`'s inputs.

**A standing obligation:** the keyword set above is re-verified against the
manual-derived table on **any SIESTA upgrade** — spellings have moved between
versions, and a stale one runs silently at the engine's default.

### 5.2 The contiguity constraint, and the sorted copy

`FC.First`..`FC.Last` nudges atoms *A through B*; it cannot take a scattered
list. The held set is whatever the person ticked in the viewer, so in general
the free atoms are not one run of atom numbers. Three ways out were weighed
(refuse and tell the person to reorder; nudge the smallest covering range and
discard the extra, which wastes exactly the compute the feature saves; reorder
invisibly with the permutation made a first-class fact) and **reorder was
decided** (2026-09-23), under the contract of
[`model/overview.md`](?doc=model/overview.md) § 2.2:

- `prep` sorts a **copy** with the `held-first` key — held atoms first, free
  atoms last, each in the input's own relative order — through the one sort
  machinery (`transport/sort.py`: `sort_by(struct, "held-first")` hands its
  order to the shared `apply_order`, which remaps every per-atom field, checks
  the bijection and records both directions; a second key over the same
  machine, never a second machine; the categorical `transport` key is the
  other);
- it records `atom-permutation.json` beside the calculation, with the key that
  made the copy (`write_permutation`; [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
  § 6.1):

  ```json
  {"schema": "molbuilder/atom-permutation@1",
   "original_to_sorted": [1, 0], "sorted_to_original": [1, 0], "key": "held-first"}
  ```

- the deck is rendered from the copy; the free range is the tail,
  `FC.First = n_held + 1`, `FC.Last = N`;
- the read-back (§ 5.5) reads the record and puts every per-atom row back in
  the input order through `Permutation.rows_to_input_order` and `original_of`
  — the first reader in the codebase that inverts a permutation, and the only
  way any reader may: **the input order never reaches the engine and the sorted
  order never reaches a person.**

The deck writer checks the shape and refuses by name a copy whose free atoms
are not one trailing run (`vibration_deck.fc_facts`): a deck rendered from the
input order would nudge the wrong atoms and converge without complaint.
TranSIESTA's `elec-pos` demands consecutive atoms for the same reason, which
is why the machinery already existed. **Open**: a structure sorted for two
reasons — a junction sorted for TranSIESTA *and* for FC contiguity — carries
one composed permutation, recorded once (ruled); how it is composed is not
built, and no calculation asks for both today.

```text
input order (the person's):        sorted copy (the engine's):
  1 H  free                          1 H  held   ← Geometry.Constraints: position 1
  2 H  held                          2 H  free   ← FC.First = 2, FC.Last = 2
                                     atom-permutation.json: sorted_to_original = [1, 0], key = held-first
```

### 5.2a The `relax` stage — the relaxation the kind runs when the box is unticked

SIESTA cannot relax and take force constants in one run (`MD.TypeOfRun` is
one thing per run), so the vibration ladder on SIESTA is **two stages when
the box is unticked** — `relax`, then `freq` — and one, `freq`, when it is
ticked (`pyscf/stages.py::vibration_stages(engine, already_relaxed=…)`,
read from the template's own value at `init` and at the hand-over's proposal).

**The `relax` stage is an ordinary SIESTA relaxation deck** rendered from the
same template and the same **sorted copy** (§ 5.2): `MD.TypeOfRun` from
`relax_type`, `MD.MaxForceTol` from `relax_force_tol`, the held atoms in
`Geometry.Constraints`. Nothing is invented for it; `spec_for` renders the
optimization deck for a stage named `relax` inside the vibration kind.

**The relaxed geometry travels as coordinates, not as a restart file.** When
`prep` prepares `freq` and the ladder holds a `relax` stage, it reads the
relaxed geometry from that stage's latest concluded attempt — the last
coordinate block of the stage's own output, through the one SIESTA output
parser, in the sorted order both decks share, **and the cell that run used**
— and writes it as the force-constant deck's coordinates, in that cell
(`jobset/prep.py::_vibration_stage_geometry`). The output rather than
`<label>.XV`, because on the flat shape both stages share one `.XV` and the
force-constant run overwrites it with its last displacement, while the output
carries the stage's token in its name; the cell rather than a re-derived
vacuum box, because the deck otherwise shifts the atoms into a box drawn
around the new bounding box, and a relaxed geometry moved against the
real-space grid is not stationary on that grid any more. The FC deck's start
state stays the kind's (§ 5.3: `MD.UseSaveXV .false.`), because honouring a
found `.XV` is exactly what would take a displaced geometry as the stationary
point on a re-run. `prep` **refuses to prepare `freq` before `relax` has
concluded** when the ladder has one, naming the stage to run first — the job
set's own order, not a guess — and **refuses `freq` when the box is unticked
and the ladder holds no enabled `relax` stage**: the box says *relax first*
and the ladder holds nothing that would, so the description contradicts
itself, and the refusal names the two ways out (add the stage, or state the
structure relaxed) rather than measuring at a geometry nobody chose.

**And the read-back reads the geometry the force constants belong to**: the
first frame of the force-constant run's output (its FC step 0, through the
one SIESTA output parser) is the geometry the projection and the artifact's
`equilibrium.positions_ang` use (§ 5.5) — the relaxed one, or the input one
when the box was ticked. The file's geometry is the Hessian's, on this route
as on PySCF's (§ 4.2).

### 5.3 The deck

`siesta/input.py::spec_for(struct, cfg, calculation="vibration")` renders the
optimisation deck's sections ([`engines/siesta.md`](?doc=engines/siesta.md)
§ 3) **without the geometry-optimisation section** (row 14 there: nothing
relaxes) and with two additions from `siesta/vibration_deck.py`, the kind's
own module: the catalogue section *Force constants* (the one item,
`fc_displacement`, with its unit and note) and the structural block the kind
derives from the structure:

```text
%block Geometry.Constraints
position 1 … n_held               # the held atoms, first in this numbering
%endblock Geometry.Constraints
FC.Displacement   0.04 Bohr       # the catalogue item, through the syntax door
MD.TypeOfRun      FC              # the force-constant run: nothing relaxes here
FC.First          n_held + 1      # the free range is the tail
FC.Last           N
DM.UseSaveDM      .true.          # the start state is the KIND's (below)
MD.UseSaveXV      .false.
```

The stage header line names the range, the counts and the displacement
(`siesta/vibration_deck.py::stage_science`); the record blocks that every deck
carries follow.

**The start state is the kind's, not the description's.** The catalogue offers
`restart` to optimisations only, and a force-constant run has no optimiser
history to resume. The density is read when present (the first displacement's
SCF starts from it; every later one from the previous displacement's, in
memory). The geometry is **never** read back: an FC run leaves its *last
displacement* in `<label>.XV` — measured on the two-atom run, the last free
atom sits `FC.Displacement` off the input along z afterwards — so a deck that
honoured that file would take a nudged geometry as the stationary point and
converge. SIESTA reads the files it finds unless told `.false.`, so the answer
is written, not left out (`start_state_lines`, from the restart group's own
declaration). `siesta/warm-files.toml` carries the `[vibration]` section under
the growth rule of [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 4.2a (a new calculation type is a new section, never a branch) — `.FC` and `.FCC`, inventory-only, since a rerun restarts
at `FC.First` — and says why the base `.XV` row means something else under
this kind.

### 5.4 The run, and what it leaves

```text
SCF at R₀ (every atom present, the held ones too)
for a in FC.First .. FC.Last:
    for α in (x, y, z):
        for s in (−, +):
            R = R₀;  R[a, α] += s · δ
            SCF at R, started from the previous density  →  forces F_b on EVERY atom b
            one row per atom b  →  <SystemLabel>.FC
```

`1 + 6·n_free` force evaluations (§ 5.7 counts what they cost). What is left:

| file | what it is | measured on the fixture (`tests/fixtures/siesta_fc`) |
|---|---|---|
| `<label>.FC` | one header (`n_atoms`, `δ` in Å), then `6·n_free·N` rows of three numbers in **eV/Å²**, in the order displaced atom → direction → side (−, +) → atom | the file's two sides average to 41.713 for the H₂ bond; two single points displaced by hand give −ΔF/2δ = 41.713 |
| `<label>.FCC` | the same with the held atoms' force rows zeroed (SIESTA's "constrained" variant) | the free block is identical, so the reader takes `.FC` and slices |
| `<label>.XV` | the **last displaced** geometry, not the input | the last free atom `FC.Displacement` off along z |
| `<label>.DM`, the usual outputs | the density of the last displacement, the `.out` with the version line | |

The wrapper `launch` writes treats the run like any SIESTA run: it activates
the env, sizes the ranks from the machine record or the description's
`execution` block, logs, and marks the attempt concluded.

### 5.5 The read-back — `jobset summarize run <stage>`

The deliverable of the run is the artifact, and it is derived **on the host**
(`spectra/from_siesta.py`, `parse/engines/siesta_fc.py`), through the same
function the PySCF deck carries:

```text
struct  = the calculation's structure, INPUT order, its positions replaced by the run's FC step 0 (§ 5.2a);  perm = read_permutation(bundle)
sorted  = apply_order(struct, perm.sorted_to_original)          # the copy the deck was written from
fc      = read_fc(<label>.FC)                                    # (n_free, 3, ±, N, 3), eV/Å²
H_AA    = mean over ± of fc[a, α, ·, b, β], a, b ∈ FREE          # the central difference
H_AA    = ½ (H_AA + H_AAᵀ);  × Bohr²/Hartree_eV                  # symmetrise; → Hartree/Bohr²
λ, L, patterns = vibrational_modes(H, masses_amu, R₀_sorted, F_sorted, axis_kind, cell)
rows → input order through Permutation.rows_to_input_order       # the recorded permutation, inverted once
write <label>.spectra.json beside the run:  engine 'siesta', the SIESTA version from the .out,
    intensities null, the MO block null, thermo = the vibrational sums (§ 4.7's second regime),
    hessian_scope 'free'|'all', removed_motions, engine_metadata {fc_file, fc_displacement_ang, fc_range_1based, fc_asymmetry_max_ev_ang2, reference_force_criterion_ev_ang},
    config {engine, calculation, stage}
```

The reader refuses a `.FC` whose row count is not `6·N·n_free`, a range whose
length disagrees with the free set, a record whose two directions are not
inverse bijections, and a sorted copy whose free atoms are not one trailing
run. The modes are at **Γ** (R3) — the centre of the Brillouin zone, `q = 0`, where
every cell moves in phase: a force-constant run over the cell as given is the
Γ matrix; a phonon dispersion over `q ≠ 0` is a different feature and is not
this one. The thermochemistry is summed at 298.15 K and 1 atm because the headline
items are PySCF's (§ 3.1) — owed. Measured end to end through jobset on this
workstation (`tests/test_siesta_vibration_e2e.py`, SIESTA 5.4.2): H₂ with the
held atom **last** in the input → one mode (3358 cm⁻¹ on the unrelaxed bond
of 2026-09-23; 3022 on the relaxed fixture, § 9), two motions removed,
the free atom reported as atom 0.

**What the read-back judges (built 2026-09-24).** Stationarity: SIESTA
evaluates the undisplaced geometry as its FC step 0 before the first nudge,
and its forces are the first `siesta: Atomic forces` block of the run's
output. `summarize` reads them (`parse/engines/siesta_fc.py::reference_forces_from_out`)
and writes `relaxation.max_force_eh_bohr` — the largest over the **free**
atoms (R5) — beside `max_force_all_atoms_eh_bohr`, and `converged` judged
against **this description's own `relax_force_tol`** — the item is on the
vibration template (§ 3.1), so the yardstick is the tolerance the person set
or left at the kind's recommendation, resolved for the stage the way `prep`
resolves it; `engine_metadata.reference_force_criterion_ev_ang` records the
number used. *(Until 2026-09-24 the catalogue's general default stood in for
it.)* The positions the projection and the artifact use are the run's own
reference geometry, its FC step 0 (§ 5.2a). The judged
number is the **largest absolute Cartesian component** over the free atoms, the
convention the PySCF deck's own check uses (§ 4.3). Above it the block carries a warning naming the number and the
two ways out, `summarize` prints it, and the viewer's relaxation row shows the
force. And the block's honesty about its own numerics:
`engine_metadata.fc_asymmetry_max_ev_ang2` is `max |H_ij − H_ji|` over the
free block **before** it is symmetrised — the first number to look at when
`FC.Displacement` is suspected of being too small (noise) or too large
(anharmonicity); on a block whose off-diagonals vanish by symmetry — H₂ on
its axis, measured 10⁻¹² eV/Å² — it says nothing, and an atom at a
low-symmetry site shows it from one free atom on.

**What the measurement taught** (H₂ with one atom held, GGA/DZP, § 9): the
unrelaxed experimental bond, 1.27 eV/Å on the reference step, gave 3358 cm⁻¹;
the same bond relaxed to 0.02 eV/Å gave 3024.4 and to 0.001 eV/Å gave 3022.3
at the same step — a tenth of the frequency from the missing relaxation, two
wavenumbers from the tolerance. The δ ladder is § 9's second table.

### 5.6 What is absent, never zero — and the transport connection

**Infrared and Raman are not offered on SIESTA**, and the controls are **not
drawn** for it rather than defaulted off — a control that silently does nothing
is worse than an absent one, because the person believes they asked for
something. This is a statement about this tool, not about SIESTA: infrared
intensities are reachable there by a different route — **Born effective
charges** from the Berry-phase polarisation, the right answer for a periodic
system where a molecular dipole is not defined — and that is a separate
feature with its own physics and validation, to be designed as one if wanted.
The molecular-orbital block is absent because a periodic system has a Fermi
level, not a HOMO. Nothing is relaxed (§ 2.2).

**Why the SIESTA route exists at all — consistency of the potential-energy
surface.** A junction study is a chain: relax → find the vibrations →
displace along one → compute transport. Done in one program, every step sees
the same pseudopotentials, the same numerical orbitals, the same functional,
the same slab and k-points; done in another, the mode displaced along is not a
mode of the system the transport runs through. And the modes differ in the way
that matters — a mode of BDT on gold includes the Au–S interface stretching,
`Au ⟷ S — C₆H₄ — S ⟷ Au`, which is what modulates the electrode–molecule
coupling and which an isolated-molecule calculation cannot produce.

Two levels of that connection, neither built:

- **Level one — displace along a mode and look.** `positions(Q) = R₀ + Q · L`
  swept from negative to positive gives real geometries caught at successive
  points of one vibration; transport at each gives how the current-carrying
  ability changes across it (`Q < 0`: Au–S shorter, stronger coupling). The
  PySCF deck already does the displacement half for its electronic-structure
  probe (§ 4.8). **How far is `Q`** is a physical quantity with one right
  answer, not a knob: the zero-point amplitude `Q_rms = √(ħ/2ω)` and its
  thermal growth, exactly the amplitudes [`web/spectra.md`](?doc=web/spectra.md)
  § 4.1 derives for the animation, and they pair with the **canonical**
  eigenvector only — the convention mismatch that put intensities out by
  1823× is the same family of error.
- **Level two — the coupling itself.** `FC.Save.dHS` writes how the Hamiltonian
  and overlap change when each atom moves; combined with a mode's pattern that
  is the electron–vibration coupling for the mode, the systematic route to
  inelastic tunnelling spectroscopy [Frederiksen2007, Galperin2007]. A research
  capability, recorded so the design does not foreclose it.

**Named since, from the discussion's update** ([`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 4b.6 G, § 4b.7): the displaced pair at the zero-point amplitude written as
a structure pair of its own, the held atoms unmoved (V1.25); the
density-difference map `Δρ_ν(r) = ρ(r, +Q_ν) − ρ(r, −Q_ν)` from SIESTA's
density grid at the two points — the discussion's intermediate quantity
before any oscillator strength, which says whether a mode polarises the
molecule, moves charge across the Au–S bond or drives the metal's screening
(V1.26); and the projected density of states, resonance energies and widths
along the mode, each a run of its own kind on that pair, with `T(E, Q)` the
transport kind on it. Each is a feature to design as one.

### 5.7 What it costs, by construction

`1 + 6·n_free` force evaluations, each a whole-system SCF started from the
previous one's density, and the memory of one SIESTA SCF whatever `n_free` is.
For 300 atoms with 50 free: 301 evaluations instead of 1 801. There is no
all-atom matrix anywhere in the route, which is why it is the one that scales
to a junction (compare § 4.4's last row). Holding atoms buys exactly the
proportion of evaluations it removes, and nothing on the size of each.

### 5.8 The checks a SIESTA vibration is told before it runs

`validation/spectra.py::siesta_vibration_checks`: every atom held is an error
(nothing to nudge); an index outside the structure is an error; with atoms
held, the rank-derived count of surviving motions and how many modes will be
reported (R7), and the statement that the held atoms sit outside the FC range
so no force constant is taken with respect to them, that frequencies are those
of the free atoms in the static field of the held ones, that thermochemistry
is vibrational-only and that intensities are not computed; the precondition
(§ 2.2): with the box unticked, an info line that the ladder relaxes first, to
the template's force tolerance, before the force constants; with it ticked, a
**warning** that the frequencies will be off if the structure is not relaxed
at this level of theory, and that the read-back measures the reference-step
forces against that tolerance (§ 5.5); the relaxation-record findings of
§ 2.2 when the structure carries one; and the
unconsumed-region-label notice every kind carries. The SIESTA engine
validator defers that notice and its own *held during relaxation* line on
this kind — one fact, one finding ([`science/validation.md`](?doc=science/validation.md) § 7).

---

## 6. The result file — `<label>.spectra.json`

One file, every engine, written by the PySCF deck itself and by the SIESTA
read-back; read by `sidecars.spectra.parse_spectra_json` →
`SpectraResults.from_dict` (`spectra/results.py`), and served to the browser by
`POST /api/spectra/load` with one derived field added at load (§ 6.6). It is
registered in [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 6.1 with `atom-permutation.json` beside it.

### 6.1 Schema history

| version | date | change |
|---|---|---|
| 1 | — | one eigenvector per mode, used for both the animation and the Raman projection — recorded as a correctness defect when the two uses were separated |
| 2 | — | `eigenvector_canonical` (`Σ m\|L\|² = 1`) and `eigenvector_display` (`max\|L\| = 1`) per mode; the per-mode reader still accepts a v1 file's single vector as both, best effort — unreachable today behind the version gate below |
| 3 | 2026-05-21 | `fixed_atom_idxs` → `frozen_atom_idxs`; no backward compatibility |
| 4 | 2026-05-22 | `runtime_info` (CPU, threads, GPU, host) |
| 5 | 2026-08-20 | the optional `relaxation` and `thermo` blocks and `phase_relaxation` — additive, a v4 file reads whole |
| **6** | 2026-09-23 | `removed_motions`, `hessian_scope`, `n_atoms_in_hessian`, `hessian_density_fit`; the equilibrium block optional as a whole; `eigenvector_display` derived when a file carries only the canonical form — additive |

`READABLE_SCHEMA_VERSIONS = {4, 5, 6}`; an older or unknown version is refused
by name.

### 6.2 The keys, and who writes each

| key | written by | meaning |
|---|---|---|
| `schema_version` · `engine` · `engine_version` · `molbuilder_version` · `timestamp` | both | provenance; `engine` is `pyscf` or `siesta` |
| `structure_hash` | both, through one function (§ 4.3) | `sha256:` over `n_atoms`, the label and the input atom lines — a reader's provenance, not a gate |
| `n_atoms_total` · `free_atom_idxs` · `frozen_atom_idxs` | both | 0-based, in the **input** order (SIESTA: inverted through the record); the two lists partition `range(n_atoms_total)` and the reader refuses otherwise |
| `equilibrium.{scf_energy_eh, mo_energies_eh, homo_idx}` | PySCF | the reference SCF and its orbital ladder — the **energy subgroup** of the equilibrium block, which travels whole or not at all: **`null` in all three on SIESTA** (a periodic engine has a Fermi level, not a HOMO), and `null` in one slot only is a broken file. The two geometry keys below are separate and always present |
| `equilibrium.{elements, positions_ang}` | both | the geometry the Hessian is taken at (§ 4.2) — what the animation draws from |
| `modes[]` | both | ascending frequency; every entry a vibration (§ 6.3) |
| `removed_motions.{count, patterns}` | both | what the harmonic analysis removed before diagonalising: the count and the orthonormal Cartesian patterns over the free atoms; `len(modes) = 3·n_free − count` by construction |
| `hessian_scope` · `n_atoms_in_hessian` · `hessian_density_fit` | both | `free` (second derivatives for the free atoms only) or `all`; how many; whether the Hessian itself was density-fitted (`false` on the reduced route, `null` on SIESTA) |
| `ir_route` · `ir_fd_step_ang` · `raman_route` · `raman_fd_step_ang` | both | which route produced each strength and the step of a difference (§ 4.6); `none` when not computed; an older file reads `""` — absence of a record, never a claim (`raman_*`: § 10) |
| `phase_relaxation` · `phase_frequencies` · `phase_raman` · `phase_es` | both | `empty` · `running` · `complete` · `not requested` (§ 4.9); the SIESTA writer writes `complete` for the frequencies and `not requested` for the other three |
| `relaxation.{enabled, already_relaxed, n_steps, max_force_eh_bohr, max_force_all_atoms_eh_bohr, converged, warning}` | both | the tracked precondition; the judged force is over the free atoms, in Eh/Bohr (§ 4.3); SIESTA writes `enabled: false`, `already_relaxed` as the person's assertion (true once made — the gate refuses the run otherwise, § 5.8), the judged force and verdict of § 5.5, and the warning the viewer shows in the phase's row |
| `thermo` | both | `regime`, the headline (T, P) with `zpe_eh`, `h_eh`, `s_eh_k`, `g_eh`, `n_modes`, `n_imag_excluded`, `n_rigid_removed`, `note`, and `grid` (§ 4.7) |
| `selected_mode_idxs_1based` | PySCF | the modes that got the electronic-structure probe |
| `config` | both | what the description held (the PySCF config as a dict; on SIESTA the engine, kind and stage) |
| `methods_text` · `bibliography_keys` | both | the composed paragraph and the keys it cites (§ 4.10) |
| `engine_metadata` | both | engine-specific facts; SIESTA: `fc_file`, `fc_displacement_ang`, `fc_range_1based`, `fc_asymmetry_max_ev_ang2`, `reference_force_criterion_ev_ang` (§ 5.5) |
| `runtime_info` | PySCF | CPU, threads, GPU, host |

### 6.3 A mode

| key | meaning |
|---|---|
| `index_1based` · `frequency_cm1` | a negative wavenumber **is** an imaginary mode (a saddle, not a minimum), reported, never dropped; `has_imag` says so |
| `eigenvector_canonical` | `(n_free, 3)` Cartesian, `Σ m_k\|L_k\|² = 1` in amu — the science form: intensities, the physical amplitudes and the element shares pair with **this** one |
| `eigenvector_display` | the same, rescaled so its **largest absolute Cartesian component** is 1 (not the largest per-atom length: an atom moving off-axis swings up to √3 of it) — the animation's exaggerated form only; derived from the canonical form by the reader when absent |
| `ir_intensity_km_mol` · `raman_activity_a4_amu` | `null` when the channel was not computed (not requested, or the engine has none); **`0.0` is a measured absence**, a symmetry-forbidden band's residue |
| `electronic_structure` | the probe of § 4.8: `amplitude_ang` (the push `A`, kept with the probe so the coupling's denominator travels with its numerator — an older provenance table listed it as a top-level mode key; the code has always kept it here), the orbital windows at −A, 0, +A, the SCF energies, `homo_index_in_window`; `null` when the mode was not selected |
| `ir_active` · `raman_active` · `activity_class` | **derived at every serialisation, never stored** (§ 6.6) |
| `zero_point_amplitude_amu12_ang` · `zero_point_displacement_ang` | **derived at every serialisation, never stored** (§ 6.6): the zero-point amplitude of the mode, `Q_zp = √(ħ/2ω)` in amu^½·Å — `√(ZERO_POINT_Q2_AMU_ANG2_CM1 / ν̃)`, the constant 16.858 amu·Å²·cm⁻¹ derived in `constants.py` from the two constants the wavenumber conversion uses plus the Bohr radius — and the Cartesian displacement of every free atom at that amplitude, `Q_zp · L_canonical` in Å: the **mass-calibrated displacement** a vibration-coupled transport step moves the structure along (the thermal amplitude is this times `√coth(ħω/2k_BT)`, [`web/spectra.md`](?doc=web/spectra.md) § 4.1). `null` for an imaginary mode, which has no amplitude |

**The two normalisations must never be crossed** — the exaggerated amplitude
(Å) pairs with the display form, the physical amplitudes (√amu·Å) with the
canonical form, and an export records which pairing produced it
([`web/spectra.md`](?doc=web/spectra.md) § 4.1, [`web/vibrationview.md`](?doc=web/vibrationview.md) § 12.2).

### 6.4 Where every number comes from

The chain from the engine to the key, so a reader can see what molbuilder only
passes through and what it **derives** — the derived half is the only part its
tests can meaningfully guard:

| key | the engine reports | what molbuilder does | units |
|---|---|---|---|
| `equilibrium.scf_energy_eh` | `mf.kernel()`'s return | stored as is | Hartree |
| `equilibrium.mo_energies_eh` | `mf.mo_energy` | non-finite entries dropped (`_filter_finite`) | Hartree |
| `equilibrium.homo_idx` | `mf.mo_occ` | **derived** — `homo_index` (§ 4.3) | index |
| `equilibrium.positions_ang` | the relaxed (or asserted) geometry | the Hessian's geometry, `COORDS_EQ_ANG` | Å |
| `modes[].frequency_cm1` | the eigenvalues of § 4.5 | `sign(λ)·√\|λ\|·5140.487`; imaginary → negative wavenumber, never dropped | cm⁻¹ |
| `modes[].eigenvector_canonical` | the eigenvectors of § 4.5 | the canonical normalisation, from the one path on both engines (an older table called this column *dimensionless*; for `Σ m\|L\|² = 1` in amu the unit is 1/√amu) | 1/√amu |
| `modes[].eigenvector_display` | — | **derived** — per-mode rescale to `max\|L\| = 1` | dimensionless |
| `modes[].ir_intensity_km_mol` | `dμ/dR` (analytic response, or dipoles at displaced geometries) | **derived** — `dμ/dQ = Σ (dμ/dR)·L_canonical` (`einsum('kai,ka->i', DMU_DR, L_canonical)`), then `42.2561·\|dμ/dQ\|²` | km/mol |
| `modes[].raman_activity_a4_amu` | polarizabilities at displaced geometries, in Bohr³ | **derived** — central differences, the Placzek scalar, one global `(Bohr/Å)⁶ ≈ 0.02197` | Å⁴/amu |
| `modes[].zero_point_amplitude_amu12_ang` · `zero_point_displacement_ang` | the mode's frequency and canonical vector | **derived at serialisation** — `√(16.858 / ν̃)`, times `L_canonical` per free atom (§ 6.3) | amu^½·Å · Å |
| `relaxation.max_force_eh_bohr` (SIESTA) | the first `siesta: Atomic forces` block of the run's output, its FC step 0 | **read** by `summarize` — the largest over the free atoms, eV/Å → Eh/Bohr; `converged` against the description's own `relax_force_tol` (§ 5.5) | Eh/Bohr |
| `engine_metadata.fc_asymmetry_max_ev_ang2` (SIESTA) | the `.FC` file | **derived** — `max \|H_ij − H_ji\|` over the free block before symmetrisation (§ 5.5) | eV/Å² |
| `engine_metadata.reference_force_criterion_ev_ang` (SIESTA) | the description's `relax_force_tol`, resolved for the stage | **read** by `summarize` — the criterion the verdict used, so the verdict carries its provenance (§ 5.5) | eV/Å |
| `modes[].electronic_structure.amplitude_ang` | — | the push `A` of § 4.8, molbuilder's own choice (`displacement_amplitude_ang`), recorded with the probe it produced | Å |
| `modes[].electronic_structure.mo_energies_*_eh`, `scf_energy_*_eh` | `mf.mo_energy`, `E` at ±A | non-finite dropped; the shift and the coupling `ΔE/(2A)` are the viewer's arithmetic | Hartree |
| `removed_motions` | — | **derived** by the one rule (§ 4.5), on both engines | count · (count, n_free, 3) |
| `thermo` | PySCF's `thermo.thermo` (rrho) or the one home's vibrational sums | the deck computes, the viewer draws; the headline is a row of the grid (§ 4.7) | Eh, Eh/K |
| `relaxation.max_force_eh_bohr` (PySCF) | the nuclear gradient at the judged geometry | the largest force over the **free** atoms; the all-atom figure beside it | Eh/Bohr |
| SIESTA's `H_AA` | `.FC` rows in eV/Å² | mean of the two sides, symmetrised, converted (§ 5.5) | Hartree/Bohr² |

**Two rules this table enforces** *(2026-09-09)*: a number molbuilder only
passes through is not ours to test — a test asserting the magnitude of
`scf_energy_eh` asserts PySCF; what is ours is that it reaches the right key in
the right unit, unrounded. And a number molbuilder **derives** must have its
rule callable, not embedded in script text: `homo_index`, the harmonic path,
the thermo sums and the hash all ship as spliced source so one implementation
runs and is tested. The IR and Raman scalars are still inline, because neither
has a branch — the trigger `siesta/makov_payne.py` states: copy a branchless
formula if you must, ship the source once it has a branch; the same ruling
covers the sign convention for imaginary modes and the free/held index
bookkeeping, both branchless. If any of them grows a branch — a second
polarizability convention, a per-mode prefactor — it moves to a callable the
way `homo_index` did.

### 6.5 What a SIESTA file looks like beside a PySCF file

| | PySCF | SIESTA |
|---|---|---|
| `equilibrium.scf_energy_eh`, `mo_energies_eh`, `homo_idx` | numbers | **`null`, all three** |
| `modes[].ir_intensity_km_mol`, `raman_activity_a4_amu` | numbers, or `null` when not requested | **`null` on every mode** |
| `modes[].electronic_structure` | present on selected modes | `null` |
| `ir_route` / `raman_route` | `analytic` · `finite-difference` · `none` | `none` |
| `thermo.regime` | `rrho` or `vibrational-only` | `vibrational-only`, the vibrational contributions alone (no total energy is reported, so `h_eh` and `g_eh` are sums above an electronic minimum taken as zero) |
| `relaxation` | the tracked phase | `enabled: false`, *the force-constant route does not relax* |
| `hessian_density_fit` | `true` / `false` | `null` |
| `engine_metadata` | `{}` | `fc_file`, `fc_displacement_ang`, `fc_range_1based`, `fc_asymmetry_max_ev_ang2`, `reference_force_criterion_ev_ang` |
| beside the calculation | — | `atom-permutation.json` (§ 5.2) |

**A missing number is absent, never zero.** A key an engine cannot produce is
`null`, and a reader treats it as *not computed* — a different statement from
`0.0`, which is a measured absence (a symmetry-forbidden band). How each
surface draws an absent number is [`web/spectra.md`](?doc=web/spectra.md)
§ 9b.3: lines at the mode positions with no heights, a `—` in the table, the
`partial` colour on the rug, and a write-up that names what was computed.

### 6.6 Derived at every serialisation — the activity classes

`ir_active`, `raman_active` and `activity_class` are computed whenever a result
is serialised (`spectra/activity.py` through `SpectraResults._modes_with_activity`)
and never stored: whether a band is active is a **decision**, not a read,
because a symmetry-forbidden mode's stored strength is floating-point residue
(measured on CO₂: 4.8·10⁻⁹ to 7.5·10⁻⁸ beside bands of 32.85 and 613.04
km/mol). The rule: a mode whose channel was not computed is `partial`;
otherwise, per channel, divide every mode by the channel's strongest, sort on
a log scale, and cut at the widest gap between neighbours when that gap is at
least two decades wide and sits below a thousandth of the peak; when the
channel will not separate itself, cut at a millionth of the peak; and a
channel whose strongest value is under an absolute floor (10⁻³ km/mol,
10⁻³ Å⁴/amu) has no band at all. It is asked of the data because where the
residue sits is a property of the calculation (measured cuts 2.6·10⁻⁶ to
1.5·10⁻⁴ across four real runs) while the separation is a property of the
symmetry. Built 2026-09-11, confirmed as the rule 2026-09-23; pinned on CO₂ by
`tests/spectra/test_activity.py`. The load path adds one more derived field,
`motion_share_by_element` — each element's share of the mass-weighted motion,
`m_i|L_i|² / Σ m_k|L_k|²` — because the browser has no masses and the file
stores none ([`web/spectra.md`](?doc=web/spectra.md) § 4.2).
The zero-point amplitude and displacement of § 6.3 are derived at the same
moment, from the frequency and the canonical vector, for the same reason: a
file written before they existed gains them on read, and no writer can put a
second convention beside the first.

### 6.7 What the reader refuses

`SpectraResults` is built at the boundary and refuses, by name: a schema
version outside `{4, 5, 6}`; free and frozen lists that do not partition
`range(n_atoms_total)` (a count-only check once passed `free = [0, 1, 5]` for
three atoms and the viewer silently dropped a displacement); a mode whose
eigenvector does not carry one row per free atom; an equilibrium energy
subgroup (`scf_energy_eh`, `mo_energies_eh`, `homo_idx`) with `null` in some
slots and numbers in others; and — in the working tree, § 10 —
a key it does not know, at every block, because a misspelled
`ir_intesity_km_mol` used to serve a chart titled *not computed* with every
number present and thrown away, and a file claiming 10¹² atoms used to be
answered with a `MemoryError`. `/api/spectra/load` turns each refusal into a
typed error (missing → 404, wrong version → 422, malformed → 400) so the viewer
reacts without parsing a message.

---

## 7. The invariants — what the code must keep true

These are the statements a code review checks the implementation against.
Each names where it holds and what pins it.

| # | invariant | where it holds | pinned by |
|---|---|---|---|
| I1 | **One derivation of the surviving motions** (R1): `rigid_motions` is the only place `n_rigid` is computed; no call site tabulates it, branches on `len(F)`, or asks whether a molecule is straight | `spectra/normal_modes.py`; the deck splices it; `methods._mode_count`, the two preflights and the SIESTA read-back call it | `tests/spectra/test_normal_modes.py` (every row of the science § 7 table, three mutations each); review |
| I2 | **One harmonic path** (R3, R4): both engines hand `vibrational_modes` the block, the masses and the geometry; there is no free-molecule branch and no engine branch after the block | the PySCF deck (spliced source), `spectra/from_siesta.py` | the rank gate against PySCF on free molecules; the held-water and H₂ end-to-end runs |
| I3 | **One mass convention**: isotope-averaged masses, `Σ m\|L\|² = 1` in amu, one wavenumber constant derived from its parts | `chemistry.atomic_mass`, `constants.CM1_PER_SQRT_HARTREE_BOHR2_AMU`, the deck's `MASSES_AMU` | the BDT pair (C–H stretches equal to 0.001 cm⁻¹ free vs held); `tests/spectra/test_atom_index_contract.py` |
| I4 | **The Hessian is over the free atoms**, and the run says so (R8): `hessian_scope`, `n_atoms_in_hessian`, `hessian_density_fit` | `dipole_derivatives` (spliced); the FC range on SIESTA | the free-atom-block check in `tests/test_vibration_e2e.py`; `tests/test_siesta_vibration_deck.py` |
| I5 | **Stationarity is judged on the free atoms** (R5), and the number is recorded beside the all-atom one | `_vib_gradient_check`; the relax callback | the held-water run's `relaxation` block |
| I6 | **The held set has one source** — the structure's `frozen_atoms` region — and reaches every phase from it: the PySCF `$freeze` file, the free-atom Hessian, the SIESTA `Geometry.Constraints` and FC range | `VibrationConfigView.frozen_indices`; `fc_facts` | `tests/test_vibration_render_gate.py`; the wrapper's constraint banner reads the deck's one spelling |
| I7 | **The reorder is recorded and inverted once, through one pair**: `write_permutation` / `read_permutation`, `Permutation.rows_to_input_order`; the input order never reaches the engine, the sorted order never reaches a person | `jobset/prep.py`, `spectra/from_siesta.py` | `tests/test_siesta_vibration_e2e.py` (held atom last in the input → the free atom reported as atom 0, 0-based) |
| I8 | **One start state per kind on SIESTA**: the density read, the geometry declined out loud; the vibration warm-file section names `.FC`/`.FCC` inventory-only | `siesta/vibration_deck.start_state_lines`, `siesta/warm-files.toml` | `tests/test_siesta_vibration_deck.py`, `tests/test_warmfiles.py` |
| I9 | **Every mean field the PySCF deck builds is dressed by the one generated door** (`_mb_configure_scf`, `_mb_configure_dft`); no SCF knob is spelled twice | `pyscf/scf_setup.py`; every `_build_mf_at` | `tests/test_pyscf_spec.py`; the honesty gate |
| I10 | **Every parameter the form shows is honoured by the render, or refused by name** | `tests/test_vibration_form_honesty.py`; `validation/spectra.py` | the same test |
| I11 | **The kind's science gate fails closed**: an engine config the dispatch does not name is refused, never given an empty verdict | `validation/__init__._validate_vibration_kind` | `tests/test_vibration_render_gate.py` |
| I12 | **A cost claim is read from the code** (R8): the advisories say what the code skips, and what it cannot | `validation/spectra.py`; § 4.4, § 5.7 | review |
| I13 | **The Methods paragraph is composed once**, route-neutral before the run, the route added at load; the count it states is R2's (its results arm has no production caller — V1.18) | `spectra/methods.py`, `_loaded` in `web/blueprints/spectra.py` | `tests/spectra/test_methods.py`; the IR-only and solvated runs assert the text |
| I14 | **Absent is never zero** in the file, on both engines; the equilibrium energy subgroup travels whole or not at all | `SpectraResults.__post_init__`, `from_siesta` | `tests/spectra/test_types.py`, the SIESTA fixture test |
| I15 | **One hash**: `structure_hash_text` has one home and is spliced into the deck | `sidecars/spectra.py` | review (the SIESTA and PySCF files hash the same structure identically) |
| I16 | **The activity classes are derived at serialisation, never stored**; the element shares at load, never in the file | `results._modes_with_activity`, `_loaded` | `tests/spectra/test_activity.py`, `test_motion_share.py` |
| I17 | **Every catalogue citation resolves** in `science/references.bib`, and every key the prose cites is an entry | the catalogue-refs test | `tests/spectra/test_methods.py` |
| I18 | **No second producer**: a deck is written by `prep` from a description, through `spec_for`; there is no engine verb and the tab renders nothing | [`engines/pyscf.md`](?doc=engines/pyscf.md) § 1 | review |
| I19 | **The topics `frequency/` and `spectrum/` are a storage vocabulary** the person picks; nothing derives a folder from an engine or a kind | `projects.py` | review |
| I20 | **Every spliced function is self-contained**: it reads no module-level name, so a rendered deck parses and every free name in every spliced helper resolves (measured 2026-09-22: both decks `ast.parse` clean and run to exit 0; the left-behind constant of 2026-09-21 is the failure this guards) | `spectra/normal_modes.py`, `sidecars.spectra.structure_hash_text`, `dipole_derivatives`, `homo_index` | the render gate and the end-to-end runs |
| I21 | **The SIESTA keyword set is re-verified on any SIESTA upgrade** against the manual-derived table, never `strings` | § 5.1 | review, on every upgrade |

---

## 8. The pieces, and how the data flows

### 8.1 The file map

| file | role |
|---|---|
| `molbuilder/spectra/normal_modes.py` | `rigid_motions`, `vibrational_modes`, `vibrational_thermo`, `vibrational_thermo_grid`, `THERMO_GRID_K` — the one harmonic path and the thermo sums, self-contained so they travel into a deck as source |
| `molbuilder/spectra/results.py` | `SpectraResults`, `ModeData`, `ModeElectronicStructure`; the schema and its history; the reader's gates; the activity classes at serialisation; `motion_share_by_element` |
| `molbuilder/spectra/activity.py` | the active/inactive decision per channel (§ 6.6) |
| `molbuilder/spectra/selection.py` | the mode selectors and the window (§ 4.8) |
| `molbuilder/spectra/methods.py` | `render_methods_md`, `with_ir_route`, `extract_citation_keys`, `_mode_count` (§ 4.10) |
| `molbuilder/spectra/from_siesta.py` | the SIESTA read-back: `spectra_results_from_fc`, `siesta_methods_text` (§ 5.5) |
| `molbuilder/parse/engines/siesta_fc.py` | `read_fc`, `hessian_from_fc` — the `.FC` reader (§ 5.4) |
| `molbuilder/sidecars/spectra.py` | `dump_spectra_json`, `parse_spectra_json`, `structure_hash_text` |
| `molbuilder/pyscf/vibration_deck.py` | the PySCF deck composer, the `VibrationConfigView`, the relaxation / gradient / thermo / IR-only blocks, `vibration_stages` in `pyscf/stages.py` |
| `molbuilder/pyscf/vibration_emitters.py` | the emitted blocks: constants, the molecule, the equilibrium SCF, the Hessian, the Raman sweep, the IR projection, the electronic-structure loop, the Methods fragment; the spliced `homo_index` and `dipole_derivatives` |
| `molbuilder/pyscf/scf_setup.py` | the generated SCF and DFT dressers (I9) |
| `molbuilder/siesta/vibration_deck.py` · `siesta/input.py::spec_for` · `siesta/layout.py::FC_SECTION` · `siesta/warm-files.toml` | the SIESTA deck (§ 5.3) |
| `molbuilder/config/pyscf.py` · `config/siesta.py` · `data/catalogue.template.toml` | the fields and the catalogue rows of § 3 |
| `molbuilder/transport/sort.py` | `sort_by`, `SORT_KEYS`, `apply_order`, `Permutation`, `write_permutation`, `read_permutation` (§ 5.2) |
| `molbuilder/validation/spectra.py` · `validation/__init__.py` | the two kinds' checks and the dispatch (§ 3.3) |
| `molbuilder/jobset/prep.py` · `jobset/_cli.py` · `jobset/materialize.py` | the sort at prep; `init`'s kind gate; `summarize run` for SIESTA |
| `molbuilder/runwrap.py` | the run wrapper; its banner names the held atoms from the deck's one spelling |
| `molbuilder/web/blueprints/spectra.py` · `build.py` | the tab page, `/api/spectra/load` and `_loaded`; the schema, preflight and hand-over doors |
| `molbuilder/web/static/spectra/viewer.js` · `lib/spectra/core.js` · `lib/inspectors/spectra.js` · `lib/spectrumchart/` · `lib/vibrationview/` · `lib/task-handover.js` · `task-setup/viewer.js` | the tab, the shared engine, the presenter, the chart, the animation, the hand-over, the rung tab that prints the commands |

### 8.2 The data flow

```mermaid
flowchart TB
  subgraph describe["describe (browser or CLI)"]
    ST["structure + frozen_atoms region<br/>(.xyz + .molstruct.json)"]
    CAT["catalogue narrowed to (engine, vibration)<br/>→ the form · → &lt;label&gt;.template.toml"]
    TJ["task.json: calculation vibration,<br/>engine, the ladder (freq; relax + freq)"]
  end
  subgraph prep["prep run freq"]
    G["validate(struct, cfg, calculation)<br/>PySCF checks #124; SIESTA checks #124; refuse"]
    SORT["SIESTA: sort a copy held-first,<br/>write atom-permutation.json"]
    D["spec_for(…, calculation='vibration') → prepare_deck<br/>PySCF: the script · SIESTA: the .fdf"]
  end
  subgraph run["launch run freq"]
    PY["PySCF script: relax → SCF → Hessian(free) →<br/>modes → strengths → thermo → ES → .spectra.json"]
    SI["SIESTA: SCF + 6·n_free displaced SCFs → .FC"]
  end
  SUM["summarize run freq (SIESTA):<br/>read .FC → H_AA → vibrational_modes →<br/>invert the permutation → .spectra.json"]
  LOAD["POST /api/spectra/load → SpectraResults.from_dict<br/>+ with_ir_route + motion_share_by_element"]
  VIEW["Results tab: chart · table · animation ·<br/>electronic structure · thermochemistry"]
  ST --> G; CAT --> G; TJ --> G; G --> SORT --> D; G --> D
  D --> PY; D --> SI; SI --> SUM; PY --> LOAD; SUM --> LOAD; LOAD --> VIEW
```

### 8.3 The doors

| door | does |
|---|---|
| `GET /api/build/schema/<engine>?calculation=vibration` | the form schema from the catalogue narrowed to the kind |
| `POST /api/build/preflight` | the live checks: `validate(struct, cfg, calculation="vibration")` |
| `POST /api/structure/analyze` | auto-detect charge, spin and method for the loaded structure (engine-agnostic, translated per engine) |
| `POST /api/task-setup/handover` | render `<label>.template.toml` and `task.1st.json` for the kind; the browser writes them where the person chose |
| `POST /api/task-setup/save` · `/prep` | write `task.json`; run `prep` for one stage on a named machine |
| `POST /api/spectra/load` | parse an existing `.spectra.json` into display data; typed errors |
| `molbuilder jobset init / prep / launch / summarize` | the same road from a terminal (§ 2.1) |

---

## 9. Validation status, and the tests

**Validated, with numbers:**

- the rank rule against PySCF's own analysis on free molecules (§ 4.5);
- water with its oxygen held through the whole road: three modes, three motions
  removed, every mode orthogonal to every removed pattern in the mass metric;
- the free-atom Hessian against compute-everything-and-slice, Hartree–Fock
  and DFT, with and without dispersion (§ 4.4); the wavenumber constant
  against PySCF's derivation to 1·10⁻⁹, `Σ m\|L\|²` at 1.00000000 on every
  path, the dμ/dR tensor layout (8.4·10⁻⁵ as emitted against 5.2·10⁻¹
  transposed), the two analytic-route corrections at 6.7·10⁻¹⁶ (§ 4.5, § 4.6);
  both rendered decks parse with every spliced name resolved and run to exit 0
  (I20);
- BDT free against BDT with both sulfurs held at one geometry (§ 11;
  [`science/normal-modes.md`](?doc=science/normal-modes.md) § 9);
- infrared at band level: water at B3LYP/def2-SVP inside the literature windows
  with bend > asymmetric > symmetric; CO₂ reproducing mutual exclusion (the
  numbers are § 11);
- the `.FC` units and the `.FCC` shape on H₂ (§ 5.4); H₂ with one atom held on
  SIESTA through jobset (§ 5.5).
- **the convergence condition and the step, measured 2026-09-24** on H₂ with
  one atom held (GGA/DZP, the road end to end: an optimization calculation,
  its final frame exported from the Results tab, the vibration on the pair):

  | input geometry | largest force component on the **free** atom at the reference step (eV/Å) | δ (Å) | ω (cm⁻¹) |
  |---|---|---|---|
  | the experimental 0.741 Å, unrelaxed | 1.274 (1.274 on the held atom) | 0.0212 | 3358.0 |
  | relaxed to `MD.MaxForceTol` 0.02 eV/Å (0.7744 Å) | 0.0068 (0.011 on the held atom) | 0.0212 | 3024.4 |
  | relaxed to 0.001 eV/Å (0.7745 Å) | 0.00006 (0.004 on the held atom) | 0.0106 | 3012.1 |
  | the same | 0.00006 | 0.0212 (the default 0.04 Bohr) | 3022.3 |
  | the same | 0.00006 | 0.0423 | 3044.7 |

  The missing relaxation costs a tenth of the frequency; the tolerance, two
  wavenumbers (the held atom keeps its constraint force, which is not judged —
  R5). The one-sided constants differ by 2.4, 4.7 and 9.0 eV/Å², linearly in
  δ: the bond's cubic term, which the central difference cancels (every odd
  order; its leading error is O(δ²)). The frequency drift — 10 cm⁻¹ between
  the default and its half, 22 more at its double — is **not** that O(δ²)
  signature: a Morse estimate of the bond's quartic term gives a few
  wavenumbers with the wrong power of δ, so most of the drift is numerical, and
  the real-space grid (0.09 Å spacing against 0.01–0.04 Å nudges) is the usual
  suspect, not isolated here. So the plateau `ω(δ) ≈ ω(δ/2)` of
  [`science/normal-modes.md`](?doc=science/normal-modes.md) § 4b.6 C is not
  reached at the default, and a δ-only ladder cannot say why; V1.23 pairs a
  mesh rung with the δ rung. The block's asymmetry is 10⁻¹² eV/Å² throughout:
  on this axial block the off-diagonals vanish by symmetry, so it shows nothing
  here.

**Not done:** a mode-by-mode cross-check of intensities against an external
code (Gaussian, ORCA, Turbomole) — absolute intensities carry that caveat;
the reduced Hessian with a GPU mean field; any SIESTA run larger than two
atoms through the road (the reorder with a scattered held set is pinned by
the deck test, not by a run); and four of the design's five held systems
through the whole road — acetylene with both carbons held (the collinear
trap), NH₃ with its three hydrogens held (nothing removed), an empty held
list reproducing the free path *exactly*, and the water dimer with one
molecule held (the over-removal guard) — which exist today as rank rows
only (V1.17); a δ-convergence comparison that also varies the mesh (V1.23);
mode matching across runs (V1.24).

**The tests**, by what each proves:

| file | proves |
|---|---|
| `tests/spectra/test_normal_modes.py` | every row of the science acceptance table — the rank rule alone, no engine |
| `tests/test_vibration_e2e.py` | the rank gate against PySCF; the water loop (relaxation, three modes, thermo, the viewer loads it); IR alone in water's windows with the route recorded; the solvated chain; frequencies unmoved by asking for IR; water with O held; the free-atom block check |
| `tests/test_spectra_from_a_real_run_e2e.py` | CO₂ computed, then read back through the Results tab's own door — nothing faked |
| `tests/test_siesta_vibration_deck.py` · `tests/test_siesta_vibration_e2e.py` | the FC deck's lines and refusals; the record table of § 2.2 on the measured relaxation fixture (`tests/fixtures/siesta_relax`: a matching record's info line, a looser record's warning, another geometry, another level of theory or engine, the unticked offer to skip, no record accepted with a hint); the read-back on the measured fixtures — the modes in the input order, the reference forces judged both ways, the zero-point displacement derived and absent for an imaginary mode; the whole SIESTA road through jobset in both states of the box — unticked, `relax` then `freq` on the experimental bond, `freq` refused before `relax` has concluded and written at the relaxed geometry afterwards, the relaxed bond's frequency and the reference forces within the template's own tolerance; ticked, `freq` alone on the relaxed fixture, after the contradiction (unticked, no `relax` stage) is refused |
| `tests/test_vibration_render_gate.py` | the deck runs the science gate and refuses; an unknown engine class is refused |
| `tests/test_vibration_form_honesty.py` | every offered parameter changes the deck |
| `tests/spectra/test_types.py` · `test_parsers_json.py` · `test_atom_index_contract.py` | the artifact's gates and round trip; the free-atom invariant |
| `tests/spectra/test_activity.py` · `test_motion_share.py` · `test_selection.py` · `test_methods.py` · `test_config.py` · `test_blueprint.py` | the derived classes; the element shares; the reference selector and its parity with the deck's inlined copy; the prose, its citations and the provenance rows of § 6.4; the defaults; the page and the load door (the old generator's `test_engine.py` / `test_script.py` died at P3) |
| `tests/spectra/test_spectrumchart_*.py` · `tests/test_vibrationview_*_js.py` · `test_results_state_contract_spectra_js.py` · `test_spectra_phase_indicator_js.py` · `test_task_setup_tab.py` | the chart's maths, seal and box; the animation's maths and mount; the viewer's state; the phase indicator; the send flow |
| `tests/test_warmfiles.py` · `tests/validation/test_siesta.py` | the vibration warm-file section; the keyword table the deck is checked against |
| fixtures: `tests/fixtures/siesta_fc/` (the measured H₂ `.FC`, `.FCC`, `.fdf`), `tests/fixtures/psml/` | |

---

## 10. Shipped and owed

Every open row below is registered under **V1** in
[`plans/plan.md`](?doc=plans/plan.md); this table says what stands, the plan
says what is next. *Working tree* would mean edited but not yet verified or committed;
nothing is in that state as of 2026-09-24.

| | status | note |
|---|---|---|
| the one harmonic path, the rank rule, its gate | **built 2026-09-23** | § 4.5 |
| one mass convention | **built 2026-09-21** | the 1823× defect |
| stationarity on the free atoms | **built 2026-09-23** | § 4.3 |
| the runs write the structure pair | **built 2026-09-22** | § 4.2 |
| the Hessian over the free atoms, the two corrections, the scope in the file | **built 2026-09-23** | § 4.4 |
| the SIESTA arm: deck, sorted copy, record, `.FC` reader, read-back, warm-file section, start state | **built 2026-09-23 / 24** | § 5 |
| the equilibrium block optional; the display form derived | **built 2026-09-23** | schema 6 |
| the wrapper names the held atoms | **built 2026-09-23** | it read a comment no deck wrote, then counted the `0` of `0-based` |
| the thermochemistry's self-defence filter: the `> 0` line deleted, the imaginary exclusion stated as `n_imag_excluded` | **built 2026-09-23** | § 4.7 — the free energy no longer depends on the sign of noise |
| the three counting sites read the one derivation; `_is_linear` and `_mode_count`'s `n_free < 2` arm deleted | **built 2026-09-23** | § 4.10, I1 |
| the pre-run notice and the tab's prose say the relaxation holds the set *when it runs*, not unconditionally | **built 2026-09-24** | § 4.2; under `already_relaxed` there is no relaxation and no constraints file |
| the emitted deck's comment on the canonical normalisation says amu, not "atomic units" | **built 2026-09-24** | the phrase the 1823× confusion lived in |
| the geometry in the file is the Hessian's | **built 2026-09-24** | § 4.2; the water loop asserts the relaxed geometry differs from the input |
| the thermochemistry headline and grid as one quantity, the headline T on the grid, no `kT` in the held regime | **built 2026-09-24** | § 4.7; the free and the held water runs assert the headline equals its grid row |
| `raman_route`, `raman_fd_step_ang`; the Methods text states the Raman method one way | **built 2026-09-24** | § 4.6; the infrared-only and the solvated runs assert both |
| the reader's unknown-key gate; the partition check without a range the size of a lie | **built 2026-09-24** | § 6.7; two forward-compatibility tests that asserted the old rule are retired into the gate test |
| the Spectrum tab offers both engines — one strip over two catalogue-built forms, the structure choosing the default, the checks and the hand-over speaking the strip's engine; the hand-over gate admits SIESTA | **built 2026-09-24** | § 2.1, § 3.1; `web/spectra.md` § 5 |
| Task setup prints `--target this` for this machine whenever the CLI would refuse to guess, and `summarize run <stage>` as the last step of a SIESTA vibration | **built 2026-09-24** | `web/task-setup.md` § 11 |
| the Results viewer draws a SIESTA file: a `null` energy as a dash, the Raman and infrared lines by route, the fingerprint on the two routes, the relaxation line carrying a disabled phase's warning, the thermochemistry headline naming its regime | **built 2026-09-24** | § 6.5 |
| the force keys say their unit: `max_force_eh_bohr`, `max_force_all_atoms_eh_bohr`, and the viewer prints Eh/Bohr | **built 2026-09-24** | § 4.3 |
| `removed_motions` and `hessian_scope` shown beside the result | **built 2026-09-24** | R7's second half |
| the phase flags' *not requested* state: a phase the description never asked for is `not requested` from the first write, on both writers; the viewer counts it finished and draws it | **built 2026-09-24** — V1.5 | § 4.9 |
| `top_n` and `threshold` retired from the config, the catalogue, `selection.py`, `methods.py`, `validation/spectra.py`, the emitter's ranking and the tab's lock map | **owed** — decided 2026-09-23 | § 4.8 |
| `temperature_K` and `pressure_atm` reachable on SIESTA | **owed** | § 5.5 sums at 298.15 K, 1 atm and says so |
| the Methods paragraph reads the effective level of theory (`cfg.method`), and a functional or dispersion set under Hartree–Fock is advised against | **owed** | § 4.10 — a Hartree–Fock run's write-up names B3LYP-D3BJ |
| the structure identity: two hashes (geometry, broad), minted at the three gates and carried, the job name out of it | **owed** — ruled 2026-09-22 | § 4.3; the run-written pair already hashes its bytes |
| the pair writer renders both halves (`pair()` returns text, the deck splices the codec's own JSON) | **owed** — ruled 2026-09-23 | § 1.1a of the structure-API audit, `plans/2026-09-22-unification-audit.md`; the deck's sidecar writer is its third serialiser |
| the reduced Hessian with a GPU mean field | **untested** | § 4.4 |
| a composed permutation for a structure sorted for two reasons | **owed** — no caller yet | § 5.2 |
| `transport/compose.py` writes and reads its record through the one pair and stamps its key | **owed** | § 5.2, I7 |
| the transport connection's level two (the coupling from `FC.Save.dHS`); Born-charge infrared on SIESTA | **not in scope** — recorded so the design does not foreclose them; level one is V1.25 below | § 5.6 |
| an external mode-by-mode intensity cross-check | **not done** | § 9 |
| four held systems through the whole road (acetylene, NH₃, the empty held list, the water dimer) | **owed** — V1.17 | § 9 |
| `_mode_count`'s results arm: wire it into the load path or delete it | **owed** — V1.18 | § 4.10 |
| a release note: every held-atom spectrum and free energy computed before 2026-09-23 contains a non-vibration | **owed** — V1.16 | true and intended; the old runs disagree with the new ones |
| the presenter's category label names no engine (*Vibrational spectrum*); the Spectrum tab's engine sentence is the strip's | **built 2026-09-24** | |
| the Molbuilder tab's save prompt doubling a typed suffix (`x.xyz.xyz`); the `#`-label unconsumed warning (needs a ruling); the vacuum notice on a gas-phase PySCF run | **owed** — UI walk 2026-09-23 | not this kind's, recorded where found |
| the SIESTA route judges stationarity: the forces at FC step 0 read into `relaxation.max_force_eh_bohr` over the free atoms, `converged` against the description's own `relax_force_tol`, the warning printed and shown | **built 2026-09-24** — V1.21 | § 5.5; R5 on both routes |
| the asymmetry diagnostic `max \|H_ij − H_ji\|` recorded as `engine_metadata.fc_asymmetry_max_ev_ang2` | **built 2026-09-24** — V1.22; no warning threshold yet; on an axial block the off-diagonals vanish by symmetry, so H₂ shows nothing | § 5.5 |
| `already_relaxed` on both engines as the person's explicit say: unticked the tool relaxes first (PySCF Phase 0; SIESTA a `relax` stage, the geometry carried as coordinates), ticked it measures and warns plainly; the relaxation's convergence items on the vibration form with the kind's recommended tight values | **built 2026-09-24** | § 2.2, § 3.1, § 5.2a, § 5.8 |
| the mass-calibrated displacement per mode — `zero_point_amplitude_amu12_ang`, `zero_point_displacement_ang`, derived at serialisation | **built 2026-09-24** — V1.29 | § 6.3, § 6.6 |
| one stationarity rule for both routes: the largest absolute force component over the free atoms against the template's own force tolerance, a plain warning above it, on PySCF and SIESTA alike | **built 2026-09-24** — V1.30, by the ruling of § 2.2 | § 2.2, § 4.3, § 5.5 |
| the structure carries its relaxation record — `info.relaxation` (engine, the run's tolerance, the largest remaining force, the held set, the geometry's fingerprint) beside `info.calculation` (the level of theory) — recorded by the Results tab's structure inspector from the run directory, read by both kinds' gates on the box's card: absent and ticked accepted with a hint, present checked against this calculation's tolerance, level and held set | **built 2026-09-24** — V1.28 | § 2.2; `model/parse.md` § 5b.1 |
| the PySCF deck's own `_optimized.xyz` pair records `info.relaxation` (the deck knows its criteria, its convergence and its held set at the moment it writes; the spliced pair writer would carry the fingerprint) | **owed** — V1.31 | § 2.2 |
| a δ-convergence report: two stages at δ and δ/2 and a printed comparison of `ω_ν` and `e_ν` | **owed, needs a design** — V1.23 | `science/normal-modes.md` § 4b.6 C |
| mode matching across runs by eigenvector overlap in the shared free subspace (Models A/B/C; PySCF against SIESTA) | **owed, needs a design** — V1.24 | `science/normal-modes.md` § 4b.6 F |
| mode-displaced structure pairs on SIESTA at the zero-point and thermal amplitudes; the density-difference maps `Δρ_ν(r)`; the projected density of states along a mode | **not built, needs a decision** — V1.25, V1.26 | § 5.6 |
| the PySCF probe: five points, the coupling per zero-point amplitude `g_ν`, a molecule-projected window for a cluster | **owed, needs a decision** — V1.27 | § 4.8 |

---

## 11. Worked examples

**Water with its oxygen held — the demonstrator.** Six numbers come out of the
free atoms' 6×6 block. Three are the hydrogens swinging about the nailed-down
oxygen: measured before R3 at RHF/STO-3G they sat at 15.7, 19.4 and 23.7 cm⁻¹
with infrared intensities of 0.0, **62.5 and 151.5 km/mol** against 45.1 for
the real O–H stretch at 4054 cm⁻¹ (the others: 2088 at 6.0 and 4236 at 32.9)
— the two loudest bands of the "spectrum" were rotations, because turning a
polar molecule turns its dipole, and 15.7–23.7 cm⁻¹ is a real far-infrared
window where no threshold could have found them — and they carried
20.1 cal/mol/K of entropy, −6.0 kcal/mol in −TS.
After: `removed_motions.count = 3`, three modes (bend, symmetric and
asymmetric stretch), every one orthogonal to every removed pattern. It runs
in a second and half its raw output is not a vibration, which is why it is the
test and the teaching case.

**BDT free and with both sulfurs held — the real case.** One RHF/STO-3G
geometry, 14 atoms. Free: `3·14 − 6 = 36` modes. Held: one surviving turn
about the S···S line (the sulfurs sit *on* it), `3·12 − 1 = 35` modes. The
ring C–H stretches agree to 0.001 cm⁻¹ between the two runs; the S–H
stretches fall by `√(μ_free/μ_held) = 0.984643` (measured 0.984673 and
0.984664); the 36th number of the old two-branch code was the turn, at
−0.93 cm⁻¹ at the minimum and **96.78 cm⁻¹ off it**, in the middle of the real
vibrations, which is why it is removed before the solve and never spotted
after ([`science/normal-modes.md`](?doc=science/normal-modes.md) §§ 1, 4.1, 9).

**H₂ on SIESTA — the road.** § 1.4 and § 5.5: the held atom last in the
input, one mode, two motions removed, the free atom reported by its own
number.

**CO₂ — mutual exclusion.** A centrosymmetric molecule's infrared-active
modes are Raman-silent and vice versa. The run: 653.45 cm⁻¹ (×2) infrared
32.85 km/mol and Raman-silent; 1388.81 Raman 14.74 Å⁴/amu and infrared-silent;
2460.11 infrared 613.04 km/mol. The silent entries came back as 10⁻⁹ to 10⁻⁸
residue, which is what § 6.6's classifier exists for. With both oxygens held
the free carbon sits *on* the O···O line, the surviving turn moves nothing,
and `n_rigid = 0` — the trap a table of cases gets wrong.

**A molecule on a metal — the layered model, and the test the person owns.**
The discussion this work started from set out the model in three layers:

```text
             molecule                ← free
       Au atoms bonded to it         ← free (the "active" layer)
       deeper Au layers              ← held
       bulk-like Au                  ← held
```

Every Au atom stays in the quantum calculation; only the free ones enter the
Hessian. Holding the substrate makes it infinitely rigid, so the answer is
converged the way the discussion recommends: molecule only, then +1 active Au
layer, +2, +3, comparing the modes that are mostly molecular. The artifact
makes two such runs comparable by their own files — `free_atom_idxs`,
`hessian_scope`, `removed_motions` — and with three or more held atoms not on
a line nothing is removed. For a periodic slab the calculation goes to SIESTA
(§ 1.3), and the held-first sort makes any tick pattern a legal FC range
(§ 5.2).

**What the thermochemistry headline contains.** Free water, `regime = rrho`:
`G = E_elec + ZPE + H_thermal(trans + rot + vib) − T·S(trans + rot + vib)`, from
PySCF's `thermo.thermo`, and the same at every point of the curve. Water with
its oxygen held, `regime = vibrational-only`: `G = E_elec + ZPE + U_vib − T·S_vib`
— no translation, no rotation, no `pV`; the note in the file says so, and the
three removed motions carry no entropy because they are not in the sum.

---

## 12. Glossary

The physics terms — Hessian, partial Hessian, mass-weighting, stationary point,
constrained minimum, whole-body motion, rank — are
[`science/normal-modes.md`](?doc=science/normal-modes.md) § 10; SCF, DFT, CPHF
are the [`science/overview.md`](?doc=science/overview.md) glossary. The words
this document adds:

| term | meaning here |
|---|---|
| **kind** | which calculation a description asks for: `optimization`, `vibration`, `transport` |
| **description** | `task.json` plus `<label>.template.toml`: what is computed, on which engine, with which parameters and stages |
| **the pair** | a structure's `.xyz` and its `.molstruct.json` half, written together by the codec; the held set lives in the second |
| **hand-over** | the Spectrum tab rendering the template and `task.1st.json` for Task setup to finish |
| **deck** | the engine input `prep` writes: the PySCF script, the SIESTA `.fdf` |
| **dresser** | the generated function every mean field of the PySCF deck is configured by |
| **phase** | one of the four tracked steps of the PySCF script, with its flag in the file |
| **the sorted copy** · **the record** | the structure SIESTA is given, and `atom-permutation.json` that undoes it |
| **artifact** | `<label>.spectra.json` |
| **route** | which way a strength was obtained (`ir_route`, `raman_route`); and, in the titles of §§ 4–5, an engine's whole way to the block |
| **rung** · **ladder** · **tier** | one stage of a description, the list of them, and the coarse/medium/tight grades an optimisation's stages take; a vibration's ladder is one rung |
| **RRHO** | rigid-rotor harmonic-oscillator: the gas-phase thermochemistry of a free molecule — electronic, translational, rotational and vibrational sums |
| **`axis_kind`** | the structure's own statement, per axis, of `isolated`, `periodic` or `transport`; with the cell it decides which turns cost nothing |
| **Γ** | the centre of the Brillouin zone, `q = 0`: every repeated cell moving in phase — the only point this calculation computes at |
| **the spectra migration** | the plan of 2026-08-20 that made a spectrum an ordinary described job, in phases P0–P3 ([`archive/2026-08-20-spectra-migration-plan.md`](?doc=archive/2026-08-20-spectra-migration-plan.md)) |

---

## 13. References

Cited by key; entries and their verification records are in
[`science/references.bib`](?doc=science/references.bib), the science
document's § 11 annotates the ones the rules rest on. Used here:
[Head1997], [LiJensen2002], [Besley2008], [Ghysels2008], [Vester2024],
[Tao2021], [QChemPHVA], [ASE2017], [Ghysels2010], [Wilson1955],
[Komornicki1979] (dipole and polarizability derivatives), [Grimme2011]
(the D3 dispersion correction), [Sun2018], [Sun2020] (PySCF),
[Galperin2007], [Frederiksen2007] (vibrations in molecular junctions and
inelastic transport, the transport connection of § 5.6).
