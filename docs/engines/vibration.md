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
[`plans/plan.md`](?doc=plans/plan.md), the one list; § 10 says what stands
and points there.

> **Consolidated 2026-09-24** from the design of 2026-09-21, the follow-up
> audit of 2026-09-22, the UI walk of 2026-09-23 and the implementation
> sections that had accumulated in the science and web contracts. The design
> and the audit are archived (`archive/2026-09-24-normal-mode-unification-design.md`,
> `archive/2026-09-24-vibration-audit.md`) as the record of how the decisions
> were reached; every decision, measurement and open item they held is here
> or in the plan. Nothing decided there is re-decided here. The spectrum rows
> of the 2026-09-23 UI walk are registered under V1 in `plans/plan.md`. One
> ordering rule
> of the design was knowingly broken: its step 4 (the API shape) was to land
> before the SIESTA arm, and part of it landed after — the cost is the second
> pass V1.0 and V1.6 name.

---

## 0. How to read this document

Three documents share this subject, and each owns one kind of statement:

| document | owns | read it when you want to know |
|---|---|---|
| [`science/normal-modes.md`](?doc=science/normal-modes.md) | **why** — the derivations, the rank rule for the motions that are not vibrations, stationarity, masses, the rules R1–R8 and the acceptance test | why a held-atom run reports fewer modes than $3\,N_{\mathrm{free}}$, and why that number is a rank and not a table; and, in its § 4b, both engines' workflows step by step against the discussion this work started from — what the tool does at each step, where it differs, and what is owed |
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

Near a geometry $R_0$, write every atom's displacement as one long vector $u$
(three numbers per atom). The energy is a quadratic bowl and the force is minus
its gradient — the many-coordinate form of Hooke's law:

$$
E(u) \approx E_0 + \tfrac12\,u^{\mathsf T} H\,u ,
\qquad
H_{I\alpha,J\beta} = \frac{\partial^2 E}{\partial R_{I\alpha}\,\partial R_{J\beta}}
\quad \text{(the Hessian)}
$$

$$
F = -\nabla E = -H\,u
$$

Moving one atom pushes on the others: that coupling IS the off-diagonal
Hessian.

Divide each entry by the square roots of the two atoms' masses and diagonalise:
each eigenvector is a **normal mode** — a pattern of atoms moving together at
one frequency $\omega = \sqrt{\lambda}$ — and the spectrum is the list of them. Six of the
eigenvectors of a free molecule are not vibrations (three slides, three turns;
five for a straight molecule), so they are removed before the diagonalisation.
Two more quantities ride on the modes: how strongly each absorbs or scatters
light (the **infrared** and **Raman** strengths, from how the dipole and the
polarizability change along the mode), and the harmonic **thermochemistry**
(zero-point energy, entropy, free energy) summed over them.

### 1.2 What holding atoms means

Hold some atoms — an anchor, a metal slab — and split the coordinates into free
$A$ and held $F$. Impose $u_F = 0$:

$$
\begin{pmatrix} F_A \\ F_F \end{pmatrix}
= -\begin{pmatrix} H_{AA} & H_{AF} \\ H_{FA} & H_{FF} \end{pmatrix}
\begin{pmatrix} u_A \\ 0 \end{pmatrix}
\quad\Longrightarrow\quad
F_A = -H_{AA}\,u_A , \qquad F_F = -H_{FA}\,u_A \neq 0
$$

Four consequences, each a rule in the science document:

1. **What is diagonalised is $H_{AA}$** — the free–free block of the Hessian of
   the *whole* system, with every held atom present in the energy. A held atom's
   displacement is zero; its interaction is not. This is the partial Hessian
   (PHVA) of Head, Li & Jensen and Besley [Head1997, LiJensen2002, Besley2008].
2. **Some eigenvectors of $H_{AA}$ are still not vibrations.** The free atoms can
   turn about one held atom, or about the line through two, at no cost. How
   many such motions survive is a **rank** computed from the geometry
   ([`science/normal-modes.md`](?doc=science/normal-modes.md) § 3.1 — six, five,
   three, one or zero, with two traps a table gets wrong), and they are
   removed **before** diagonalising (R3). With three held atoms not on one line,
   or any held atom in a periodic cell, nothing survives and nothing is removed.
3. **Stationarity is asked of the free atoms only** (R5): $\nabla_A E = 0$. The
   held atoms carry the constraint force $-H_{FA}\,u_A$ by definition.
4. **One mass convention on every route** (§ 3.2 there): isotope-averaged
   masses (H = 1.008, not 1 — whole mass numbers move a stretch by 15 cm⁻¹),
   modes normalised $\sum_k m_k \lvert L_k\rvert^2 = 1$ in amu. Crossing the normalisations
   once put every infrared intensity out by 1823×.

**The eight rules of the science contract, one line each** — cited by number
throughout this document; the full statements and their reasons are
[`science/normal-modes.md`](?doc=science/normal-modes.md) § 6:

| rule | in one line |
|---|---|
| **R1** | the count of surviving whole-body motions is computed in one place, by a rank — never tabulated or branched on |
| **R2** | one formula for the mode count, $3\,N_{\mathrm{free}} - n_{\mathrm{rigid}}$, for every system; $3N - 6$ is the empty-held-set case |
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
| how it gets $H_{AA}$ | **analytic** second derivatives, for the free atoms (§ 4.4) | **central differences** of its analytic forces: nudge each free atom, read every force (§ 5) |
| frequencies + mode shapes | yes | yes |
| infrared / Raman strengths | yes (§ 4.6) | infrared from the run's charges, for the atoms the person names (§ 5.10 ②); Raman from the run's optical phase (§ 5.11), at a laser the person names after the run (§ 5.11 § 10) |
| thermochemistry | full RRHO (rigid-rotor harmonic-oscillator) for a free molecule; vibrational sums with atoms held | vibrational sums |
| per-mode electronic structure | yes (§ 4.8) | no |
| a gold electrode | must be faked as a finite cluster | its natural home — and the same pseudopotentials, orbitals and k-points as the transport step (§ 5.6) |

Only *how the block is obtained* differs. Everything after it is one function,
`spectra/normal_modes.py::vibrational_modes`, which both engines hand the same
things — the block, the masses, the geometry with its held set, and which
axes repeat (`axis_kind`, with the cell when one does) — and which returns the
modes with the surviving whole-body motions removed. The
PySCF deck imports it from `mb_pyscf.pyz`; the
SIESTA job's finish calls it beside the run (§ 5.5). There is no second path (R1–R4).

### 1.4 The smallest example, with real numbers

H₂ measured on a two-atom SIESTA force-constant run (2026-09-23): the bond's
force constant read from the `.FC` file is $k = 41.713\ \mathrm{eV}/\text{Å}^2$, $m = 1.008\ \mathrm{amu}$.

| | diagonalised | motions removed | modes | $\omega$ |
|---|---|---|---|---|
| both atoms free | the 6×6 block | 5 (three slides, two turns; the turn about the bond moves nothing) | 1 | $\sqrt{2k/m}$ = **4744 cm⁻¹** |
| one atom held | the free atom's 3×3 block, `[k]` along the bond | 2 (the free atom swinging about the held one) | 1 | $\sqrt{k/m}$ = **3355 cm⁻¹** |

*(The 2026-09-23 run of § 5.5, on the same unrelaxed bond, reported 3358 cm⁻¹:
the same mode from its own SCF and its own force constant, the fixture's from
a hand-set deck — one quantity, two SCF settings. On the relaxed bond the road
reports 3022, § 9.)* Holding one end lowers the stretch by exactly $\sqrt{2}$, because
the partner no longer recoils: the reduced mass is $m$ instead of `m/2`. Both numbers are
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
  M["Molbuilder tab<br/>build the structure,<br/>hold atoms in the viewer,<br/>Save to project (the pair)"] --> S["Spectrum tab<br/>load the structure, set what the calculation gives<br/>(once, for both engines), pick the engine on the strip<br/>(a periodic structure defaults to SIESTA) and set its own settings,<br/>read the live checks"]
  S -->|"Send to Task setup<br/>= the hand-over"| T["Task setup<br/>shape, machine, the kind's ladder;<br/>Save writes task.json and job-request.json;<br/>the stage's tab prints the commands<br/>(--target when the CLI would refuse to guess)"]
  T -->|"prep task"| P["the deck<br/>PySCF: &lt;label&gt;_01_freq.py<br/>SIESTA, from a sorted copy: one stage, &lt;label&gt;_01_freq,<br/>its decks one per phase -- relax (box unticked), fc, optical (Compute Raman on)"]
  P -->|"launch task"| R["the run — the job writes the result itself<br/>PySCF: the script writes it<br/>SIESTA: the force-constant run leaves &lt;label&gt;.FC,<br/>then the finish (mb_vibration.pyz) derives the modes"]
  R --> A["&lt;label&gt;.spectra.json<br/>the one artifact, schema 13,<br/>its data file .spectra.nc beside it"]
  A --> V["Results tab<br/>chart · modes table · animation ·<br/>electronic structure · thermochemistry"]
```

The CLI walks the same road without the browser:

```bash
molbuilder describe --structure P/structure/x.xyz --bundle P/spectrum/F \
    --engine pyscf|siesta --calculation vibration --name X --shape hierarchical
# SIESTA: its ranks and the cores per rank
molbuilder jobset prep task --bundle P/spectrum/F --target this \
    --np 2 --cpus-per-task 1
# PySCF: its threads alone -- a rank count (--np) is refused
molbuilder jobset prep task --bundle P/spectrum/F --target this \
    --cpus-per-task 2
molbuilder jobset launch task --bundle P/spectrum/F --mode direct --yes
```

`prep` states the launch shape — SIESTA's ranks and cores per rank, PySCF's
threads — on its line or in the job request's run card, or it refuses
(`placement.launch_refusal`, which also refuses a rank count for PySCF, an
engine that runs one process). On SIESTA the one stage `freq` is one run in phases (§ 5.11): with the box
unticked its relaxation runs first, inside the run — one prep, one launch.
The measuring run — PySCF's one, SIESTA's `freq` — ends with
`<label>.spectra.json` in its attempt (§ 5.5 for SIESTA) — the launch, started in the background,
returns at once, and `jobset status` follows it; there is no step after it.

`molbuilder describe` refuses `--stage-strategy` for this kind (a *ladder* is a
description's list of stages, each a *rung* with its own parameter set; a
*tier* ladder — coarse, medium, tight — grades an optimisation's convergence,
and a vibration's ladder is its measurement, with the relaxation before it on
SIESTA when the box is unticked and, for a displacement sweep, more
force-constant stages, § 5.9) and any engine but the two named.

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
| **unticked** (the default) | the calculation **relaxes first**, at the template's convergence settings — recommended tight, a largest remaining force of about 0.01 eV/Å — and then takes the second derivatives at the relaxed geometry. On PySCF that is Phase 0 of the one script (§ 4.2); on SIESTA it is the **`relax` phase of the one run** (§ 5.11 § 1), the relaxed geometry written beside the force-constant call's deck as its coordinates by the run's own plan step (§ 5.11 § 3) |
| **ticked** | the person states the structure is already relaxed **at this level of theory**. Nothing is relaxed; the tool measures the forces at the starting geometry and, above the template's force tolerance, **warns in plain words that the frequencies will be off** — never refuses, because the statement is the person's to make. The hint beside the box says the same before the run is paid for. **And the structure keeps the engine's frame** when it was read back from a run: it is used as the force minimum on the grid it was found on, so it enters this calculation unmoved, where every other new calculation is centred ([`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 6.0, *Entering a new calculation*; user, 2026-10-11) |

Why the warning is plain and the default is to relax *(measured 2026-09-24,
§ 9)*: the H₂ fixture at the experimental bond length, sent through the
SIESTA road unrelaxed, reported 3358 cm⁻¹ where the relaxed bond gives 3022 —
a tenth of the frequency — and the two relaxation tolerances 0.02 and
0.001 eV/Å differ by two wavenumbers. An unrelaxed structure is the one error
that looks like a result.

**How the tool answers the ticked box with numbers**: PySCF's deck checks
the gradient at the input geometry (§ 4.3); SIESTA's finish reads the
forces SIESTA evaluated at its FC step 0 (§ 5.5). Both judge the largest
absolute force component over the **free** atoms (R5) against the template's
own force tolerance — `geom_gmax` on PySCF, `relax_force_tol` on SIESTA — the
one the person set or left at the kind's recommendation; both write the
number and the verdict into the result, and the viewer shows them; above the
tolerance, the remedy is one text on both engines (§ 5.5).

**The structure can carry its own evidence** *(user direction 2026-09-24:
"if the info meta data is present, we should display it and check against
tolerance of calculation; if not present … we will just accept that but with
a warning/hint")*. A pair exported from the Results tab of a finished
relaxation carries `info.relaxation` — the run's engine, its force tolerance,
the largest force left on the atoms it moved, the atoms it held, and a
fingerprint of the geometry it describes
([`model/parse.md` § 5b.1](?doc=model/parse.md)) — beside `info.calculation`,
the level of theory the deck stated (on SIESTA). Both kinds' gates read
them (`validation/sidecar.py::check_relaxation_record`), and every finding
lands on the box's own card, so the record is displayed where the choice is
made; the Metadata pane shows the raw store.

| the box | the record | the finding |
|---|---|---|
| ticked | absent | the plain warning of the table above, and an info line that no record travels with this structure — the statement stands on its own and the finish measures it |
| ticked | present, for a different geometry (another frame of that run, or edited since — the fingerprint differs) | **warning**: the record does not vouch for these coordinates |
| any | edited by hand (`edited_by_hand`, a person's change on MolView's Metadata page — [`web/molview.md`](?doc=web/molview.md) § 8.4a) | an info line first: the record's values are the person's word, not the run's — then the findings below, read from those values |
| ticked | present, for these coordinates | the engine, the tolerance, the steps and the largest remaining force are shown; a different engine, a different level of theory (`info.calculation` against this form's basis, functional, mesh cutoff, electronic temperature, the k-point grid its deck will write, and the charge and spin it will carry, resolved), a largest force above **this calculation's** tolerance, or an atom the relaxation held left free here (§ 2.3a) is each a **warning** naming the number, the field or the atoms; within tolerance at the same level is an info line, and so is holding more atoms than the relaxation held (§ 2.3a) |
| unticked | present, for these coordinates, within this calculation's tolerance at the same level, holding every atom the relaxation held | an info line: the record already meets this calculation's criterion, so the box may be ticked and the relaxation skipped |
| unticked | anything else | the same facts as information; the calculation relaxes first regardless — the SIESTA run's `relax` phase, the PySCF deck's own Phase 0 |

Never a refusal: the record informs the person's explicit choice; it does not
make it. The PySCF deck's own `_optimized.xyz` pair records neither: the
relaxation record is the one reader's, written by the Results tab's export
(V1.31, closed 2026-09-29), and a PySCF run's level of theory has no reader yet
([`pyscf.md`](?doc=engines/pyscf.md) § 2; plan § 5w K18).

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
purpose — the one finding about it is a fact the structure's own relaxation
record shows against it, an atom the relaxation held that this calculation
frees (§ 2.3a). What the run **says** about the choice is the point — the regime, the
motions removed, the Methods sentence (§ 4.10, § 6).

### 2.3a Holding more for the spectrum than for the relaxation — the label, and the record *(user, 2026-10-14)*

**The common road.** A junction is relaxed with its electrode blocks held and
every atom between them free; its relaxed geometry is exported from the
Results tab; and before the spectrum the person holds **more** — everything but
the molecule in the gap and a few surface metal atoms — and describes the
vibration with the box ticked. The Hessian is then taken over the smaller
region, with the rest fixed: a partial Hessian
([`science/normal-modes.md`](?doc=science/normal-modes.md) § 1). Holding more
is a choice with a cost the person weighs: the clamped neighbours stiffen the
contact modes ([`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 4c.8).

**Two stores, and neither is the other.**

| | where | written by | changed by |
|---|---|---|---|
| **`frozen_atoms`** — the atoms **this calculation** holds | the structure's `regions`, the one label store ([`model/structure-annotations.md`](?doc=model/structure-annotations.md)); every label lives there, and nowhere else | the person, in the viewer | the person, any time — it is what each calculation reads |
| **`held_atom_keys`** — the atoms **the relaxation** held | the structure's `info.relaxation` record ([`model/parse.md`](?doc=model/parse.md) § 5b.1), beside the relaxation's steps, forces and verdict | the relaxation run, read back from its output when the structure is exported from it | nobody — a person may edit the record on MolView's Metadata page, which stamps it `edited_by_hand`, and the check then says whose word it is |

The record is history, not a label: nothing is held because of it. It names
the atoms by element and position (`Structure.geometry_lines`), never by an
index, so a reordering does not break it; changing the labels does not change
it; and its geometry fingerprint says whether the coordinates are still the
ones it describes (§ 2.2).

**The comparison** — at every vibration's gate (`check_relaxation_record`):

| the held sets | the finding |
|---|---|
| the same atoms | none |
| every atom the relaxation held is held here, and more | **information**: every atom this calculation moves was free, and balanced, in the relaxation — the partial Hessian over fewer atoms |
| an atom the relaxation held is **free** here | a **warning** when the box is ticked, naming the atoms — their forces were never balanced, so the geometry is not stationary for them; information when unticked, since the ladder relaxes this set first |

The last row is not on the common road. It is an atom the relaxation held let
go for the spectrum — the electrode's top layer, say, the gold the sulfur
binds to — or a structure relaxed elsewhere with a larger held set. Holding
more never asks for it: **the force judgment stays valid**, because the
record's largest force over the relaxation's free atoms bounds the largest
over this calculation's fewer free atoms. A structure with no relaxation
record has nothing to compare, and nothing is said.

**The result states what this calculation held** (§ 6.10.1): its
structure's `frozen_atoms` is the set the Hessian held. The record of what
the earlier relaxation held stays with the structure it came in on — the
result's `info` keeps none of the input's run records — so the comparison is
said at the gate, above, and only there. A SIESTA run's own `relax` call
rides the result whole (`relaxation.record`, § 6.2), and it held this
calculation's set (below). A transport calculation built from the result
needs only every electrode atom among the held ([`engines/transport.md`](?doc=engines/transport.md)
§ 1.1, Step 0, *what transport asks of the held set*).

**Within one calculation the held set is one.** Every stage of a ladder is
prepared from the calculation's one structure, which its description pins
(its fingerprint, [`engines/stages.md`](?doc=engines/stages.md) § 6.3): a
structure edited after it was described is refused at prep, so `relax` and
`freq` hold the same atoms, their decks list them in one order, and `relax`'s
density is always in `freq`'s order (*user, 2026-10-14*).

### 2.4 Where results land — `spectrum/`

A vibration's topic is `spectrum/` — the kind table's `topic` for it, so
every new project lays the folder down
([`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 2.5) —
whichever engine runs it and whichever results it asks: frequencies and
thermochemistry alone land there as infrared and Raman do. A person may
still put one under another topic; nothing moves or refuses a calculation by
its folder. There is no `frequency` calculation kind and no `frequency/`
topic.

---

## 3. The parameters

### 3.1 One catalogue, narrowed to the kind

Every parameter of every calculation lives in the one catalogue
(`molbuilder/data/catalogue.template.toml`;
[`engines/template.md`](?doc=engines/template.md)). An item declares which
engines it applies to (`engines`, absent = all) and which calculation kinds
select it (`calculations`, absent = every kind — § 6.3 there). The vibration
template, the Spectrum tab's form (`GET /api/build/schema/common?calculation=vibration&surface=state`,
the charge and spin both engines share, `GET /api/build/schema/common?calculation=vibration`,
the other settings they share, and `GET /api/build/schema/<engine>?calculation=vibration&surface=own`,
each engine's own) and the deck's own item lines are all the same narrowing of the same file, so
a parameter is defined once and rendered the same everywhere.

**Three kinds of item, and how each is shown** *(user, 2026-09-28: "there are
parameters that are only available for pyscf, and parameters that are engine
neutral. so you need to make this correct"; "we know that there are limits of
all engines, and the options and presentation just need to be clear about
that")*:

| kind | the items | on the forms | in the result |
|---|---|---|---|
| **one meaning on both engines** | `already_relaxed`, `compute_ir`, `compute_raman`, `raman_modes`, `raman_freq_min_cm1`, `raman_freq_max_cm1`, `net_charge`, `spin_treatment`, `unpaired_electrons`, `temperature_K` | drawn **once**, before the engine is chosen — the charge and spin in the chemistry card, the rest under *The calculation* ([`engines/template.md`](?doc=engines/template.md) § 6.2a); one default on both engines; both routes honour each — *Raman for* is SIESTA's to apply, and on PySCF, whose sweep gives every mode at once, the settings gate and the deck say a range narrows nothing | `thermo.temperature_K` and `relaxation.already_relaxed` in both results; the routes a result took (`ir_route`, `raman_route`); the charge and spin as resolved in PySCF's `electronic_state` block — a SIESTA result does not record its deck's state |
| **one idea, each engine's own setting** | the relaxation's criteria (`geom_*` on PySCF, `relax_*` on SIESTA — two optimisers), the SCF machinery, the basis, the functional | each form shows its own | each engine's own record |
| **a capability one route has** | PySCF's probe — `es_n_homo_below`, `es_n_lumo_above`, `displacement_amplitude_ang` — and the gas-phase pressure `pressure_atm`; SIESTA's `fc_displacement` and its optical items (`optical_*`, § 5.11 § 9). **The probe's mode selector** — `es_mode_selection`, `es_explicit_indices`, `freq_min_cm1`, `freq_max_cm1` — is PySCF's alone: the modes the orbital probe looks at, not tied to Raman — one meaning per setting *(the user, 2026-10-27)* | only that engine's form | the other route's file has no switch for it in its `config`, so the viewer reads the absence **by role** and says *not computed on this route* — it never asks a person to set an item that route does not have ([`web/spectra.md`](?doc=web/spectra.md) § 9b.3) |

**The vibration-only items** *(the catalogue as of 2.45.0)*:

| item | engine | what it reaches | default | note |
|---|---|---|---|---|
| `already_relaxed` | both | the person's explicit say (§ 2.2). Unticked: PySCF runs Phase 0, SIESTA's run relaxes first, its `relax` call (§ 5.11). Ticked: nothing is relaxed; the forces at the starting geometry are measured against the template's force tolerance and a plain warning says the frequencies will be off when they fail | `false` | never refused; the hint beside the box carries the same warning |
| `compute_raman` | both | PySCF: `COMPUTE_RAMAN` — the polarizability sweep (§ 4.6), every mode at once. SIESTA: the optical phase — SIESTA's optical calculation at the equilibrium and at $\pm\delta Q$ along each mode *Raman for* picks (§ 5.11 § 9) | `false` | the expensive one: on PySCF about $6\,N_{\mathrm{free}}$ extra SCFs, each with a response calculation; on SIESTA $1 + 2N$ single points with their optical calculation, $N$ the modes picked |
| `raman_modes` | both | *Raman for*: `all` every mode, `range` the modes whose frequency lies in `raman_freq_min_cm1`–`raman_freq_max_cm1` — SIESTA's optical calls (`select_modes`); on PySCF nothing, its sweep gives every mode | `all` | live while `compute_raman` is on (`active_when`); a range is the one choice that can be made before the run, since a chemist knows where the bands are — no list of mode numbers, before there are modes to number. `range` with neither bound is refused at prep, a range stated nowhere; on PySCF `range` is said to narrow nothing |
| `raman_freq_min_cm1` · `raman_freq_max_cm1` | both | the range `range` keeps, either bound blank for none | unset | live under `range` with Raman on; with Raman on and a range asked, a minimum above the maximum is refused at prep — two ends that conflict |
| `compute_ir` | both | PySCF: `COMPUTE_IR` — dipole derivatives (§ 4.6). SIESTA: every atom's Hirshfeld and Voronoi charge at every force-constant step (`Charge.Hirshfeld`, `Charge.Voronoi`), from which the infrared of the atoms a person names is derived after the run (§ 5.10 ②) | `true` | on PySCF nearly free when it is the only strength asked for and no atom is held, otherwise its own dipole sweep or the Raman sweep's; on SIESTA a grid integration per step, no extra SCF |
| `temperature_K` | both | the thermochemistry's temperature (§ 4.7): `THERMO_T_K` in the PySCF deck; on SIESTA the deck's `vibration` block, which the finish sums at (§ 5.3, § 5.5) | 298.15 K | on SIESTA since 2026-09-28 (V1.7); greater than 0 by rule, on both engines — PySCF's thermochemistry divides by kT, and what 0 K would leave on the harmonic sums, the zero-point energy, every record carries at any temperature (`template.md` § 5.3) |
| `pressure_atm` | pyscf | `THERMO_P_ATM` — the pressure of the gas-phase translational term, which only the full RRHO regime has (PySCF, nothing held; § 4.7) | 1 atm | not on SIESTA's form: no SIESTA result has a translational term for it to enter. With atoms held it enters nothing either, and a value other than the default is **warned about** — kept on the form, so the form's shape does not follow the structure *(user, 2026-09-28)*; greater than 0 by rule — its entropy takes a logarithm of kT/P (`template.md` § 5.3) |
| `displacement_amplitude_ang` | pyscf | `DISPLACEMENT_AMPLITUDE_ANG` — the ± push along a mode for the electronic-structure probe (§ 4.8) | 0.02 Å | window 0.02–0.20 Å: smaller drowns in SCF noise, larger leaves the harmonic region |
| `es_mode_selection` | pyscf | `ES_MODE_SELECTION` — which modes get the probe (§ 4.8) | `skip` | three choices — `skip` · `all` · `explicit`; `top_n` and `threshold` were retired by decision (2026-09-23) and removed (2026-09-28, V1.6) |
| `es_explicit_indices` | pyscf | the list for `explicit` — mode numbers from an earlier run of the same structure | `""` | 1-based, `"3, 7, 12"` or `"3-7, 12"`; one format, read by one reader, `PySCFConfig.explicit_modes` — the atom index list's grammar (`selection.parse_index_list`), sorted, each mode once — for the deck's constant, the Methods count and the selector; text it cannot read (`"0, 2"`, `"3,,4"`) is refused at `prep`, before a deck is written |
| `freq_min_cm1` · `freq_max_cm1` | pyscf | the window `all` filters by | unset | `skip` selects nothing; ignored by `explicit` — naming a mode is saying *that one* — so the form **greys out** the window outside `all`, as it greys out the explicit list outside `explicit` — each field's `active_when` *(user, 2026-09-28)* |
| `es_n_homo_below` · `es_n_lumo_above` | pyscf | the orbital window recorded per displaced geometry | 5 · 5 | record size, not cost |
| `net_charge` | both | `NetCharge` / `gto.M(charge=)` | auto | shared with every kind; resolved once by the electronic state (a stated value wins, 0 included; else the charge of the run the structure came out of; else the phosphate rule — one negative charge per nucleic-acid backbone phosphate, [`model/chemistry.md`](?doc=model/chemistry.md); [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) § 2a) |
| `fc_displacement` | siesta | `FC.Displacement` — the nudge of the force-constant run (§ 5.3) | 0.04 Bohr | range 0.005–0.2 Bohr; smaller pushes the force difference toward the SCF noise floor, larger picks up anharmonic terms; greater than 0 by rule — SIESTA divides each force difference by it (`ofc.f90`, `template.md` § 5.3) |
| `relax_type` · `relax_steps` · `relax_force_tol` · `relax_max_displ` | siesta | the relaxation phase's driver, step cap, force tolerance and largest step (§ 5.11 § 9) — the same items the optimization kind shows; the driver offers the three relaxers only, CG · Broyden · FIRE (`offered`, [`template.md`](?doc=engines/template.md) § 6.3a): `none` relaxes nothing | the kind's **recommended** values are the tight tier: Broyden, 100 steps, **0.01 eV/Å**, 0.02 Å (`recommended = { vibration = … }` in the catalogue, [`engines/template.md`](?doc=engines/template.md) § 6.3a) | editable like every other item; the tolerance is also the yardstick the finish judges the reference geometry by (§ 5.5) |
| `write_hs` | siesta | `SaveHS` — every phase's `<label>.HSX`, the Hamiltonian and overlap at the end of each SCF | the kind's **recommended** value is **off** (`recommended = { vibration = false }`); SIESTA's own and every other kind's is on | nothing of a vibration reads an `.HSX`: the force-constant call's holds the last geometry it nudged (SIESTA writes it after every step, `final_H_f_stress.F`), an optical call's a structure pushed along a mode — 53 MB a call on a 312-atom gold junction, 6.4 GB over 121 optical calls, which a warm relaunch copies again (§ 5.11 § 6) *(the user, 2026-10-28)*. On, the relax call's is the relaxed structure's, for bands or DOS without another SCF |
| `geom_gmax` · `geom_grms` · `geom_dmax` · `geom_drms` · `geom_etol` · `geom_max_steps` | pyscf | Phase 0's geomeTRIC criteria (§ 4.2) — the optimization kind's own items | the kind's **recommended** values are the tight tier: `geom_gmax` **2·10⁻⁴ Eh/Bohr** (0.010 eV/Å), `geom_grms` 1·10⁻⁴, `geom_dmax` 1·10⁻³ Å, `geom_drms` 5·10⁻⁴ Å, `geom_etol` 1·10⁻⁶ Eh, 100 steps | the general default stays the medium tier for an optimization; the vibration kind recommends tight because a frequency deserves a real stationary point |

**Where each item sits on the form — by the result it is for** *(2026-10-27;
[`engines/template.md`](?doc=engines/template.md) § 6.2a)*. The vibration
lists its results, and a setting for one of them is drawn in that result's
card, the cards in the order a person decides them, before the method's:

| card | settings both engines share — drawn once | PySCF's own | SIESTA's own |
|---|---|---|---|
| **Frequencies and mode shapes** — relax first, then the modes | `already_relaxed` | geomeTRIC's criteria (`geom_gmax`, `geom_grms`, `geom_dmax`, `geom_drms`, `geom_etol`), `geom_max_steps`, `on_nonconvergence`, `geom_continue_retries` | `relax_type`, `relax_steps`, `relax_max_displ`, `relax_force_tol`, `fc_displacement` |
| **Infrared** | `compute_ir` | — | — |
| **Raman** | `compute_raman`, `raman_modes`, `raman_freq_min_cm1`, `raman_freq_max_cm1` | — | `optical_direction`, `optical_mesh`, `optical_broaden`, `optical_energy_max`, `optical_bands` |
| **Thermochemistry** | `temperature_K` | `pressure_atm` | — |
| **Orbital energies along a mode** | — | `es_mode_selection`, `es_explicit_indices`, `freq_min_cm1`, `freq_max_cm1`, `displacement_amplitude_ang`, `es_n_homo_below`, `es_n_lumo_above` | — |

**A setting that takes effect only under another's value is greyed out
otherwise**, by its declaration (`active_when`,
[`engines/template.md`](?doc=engines/template.md) § 5): *Raman for* and
SIESTA's optical settings while *Compute Raman* is off, the range's ends
outside *Modes in a frequency range*; PySCF's probe — its amplitude and
orbital window — under `skip`, its list outside `explicit`, its window
outside `all`. The settings gate says a value given there enters nothing.

Every other setting keeps its `group`'s card — *Run profile*, *Convergence
targets*, *Compute & budget*, *Output files* — the shared ones drawn once
(`net_charge`, `spin_treatment`, `unpaired_electrons` in *Run profile*, in
the chemistry card; `verbose_comments`, `write_molwatch_log` in *Output
files*, with the results under *The calculation*), each engine's own under
its name (*SIESTA · Run profile*). Nothing about a
setting's value changes with its card: the template, the description and
every deck are what they were. `group` still decides that *vary per stage*
starts ticked on Task setup for the `stage` items and that the run-card
settings stay off this form.

**The engine is not an item.** Which program runs a described job is the
description's `engine` (`task.json`), chosen on the Spectrum tab's engine
strip the way the Structure-optimization tab has always chosen it, carried by
the hand-over, refused by `molbuilder describe` for anything but the two engines. A
one-choice `engine` form item stood in for the strip until 2026-09-24 and
retired with it: a parameter of the deck it never was.

> **The electronic state is inherited, and a PySCF vibration is restricted**
> *(W34, decided 2026-09-25; unrestricted 2026-09-29, plan § 5w K17)*: a
> vibration built from a relaxed structure defaults to that relaxation's charge
> and spin, and a difference is warned; a PySCF vibration offers `restricted`
> alone — restricted-open has no analytic ROHF/ROKS Hessian to take, and
> unrestricted waits for the spectrum record's two spin channels (PS-C2: the
> orbital block is written one channel wide)
> ([`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) §§ 2a.2 ES7,
> 2a.3).

**The items the vibration shares with the optimization kind** — method, functional, basis, the spin treatment and count, dispersion, density
fitting, the implicit solvent (`solvent`, PCM — on the frequencies-only
route, § 4.6), the SCF machinery, the
geometry-convergence criteria (`geom_gmax` family, whose values for this
kind are the tight tier through the catalogue's `recommended` key — the row
above; V1.20 closed 2026-09-24), the relaxation's workflow knobs
(`on_nonconvergence`, `geom_max_steps`, `geom_continue_retries`,
`write_trajectory`, `write_molwatch_log`, `save_initial_xyz`,
`save_optimized_xyz` — § 4.2 says what each does here), the execution category
(threads, memory, GPU, `mpi_np`) — are the catalogue's ordinary rows, selected
into the vibration template with the kind's defaults; those one row serves on
both engines (the charge and spin, the output switches) are drawn once, in the
parts both engines share (above).

**Why infrared is on by default and Raman off, on both engines** *(the user,
2026-10-27: "both engine should share the same meaning and default
behavior")*: the infrared is the cheap result on both — PySCF's analytic
derivative costs about a seventh of the Hessian with no atom held, SIESTA's
charges a grid integration per step; with atoms held PySCF takes the dipole
at small pushes of each free atom instead, 6 whole-system SCFs per free atom,
and the settings gate says how many before the run (§ 3.3) — and Raman the
expensive one on both:
PySCF's polarizability sweep, SIESTA's optical calculations. One default on
both engines means a description reads the same whichever engine runs it.
**Ticked, Raman is for every mode on both engines** — *Raman for* starts at
*All modes* — so ticking it alone gives the same result whichever engine
runs; a frequency range is SIESTA's way to spend less, chosen from chemistry
before any calculation *(the user, 2026-10-27: "i think that should be by
default")*. A frequencies-only run turns both off; a solvated PySCF run must (§ 4.6). Their meaning is
[`engines/pyscf.md`](?doc=engines/pyscf.md) and
[`engines/siesta.md`](?doc=engines/siesta.md); the SCF knobs reach every mean
field the PySCF deck builds through one generated dresser (§ 4.3).

**Items the kind does NOT show, and why:** `restart` on a force-constant rung
(its start state is the kind's own, § 5.3; the `relax` phase reads it,
§ 5.11 § 3, and offers it; the PySCF deck carries no deck-level restart state);
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
be removed, and how many modes will be reported (R7) — counted on the axes the
engine computes with (`cell.engine_axis_kinds`,
[`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 2.1),
which on PySCF are a cluster's whatever the structure's: the count the deck
removes, the one its Methods paragraph states. PySCF's adds the
finite-difference and amplitude checks and the cost of what was ticked (the
parity of the electron count is the electronic state's, asked for every engine), stated from what the code does (R8); SIESTA's
checks are § 5.8. **A structure with a repeating axis is computed as an
isolated cluster, and the check says so** — PySCF builds a molecule in free
space, so the periodicity is not respected (`cell.periodic_in_gas_phase`, the
engine's one note for every PySCF calculation). A note, not a refusal *(user,
2026-09-29: "just note that periodicity will not be respected in pySCF"; it was
refused here until then)*: the script's harmonic analysis removes the motions
of the free cluster it computes, so the result agrees with itself, and whether
a cluster is what the person means is theirs to judge.

---

## 4. The PySCF route — the generated script, phase by phase

The deck is a Python script generated by `pyscf/vibration_deck.py` (the
composer) from blocks in `pyscf/vibration_emitters.py`, through the same seam
every deck uses — `pyscf.input.spec_for(struct, cfg, calculation="vibration")`
→ `script_emit.prepare_deck` ([`execution/script-preparation.md`](?doc=execution/script-preparation.md)
§ 4). It runs in the `molbuilder-pySCF` env, where molbuilder is not installed,
and no molbuilder function is copied into it: the relaxation, the HOMO rule and
its orbital window, the Hessian rule, the harmonic path and the steps after it
(the wavenumbers, the display form, the thermochemistry and its grid), the
hash, the result's writer and its scrub, the mode selector (§ 4.8), the
progress-log writer and the structure codec it imports from `mb_pyscf.pyz`
beside it ([`engines/pyscf.md`](?doc=engines/pyscf.md) § 3 lists what stays
written as text). It writes `<label>.spectra.json` beside itself after every
phase.

### 4.1 The script, top to bottom

```text
# constants from the description: JOB, ATOMS, ELEMENTS, N_ATOMS, FROZEN_INDICES_USER
# (the held set as the structure gave it) → FROZEN_ATOM_IDXS / FREE_ATOM_IDXS,
# AXIS_KIND (the axes gto.M computes on: a cluster's, cell.engine_axis_kinds),
# ALREADY_RELAXED, GEOM_* criteria, THERMO_T_K, THERMO_P_ATM, COMPUTE_IR,
# COMPUTE_RAMAN, RAMAN_FD_STEP_ANG, DISPLACEMENT_AMPLITUDE_ANG,
# ES_MODE_SELECTION, ...
# from mb_pyscf.pyz, imported as _mb_<name>: relax, homo_index, mo_window,
# dipole_derivatives, as_numpy, vibrational_modes, signed_omega, frequencies_cm1,
# display_form, thermo_temperatures, vibrational_thermo, vibrational_thermo_grid,
# structure_hash_text, write_spectra_payload, finite_or_none, select_modes,
# MolwatchEmitter, StructureCodec, and the constants module — each from its one
# home

mol = gto.M(atoms = EVERY atom, held ones included; basis, charge, spin, ...)
_mb_configure_scf / _mb_configure_theory   # the generated dressers (§ 4.3)
state = {schema_version: SCHEMA_VERSION, engine: 'pyscf', phases: empty, ...}; write the artifact

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
`_mb_configure_scf(mf)` and `_mb_configure_theory(mf)` —
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
When it ends, the phase records `relaxation.converged` — the judged force at
the geometry it reached (the free atoms' largest component, R5) against
`geom_gmax`, the one meaning that key has on every route (§ 4.3, § 5.5) —
while geomeTRIC's own verdict, all of its criteria, is what decides the
policy below.

What the phase honours from the description: the convergence criteria
(`geom_gmax`, `geom_grms`, `geom_dmax`, `geom_drms`, `geom_etol`,
`geom_max_steps` — the tight tier at the kind's recommendation, § 3.1), `on_nonconvergence`
(**this** is the phase that policy governs, through the one relaxation function
the optimisation deck relaxes with — [`pyscf.md` § 3](?doc=engines/pyscf.md),
which asks geomeTRIC whether it converged rather than assuming it: `halt` stops
the run before the Hessian; `continue` re-enters from the geometry reached, up to
the retry budget; `proceed` takes that geometry, with a warning that geomeTRIC's
criteria were not met beside `converged`, the judged force's verdict above),
`write_trajectory` (geomeTRIC's streaming XYZ), `write_molwatch_log` (the same
live-watch hooks the optimisation deck emits), `save_initial_xyz` /
`save_optimized_xyz` (`<job>_initial.xyz`, `<job>_optimized.xyz`, written as
**pairs**, geometry plus `.molstruct.json`, through the codec — a bare `.xyz`
carries no labels, no cell and no identity; fixed 2026-09-22 and proven by a
held set `frozen_atoms: [6, 7]` surviving into the output and reading back
through molbuilder's own codec). The optimizer is geomeTRIC, the engine's one
([`pyscf.md` § 3](?doc=engines/pyscf.md) says why `berny`, and with it the
`optimizer` item, was retired).

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
stored as is), `mo_energies_eh` (a non-finite entry written as null by
`sidecars.spectra.finite_or_none`) and `homo_idx` (derived by `homo_index`: sum the
two spin channels of a 2-D `mo_occ`, take the highest index with occupancy
above 0.5 — a rule with a branch whose failure would be silent and would
affect only open-shell work, which is why it was lifted out of inline script
text on 2026-09-09 so one implementation runs and is tested), plus `elements`
and `positions_ang`.

**The stationarity check (R5)** takes the SCF's nuclear gradient and judges
the largest absolute force component **over the free atoms** against the
template's own `geom_gmax` — 2·10⁻⁴ Eh/Bohr at the kind's recommendation —
the one rule both routes judge by (§ 2.2; SIESTA's finish judges the same
quantity against `relax_force_tol`, § 5.5) *(ten times it until 2026-09-24)*,
recording it as `relaxation.max_force_eh_bohr` and the all-atom figure beside
it as `max_force_all_atoms_eh_bohr`, and the verdict as `relaxation.converged`
— the key the relaxation phase sets from the force at the geometry it reached
(§ 4.2) and SIESTA's finish from its reference step (§ 5.5), so it means one
thing on every route: the judged force within this calculation's criterion — the keys say their unit *(they said
`_a` until 2026-09-24 while the viewer printed "Eh/Å"; the number was always
Eh/Bohr)*. Measured
2026-09-22, why the free-atom rule matters: water with O and one H held,
relaxed to the deck's own criterion, reads 3.35·10⁻² over all atoms against
7.01·10⁻⁵ on the free ones; CO₂ with both O held 7.2·10⁻² against
3.9·10⁻¹⁴. Judging every atom (the old `np.abs(_g0).max()`) warned on four of
five correct constrained minima, and the warning reached the web UI.

**The structure the result states** (`structure`, § 6.10.1) is built at the
equilibrium write: the geometry the Hessian is taken at with the run's own
sidecar payload applied whole (`_MB_SIDECAR`, through
`molstruct.apply_to_structure` — `spectra.pyscf_vibration.equilibrium_structure`,
imported from `mb_pyscf.pyz`). A schema-8 file carried a hash of a structure
kept elsewhere instead (`structure_hash`); it is read from such a file and
never written.

### 4.4 Phase 2 — the Hessian over the free atoms

**The rule** (R8): with atoms held, second derivatives are computed **for the
free atoms only**, and the run says so (`hessian_scope = 'free'`,
`n_atoms_in_hessian`). *(The audit of 2026-09-22 had ruled this an opt-in
with analytic infrared as the default; that ruling was **superseded on
2026-09-23 when the route was built**: no option was added, holding atoms is
what asks for the reduced calculation, and the run states which way it went.
The archived design records both.)* The function named `dipole_derivatives`
is the one place this is decided — the name is historical (it first produced
$d\mu/dR$ beside the Hessian), and it is the sole producer of the Hessian in the
deck. With nothing held the
Hessian is the whole molecule's and `hessian_scope = 'all'`.
`spectra/pyscf_vibration.dipole_derivatives(mf, free_atom_idxs, want_ir)`,
which the deck imports, is the one place this is decided:

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
— imported by the PySCF deck from `mb_pyscf.pyz`, called beside the run by the SIESTA finish (§ 5.5) —
is the only place modes are made. In order: take the free–free block; divide
by $\sqrt{m_i m_j}$ per 3×3; build the surviving whole-body motions with
`rigid_motions` (three slides, the turns the lattice permits, then only the
combinations that leave every held atom in place, then only those that move a
free atom — the rank of [`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 3.1, tolerance 10⁻³ Å per unit motion); diagonalise in the **complement** of
those motions in the mass-weighted metric; return $3\,N_{\mathrm{free}} - n_{\mathrm{rigid}}$
eigenvalues (ascending; negative = imaginary), the modes in the canonical
normalisation $\sum m_k \lvert L_k\rvert^2 = 1$ (amu), and the removed patterns.

Conventions that must not fork: isotope-averaged masses, stated once
(`chemistry.atomic_mass` at prep, written into the deck and the result — I3,
§ 6.2; the table PySCF's `thermo` uses holds the same numbers, and the deck
checks it does); the wavenumber conversion
$\sqrt{\lambda}\times 5140.487\ldots$ for a Hessian in Hartree/Bohr² weighted in amu
(`constants.CM1_PER_SQRT_HARTREE_BOHR2_AMU`, derived from the two constants it
is made of); the display form `eigenvector_display` = the canonical vector
rescaled per mode so $\max\lvert L_k\rvert = 1$, for the animation only (§ 6.3). Both
conventions were measured on 2026-09-22: the wavenumber constant agrees with
PySCF's own derivation to 1·10⁻⁹ relative, and $\sum m\lvert L\rvert^2$ comes out
1.00000000 on the free path and on every held system. The
artifact records what was removed (`removed_motions.count`, `patterns`), so a
reader can see the difference between $3\,N_{\mathrm{free}}$ and `len(modes)` (R7).

The gate this rests on: the rank rule reproduces PySCF's own
`harmonic_analysis` on free molecules — water, CO₂, HF, methane at
RHF/STO-3G: eigenvalues to 1·10⁻⁸ relative, wavenumbers to 1·10⁻⁴ cm⁻¹,
water's and HF's vectors to 1·10⁻⁶ in the mass metric (methane's degenerate
sets compared by frequency only) — before the old two-branch analysis was
deleted (2026-09-23).

### 4.6 Phase 3 — infrared and Raman strengths

> **PCM reaches one route** *(ruled 2026-09-29, plan § 5w K17; refused since
> 2026-09-30, K2)*. PySCF's PCM Hessian solves under equilibrium solvation and
> adds the solvent's own second derivative (`with_solvent.hess`); the deck's
> other routes are built without it — the held-atom Hessian
> (`hess_elec(atmlst=)` + `hess_nuc`), the analytic IR block, and Raman's
> polarizability loop. So with a solvent set, a PySCF vibration runs the
> frequencies of a structure with no atoms held, with IR and Raman off; any
> of the three is refused by name at `prep`, until those routes carry the
> solvent and are measured.

**What an intensity is.** An infrared band is strong when the vibration moves
charge: the quantity is the dipole's rate of change along the mode,

$$
I = 42.2561 \times \Bigl\lvert \frac{d\mu}{dQ} \Bigr\rvert^2 \ \mathrm{km/mol},
\qquad
\frac{d\mu}{dQ_n} = \sum_{k,\alpha} \frac{d\mu}{dR_{k,\alpha}}\, L^{\mathrm{canonical}}_{k,\alpha,n}
$$

with $\mu$ in Debye and $Q$ in Å·√amu; and a Raman band scatters when the mode changes the polarizability: $d\alpha/dQ$,
combined by Placzek's formula $S = 45\,a^2 + 7\,\gamma^2$ into an activity in Å⁴/amu.
The `42.2561` prefactor is derived for modes normalised in **amu** — the
convention of § 4.5 — which is where the 1823× defect lived.

**Three ways to obtain $d\mu/dR$, and what each costs** (measured NH₃/PBE0/6-31G,
2026-09-11; the two tensors agree to 0.02 %, so the choice is cost, not
accuracy):

| route | how | cost | availability |
|---|---|---|---|
| **analytic** | the same coupled-perturbed response the Hessian solves, contracted with dipole integrals — one solve for both | **+14 %** over the Hessian alone | `pyscf.prop.infrared`, master branch only, never released to PyPI |
| **finite-difference dipoles** | nudge each free coordinate ±0.005 Å, read the converged dipole, difference | **+486 %** — $6\,N_{\mathrm{free}}$ extra SCFs | always |
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
fitting) and 5.7·10⁻¹² (dispersion) — $\max\lvert H(\text{analytic route}) - H(\text{plain})\rvert = 6.7\cdot10^{-16}$ on the corrected deck — **and the corrected route is faster**
(14.8 s against 14.4 s for the Hessian alone: still one solve). The tensor
layout is measured too: against the finite-difference tensor the emitted
$d\mu/dR$ differs by 8.4·10⁻⁵, its transpose by 5.2·10⁻¹.

**Raman** has one route: the static polarizability is analytic
(`pyscf.prop.polarizability`, coupled-perturbed; the polarizability points run
**without density fitting** because the module has no DF implementation) at
each displaced geometry, and its **derivative** is a central difference over
the free Cartesian coordinates at ±`RAMAN_FD_STEP_ANG` = 0.005 Å. PySCF reports the
polarizability in atomic units (Bohr³), so one global factor $(\mathrm{Bohr}/\text{Å})^6 \approx 0.021959$ is applied on the final scalar, and the stored value is genuine Å⁴/amu,
comparable to Gaussian's and ORCA's activity columns — without it the
spectrum's shape is right and every absolute activity is about 45.5× too large. The run records the
route as `raman_route = 'finite-difference'` with `raman_fd_step_ang` (§ 10 for
status), and the Methods text states the method one way: analytic
polarizability, finite-difference derivative.

**Where the numbers are not well defined.** The dipole is taken about the
fixed point (0, 0, 0), PySCF's default, so its derivative is the same from any
fixed origin. A charged molecule with no atom held has intensities as well
defined as a neutral one's. With atoms held, a charged molecule's intensity
carries $q\,d\mathbf R_{\mathrm{cm}}/dQ$, the net charge's own motion
([`science/normal-modes.md`](?doc=science/normal-modes.md) § 4a.5). With atoms held the formulas
consume the free-atom vectors and derivatives, both of which exist, but a held
atom contributes no dipole derivative: for a molecule anchored at one or two
atoms a small correction, for a molecule on a metal that screens and carries
the charge transfer, not small. The number is computed; how much it means is
the reader's judgement.

**What the file keeps, and how it is used — held or free** *(2.37.0; the
user, 2026-10-24: "if no fix atoms are listed, that means gas phase, and you
can keep conventional average, but when we have fixed atoms that means we
have rigidity and surface effects, so it is correct to see specific peak
respond better in certain EM and the result is real"; "would need document
to be updated explaining how pyscf calculates, and how saved values are
used, justified by scientific arguments")*.

1. **What PySCF computes.** Whichever route above ran, the deck holds
   `DMU_DR`: the derivative of the whole system's dipole with respect to
   every free atom's Cartesian coordinates, in D/Å — each free atom moved,
   every electron answering, the held atoms' electrons among them, their
   nuclei fixed (§ 4.4). It projects that on each mode's canonical pattern,
   normalised in amu (§ 4.5):

   $$
   \frac{d\mu}{dQ_n} = \sum_{k\ \text{free},\ \alpha} \frac{d\mu}{dR_{k,\alpha}}\, L^{\mathrm{canonical}}_{k,\alpha,n}
   $$

   (a 3-vector, in D/(Å·√amu))

   This vector is the mode's **transition dipole** up to a constant: in the
   harmonic picture the fundamental's transition dipole is
   $(d\mu/dQ_n)\sqrt{\hbar/2\omega_n}$ [Wilson1955], and light whose field points along
   the unit vector $\hat e$ is absorbed at a rate proportional to the square of its
   projection, $(\hat e\cdot d\mu/dQ_n)^2$.

2. **What the file keeps.** For every mode, two things computed from the
   one vector:
   - `ir_intensity_km_mol` — $42.2561\times\lvert d\mu/dQ_n\rvert^2$, the **orientation
     average**: a molecule tumbling at random meets the field at every angle
     $\theta$, and $\langle\cos^2\theta\rangle = 1/3$, so the average of $(\hat e\cdot v)^2$ is $\lvert v\rvert^2/3$. It is
     the number Gaussian, ORCA and the literature report, comparable with
     them as it always was;
   - `dipole_derivative_d_per_ang_amu12` — **the vector itself**, its three
     components x, y, z (schema 12, § 6.2): what every direction is computed
     from. A file written before it states none.

3. **How the page uses them — by whether the result holds atoms**
   (`spectra.infrared.orientation_of`, the same rule as SIESTA's,
   [`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)
   § 2.3):
   - **nothing held — a free molecule**, the gas phase: it tumbles, and the
     infrared column shows the file's `ir_intensity_km_mol`, the orientation
     average, unchanged;
   - **atoms held — a molecule anchored to a cluster standing for a surface
     or a junction**: it is rigid, its orientation to the light fixed, and
     the column shows the strength along the light's direction the person
     picks (x, y, z or any vector, z by default):

     $$
     A_n(\hat e) = 3 \times 42.2561 \times \Bigl(\hat e\cdot\frac{d\mu}{dQ_n}\Bigr)^2 \ \mathrm{km/mol}
     $$

     A mode whose dipole change lies along $\hat e$ reads three times its average,
     one across it reads zero; the mean over x, y and z gives the average
     back, $[A(\hat x) + A(\hat y) + A(\hat z)]/3 = 42.2561\times\lvert d\mu/dQ_n\rvert^2$ — the two columns
     are one vector read two ways, so nothing is lost and nothing is made up.

4. **Why the held molecule is not averaged.** Averaging over orientations
   assumes the molecule takes every orientation with equal chance; one
   anchored to a surface does not, so its peaks answer one polarization more
   than another, and that is a measurement, not an artefact: at a metal
   surface only the field's component normal to the surface survives, so an
   adsorbed molecule's infrared shows the dipole changes along the normal
   and loses those in the plane [Greenler1966]. The held atoms are the
   model's statement that the molecule is anchored, which is why they decide
   the rule. What the anchoring does to the electrons — the charge moved
   across the contact, the cluster's own response — is in $d\mu/dR$ already,
   with the caveat above for what a held atom's fixed nucleus leaves out.

5. **A file written before 2.37.0** states no vector: a held result then
   shows its orientation average, as before, and the page says the direction
   cannot be applied to it.

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
| `vibrational-only` | any atom held | the vibrational sums above the electronic energy, `vibrational_thermo` / `vibrational_thermo_grid` from the one home — there is no gas-phase translational or rotational partition function to add, and no $kT$ (the ideal gas's $pV$) either |

**One quantity under one label**: the headline numbers and the curves the
viewer draws are the same sum, and the headline temperature is a point of the
grid (`THERMO_GRID_K` = 50–1500 K in 30 points, a documented presentation
default, plus the headline T). *(Until 2026-09-24 the headline was full RRHO and
the grid vibrational-only under one note — measured on free water at a
temperature exactly on the grid: headline G = −74.95926467 Eh against the
grid's −74.94057624 at 300 K, ΔG = +11.73 kcal/mol, ΔS = −45.28 cal/mol/K.
Status in § 10.)* On SIESTA no total energy is reported, so `h_eh` and `g_eh`
there are the vibrational contributions alone, above an electronic minimum
taken as zero; the file's `note` says so. The deck computes, the viewer draws;
the viewer derives nothing — its electronic reference is the file's own
equilibrium energy, and zero where the route carries none
([`web/spectra.md`](?doc=web/spectra.md) § 3, which labels each regime).

**What the vibrational-only numbers are good for, and where they stop** *(user,
2026-09-28: "if thermochemistry is not really useful/correctly done in siesta,
then say so. the science must be correct and consistent")*. They are the
harmonic vibrational contributions of the free atoms at the stated
temperature — the zero-point energy, the thermal vibrational energy $U_{\mathrm{vib}}$,
the vibrational entropy $S_{\mathrm{vib}}$, and $F_{\mathrm{vib}} = \mathrm{ZPE} + U_{\mathrm{vib}} - T\,S_{\mathrm{vib}}$ — computed
correctly for what they are, and they are the standard harmonic limit for an
adsorbate or a junction: the zero-point and vibrational free-energy
correction of one structure against another computed the same way (two
adsorption sites; a molecule before and after binding, with the same held
set). They are **not** an enthalpy or a free energy of the system: SIESTA's
route carries no electronic energy (the reference is zero), and a free
molecule's translation and rotation are not added — for a gas-phase free
energy, PySCF's full RRHO with nothing held is the route. Modes below about
100 cm⁻¹ — an adsorbate's frustrated translations and rotations, torsions —
are anharmonic in practice and carry most of $S_{\mathrm{vib}}$, so the entropy is the
least reliable number here, and the partial Hessian's clamped atoms stiffen
the lowest contact modes besides (§ 1.2). **No pressure enters** any of them
— only the gas-phase translational term takes one — so a vibrational-only
result records `pressure_atm: null`, on both engines. The file's `note` says
this beside the numbers, on both engines and in both regimes: what they are
good for and where they stop, and, for a vibrational-only answer, why it is
one — atoms held, or a free molecule whose full RRHO PySCF could not give.
**Near 0 K** every sum is the zero-point energy: the thermal energy and the
entropy go to zero and the ZPE does not, which is what `vibrational_thermo`
returns at $T \le 0$; the item itself starts at 1 K, because the free
molecule's translational entropy has no value at 0 K.

### 4.8 Phase 4 — the electronic structure along a mode

For each selected mode the deck pushes the geometry to $q \pm A\,L_{\mathrm{display}}$ —
$A$ = `displacement_amplitude_ang` along the **display** form of the eigenvector
(its largest absolute Cartesian component = $A$) — runs an SCF at each, and records the orbital
window `[HOMO − es_n_homo_below, LUMO + es_n_lumo_above]` and the SCF energy at
both, beside the equilibrium ones: **two SCFs per selected mode**. The viewer
draws the three level stacks joined orbital by orbital, the gap's shift, and
the coupling $\Delta E/(2A)$ — the electron–vibration coupling that decides how a
mode modulates a junction's transmission [Galperin2007, Frederiksen2007].
**Which coordinate.** $A$ is the largest absolute Cartesian component of
the motion along the display eigenvector (not the largest per-atom length —
an atom moving off-axis swings up to $\sqrt{3}\,A$), so $\Delta E/(2A)$ is a slope per Å
of that component. The slope per unit
normal coordinate, $\partial\varepsilon/\partial Q_\nu = \Delta E/(2A)\cdot\max\lvert L_{\mathrm{canonical}}\rvert$ (the largest absolute component), and the
coupling per zero-point amplitude, $g_\nu = (\partial\varepsilon/\partial Q_\nu)\sqrt{\hbar/2\omega}$ — the number
the inelastic-transport literature quotes — follow from the file, since both
eigenvector forms are in it ([`science/normal-modes.md`](?doc=science/normal-modes.md)
§ 4b.5 F), and neither is written today. Two points give the slope and not
its linearity over $A$; the five-point sample, $g_\nu$, and a molecule-projected
window for a cluster whose HOMO and LUMO are metal states are V1.27 (§ 10).

Which modes get it is `es_mode_selection` with the frequency window
(`spectra/selection.py::select_modes`, which the deck imports from
`mb_pyscf.pyz`): `skip` (none), `all` (every mode inside the window), `explicit`
(the listed modes, `PySCFConfig.explicit_modes`, written into the deck as
numbers; the window is ignored — naming a mode is saying *that one*). A
window set outside `all` enters nothing and `prep` says so; the form locks it
there, but a locked field keeps its value and the hand-over carries it. `top_n` and `threshold` ranked modes by Raman activity; they were
**retired by decision** (2026-09-23) and removed (2026-09-28, V1.6): the probe measures $\partial\varepsilon/\partial Q$, which
follows its own selection rule — in a centrosymmetric molecule the
Raman-bright modes are exactly the infrared-dark ones, so the filter kept one
symmetry class and dropped the other every time, and for an engine that
computes no strengths they were undefined rather than empty. The window is the
cost control, and `all` is cheap where it matters (8 SCFs for CO₂). A template
written before the removal that still answers `es_top_n` or `es_threshold` is
refused at `prep` with the two named as retired (`template.RETIRED_ITEMS`) —
delete them: no value of theirs runs as before, so `molbuilder jobset migrate`
refuses one too, and takes out only a blank one, which asks nothing; one that
still selects `top_n` or `threshold` is refused as a
choice the item does not have. A run probes every mode its selection
names: it starts its `.spectra.json` afresh and takes nothing from an earlier
run's file *(a resume that skipped the modes already probed was written here
until 2026-10-05; nothing ever did it -- the selector's `prior` argument,
removed that day, had no caller)*. A listed number that names no mode is
dropped from the selection and said in the run's output.

### 4.9 The live artifact

The script writes `<label>.spectra.json` at the end of every phase, after
every relaxation step, and after every per-mode SCF, always by **atomic
replace** (a temporary file, then `os.replace`), so a reader polling every two
seconds never sees a torn file. Five phases, five flags — the Hessian and the
harmonic analysis (Phase 2) close `phase_frequencies`, which Phase 1 opened, and
the infrared intensities carry a flag of their own, closed by whichever sweep
computes them — are the reader's clock:

```text
phase_relaxation   empty → running (step count, max force ticking) → complete   (complete by assertion under already_relaxed;
                                                                                 on SIESTA `complete` when the run's `relax` call ran — its
                                                                                 record gives `n_steps` — and `not requested` when the box was
                                                                                 ticked: the force-constant run itself relaxes nothing)
phase_frequencies  empty → running → complete
phase_raman        empty → running → complete      (`not requested` from the first write when the description
                                                    asked for no Raman (`compute_raman` off, on both engines); on
                                                    SIESTA, running while the optical calls run, § 5.11 § 8)
phase_ir           empty → running → complete      (running with the Raman sweep, which carries the dipole, or through
                                                    the dipole sweep alone; `not requested` when no infrared was asked for
                                                    (`compute_ir` off, on both engines); on SIESTA `complete` once the
                                                    run's charges are stored, § 5.11 § 8;
                                                    `""` in a file written before 2026-09-28 -- no record,
                                                    which a reader must not take for a phase still to come)
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

**Why infrared has a flag of its own** *(2026-09-28, the review of M2b‴)*: the
dipole sweep writes nothing until it ends, so without one a run asking for
infrared alone showed its frequencies complete, no intensity and no phase
running — and the viewer called the run finished, stopped watching it, and
said it had recorded none.

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
infrared — which $d\mu/dR$ route runs is settled inside the job — and the load path
adds the route sentence from `ir_route` and `ir_fd_step_ang`
(`with_ir_route`, § 6). The count it states is R2 from the one derivation
(R6), equal to the run's by construction. **The level of theory is the one
answer `PySCFConfig.is_dft`** — asked by the deck's SCF construction, its
header, its constants and line spellings, the grid advisory and this
paragraph alike: under Hartree–Fock (RHF, UHF) the paragraph names the method
and the basis, never a functional, and a functional changed from its default
under Hartree–Fock is warned about, since it enters nothing (§ 3.1). **The
dispersion correction is not a DFT question**: Hartree–Fock has no electron
correlation, so it misses London dispersion entirely; D3 and D4 carry damping
parameters fitted for it; and PySCF carries `mf.disp` on an HF object through
the energy, the gradient and the Hessian, reading the method as `hf`
([`engines/pyscf.md`](?doc=engines/pyscf.md) § 7a). So the item is applied
as written on either method — RHF with `d3bj` is HF-D3(BJ) — and the
paragraph names it with its own papers: D3(BJ) [Grimme2010, Grimme2011],
D3(0) [Grimme2010], D4 [Caldeweyher2019]. *(Until 2026-09-28 every place
that asked "is this DFT" decided it by hand, and the paragraph read
`cfg.functional` whatever the method, so a Hartree–Fock run's write-up named
B3LYP while its next sentence named `pyscf.hessian.rhf`, and the grid
advisory warned about B3LYP's grid on a run with no grid; both decks dropped
the dispersion correction under Hartree–Fock — silently at its default — as if
the method could not take one, while the result recorded the value as run; and
every dispersion version cited the BJ-damping paper alone. A `results=` arm that would have re-composed the
paragraph after the run, which nothing in production called, was deleted —
V1.8, V1.18.)*

---

## 5. The SIESTA route — force constants, a sorted copy, and the job's own finish

### 5.1 What SIESTA does

SIESTA takes no second derivative. A **force-constant run** nudges each atom of
one contiguous range by a fixed displacement along x, y and z, both ways, runs
an SCF at each displaced geometry, and writes the forces on every atom to
`<SystemLabel>.FC`. The keywords, with the spellings SIESTA 5.4.2 honours
(matched the way fdf matches, `parse/fdf.py::fdf_key` — never against `strings`
on the binary, whose concatenated keyword table matches any prefix; `FC.Displ`
"matched" four times and is not a keyword):

| what | keyword | note |
|---|---|---|
| ask for a force-constant run | `MD.TypeOfRun FC` | `PHONON` is retired |
| first / last atom to nudge | `FC.First` / `FC.Last` | 1-based, **one contiguous range**; `MD.FCFirst` / `MD.FCLast` are deprecated aliases |
| how far to nudge | `FC.Displacement` | Bohr; SIESTA's default 0.04; `MD.FCDispl` deprecated |
| the result | `<SystemLabel>.FC` (and `.FCC`) | § 5.4 |
| the derivatives of H and S | `FC.Save.dHS` | the manual's option for writing $\partial H/\partial R$ and $\partial S/\partial R$, the ingredient of the electron–vibration coupling (§ 5.6); not used, and its presence in the shipped binary is unverified |

SIESTA has no unrecognised-keyword diagnostic, so a misspelled keyword runs at
the engine's own default without a word. Every keyword this deck writes is a
catalogue anchor or a line of `siesta/vibration_deck.py`.

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
computes lies outside the one harmonic path, so the job reads the file itself
through that path (§ 5.5) and the deck carries none of `vibra`'s inputs.

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
  atoms last, each in the user order among themselves — through the one sort
  machinery (`transport/sort.py`: `sort_by(struct, "held-first")` hands its
  order to the shared `apply_order`, which remaps every per-atom field, checks
  the bijection and records both directions; a second key over the same
  machine, never a second machine; the categorical `transport` key is the
  other);
- the copy carries its order record in itself — `atom_order`, the key that
  made it and both directions, set by `sort_by` ([`model/overview.md`](?doc=model/overview.md)
  § 2.2a) — none when the sort moved nothing (I2);
- the deck is rendered from the copy and **states its order record** in its
  ATOM-METADATA block (I5); the free range is the tail,
  `FC.First = n_held + 1`, `FC.Last = N`;
- the finish (§ 5.5) reads the order record from its own deck beside the job
  (`deck_record.order_record_of`) and puts every per-atom row back in the
  user order through `Permutation.map_rows_to_user_order` and `map_to_user_index` —
  the only way any reader may: **the user order never reaches the engine and
  the sorted order never reaches a person.** A deck stating none is in the
  user order (I2).

The deck writer checks the shape and refuses by name a copy whose free atoms
are not one trailing run (`vibration_deck.fc_facts`): a deck rendered from the
user order would nudge the wrong atoms and converge without complaint.
TranSIESTA's `elec-pos` demands consecutive atoms for the same reason, which
is why the machinery already existed. Two sorts are never composed: `sort_by`
refuses a structure already remapped, and a door restores before it sorts
again ([`model/overview.md`](?doc=model/overview.md) § 2.2a, I4). The sorted
order never reaches a person: a door that shows a reordered run marks it, and
a door that writes for a person restores it (I1).

```text
the user order:                    sorted copy (the engine's):
  user index 0  H  free              engine index 0  H  held   ← Geometry.Constraints: SIESTA's atom 1
  user index 1  H  held              engine index 1  H  free   ← FC.First = 2, FC.Last = 2 (SIESTA's atom 2)
                                     the deck's ATOM-METADATA block:
                                     "atom_order": {"key": "held-first",
                                       "sorted_to_original": [1, 0], "original_to_sorted": [1, 0]}
```

### 5.2a The relaxation — a phase of the one run

With the box unticked a SIESTA vibration relaxes first **inside its one
run**: the `relax` phase, an ordinary relaxation deck rendered from the
stage's one resolved configuration (`spec_for(part="relax")`, § 5.11 § 4)
— `MD.TypeOfRun` from `relax_type`, `MD.MaxForceTol` from
`relax_force_tol`, the held atoms in `Geometry.Constraints` — then the
force constants at the geometry it reached, which the run's own plan step
writes as the force-constant call's `coordinates.fdf` (§ 5.11 § 3). With
the box ticked the run has no `relax` phase and measures the structure as
given (§ 2.2). The stage carries every phase's role, so the stage table
offers the relaxation's driver, step cap, force tolerance and largest step
and the displacement on its one row (`template.stage_roles`,
[`template.md`](?doc=engines/template.md) § 6.4); nothing is handed over
between stages, and prep reads no run (§ 5.11 § 1, § 6;
[`execution/job-system.md`](?doc=execution/job-system.md) § 5.4). On
PySCF the one `freq` rung relaxes inside its deck (§ 4.2).

### 5.3 The deck

`siesta/input.py::spec_for(struct, cfg, calculation="vibration", part="fc")`
— the `fc` phase's deck (§ 5.11 § 4) — renders the optimisation deck's sections ([`engines/siesta.md`](?doc=engines/siesta.md)
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
Charge.Hirshfeld  geometry        # every atom's partial charge at every step, when
Charge.Voronoi    geometry        #   Compute IR is on (below)
DM.UseSaveDM      .true.          # the start state is the KIND's (below)
MD.UseSaveXV      .false.
```

**Every atom's partial charge at every step, by both partitions, when
*Compute IR* is on** — the output § 5.10 ② decided (D3; the user, 2026-10-24:
"both on"), asked by the setting both engines share (§ 3.1). With
`Charge.Hirshfeld geometry` and `Charge.Voronoi geometry`, SIESTA computes
every atom's Hirshfeld and Voronoi charge at the end of every geometry step —
the reference step and every displacement — and prints both to its output,
to 10⁻⁶ e, positive where electrons left the atom (SIESTA 5.4.2 `grdsam.F`,
`partial_charges.F90`): a grid integration each per step, no extra SCF.
Written by the force-constant block as its own text, beside the run type;
with *Compute IR* off the block writes neither and says so in one comment —
SIESTA then computes no Hirshfeld or Voronoi population (`read_options.F90`)
— and the result's infrared is *not requested*. The finish reads which were
asked from the deck's own two lines, and a partition asked whose charges the
output does not print is refused by name. What reads them is § 5.10 ②, and
what they give, [`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)
§ 2.

The stage header line names the range, the counts and the displacement
(`siesta/vibration_deck.py::stage_science`); the record blocks that every deck
carries follow, and one more that only this kind's deck carries — the
**`vibration` block**, what the finish (§ 5.5) needs and no SIESTA keyword
states:

```text
# === molbuilder vibration BEGIN ===
# format: molbuilder-vibration/v5
# {
#   "stage":"freq",                   # the stage's name -> the result's config.stage
#   "force_criterion_ev_ang":0.01,    # the tolerance the relaxation runs to (§ 5.5)
#   "force_criterion_from":"...",     # where that tolerance came from: the stage's
#                                     #   relax_force_tol -- said, not read
#   "already_relaxed":false,          # the person's statement, as made
#   "relaxation":null,                # never written here: the relax call's record is the
#                                     #   run's own, read from its index at the finish (§ 5.11 § 3)
#   "relaxation_stage":null,          # null here too: the finish names this stage, whose
#                                     #   relax call it was, beside the record it carries
#   "raman":{"modes":"all",           # Raman for, as the optical phase applies it (§ 5.11 § 4):
#            "freq_min_cm1":null,     #   all or range, and the range's two ends (null for an
#            "freq_max_cm1":null},    #   open end, and under all) -- the whole key null when
#                                     #   Compute Raman is off
#   "temperature_K":298.15,           # the thermochemistry's temperature (§ 4.7)
#   "masses_amu":[1.008, 1.008],      # every atom's mass, the deck's order (I3)
#   "title":"...",                    # the structure's title -> the result's structure (§ 6.10.1)
#   "identity":{...},                 # the identity columns, the deck's order -- restored with the rest -> the result's structure
#   "molbuilder_version":"1.1.0"      # the molbuilder that prepared it -> the result
# }
# === molbuilder vibration END ===
```

`relaxation` is `null` in the deck: the relaxation record
(`parse.contract.relaxation_of_output`) of the `relax` call is written into the
run's index by that call's record step and carried into the result by the
finish (§ 5.11 § 3), so the finish never re-picks an attempt. `raman` is
*Raman for* as asked, the one fact of the optical phase no SIESTA keyword
states — `null` when *Compute Raman* is off, and the stage then runs no
optical phase; the record step applies it (`select_modes`) and the index
records what was applied. A block of format v3 or v4 — v4 carried SIESTA's
retired Raman selector as `selection` — states no `raman`, and the record
step after the force constants refuses it naming *prepare the stage again*
(§ 5.11 § 4). The block is written by `script_emit.emit_vibration_record`, assembled by
the framework from `DeckSpec.vibration` — the engine places the values `prep`
built (`spectra.siesta_vibration.vibration_record`), as it does the bench
marks' — and read by `deck_record.extract_vibration_record`, beside the
`engine-offset` reader; the spec also names the finish, `DeckSpec.finish`,
which `prep` copies onto the stage's job (`Job.finish`).

**The start state is stated where it is read.** The catalogue offers
`restart` to optimisations and, since 2026-10-19, to **a vibration's
relaxation** (`[item.restart]`, `stages = ["relaxation"]`): the `relax`
phase's deck is the ordinary relaxation deck (§ 5.2a), which writes the
restart group from the field, so the field stands on the stage's cell and in
that deck's provenance like any other — its default `continue` (a relaxation launched
again picks up where its latest attempt stopped), `clean` if chosen. **A force-constant run**
has no optimiser history to
resume. The density is read when present — the reference step's
SCF starts from it — and SIESTA saves the reference step's converged density and
reloads it before **every** displacement, so each displacement's SCF starts from
the undisplaced density, not the previous displacement's (SIESTA 5.4.2
`save_density_matrix.F90`, `m_new_dm.F90`; the K6 review). The geometry is
**never** read back: an FC run leaves its *last displacement* in `<label>.XV` —
measured on the two-atom run, the last free atom sits `FC.Displacement` off the
input along z afterwards — so a deck that honoured that file would take a nudged
geometry as the stationary point and converge. SIESTA reads a density it finds
unless told `.false.` (`DM.UseSaveDM` defaults true) and a geometry only when
told `.true.` (`MD.UseSaveXV` defaults false), so both answers are written, not
left out (`start_state_lines`, from the restart group's own
declaration). `siesta/warm-files.toml` carries the `[vibration]` section under
the growth rule of [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 4.2a (a new calculation type is a new section, never a branch) — `.FC` and `.FCC`, inventory-only, since a rerun restarts
at `FC.First`, which the section states as its one section-level fact,
`resumes = false`, so the stage's wrapper says a retry repeats the run rather
than calling it a resume — and says why the base `.XV` row means something else
under this kind. A vibration's `relax` phase reads `[optimization]` and
carries its `.CG`; its optical calls read `[optical]` (§ 5.11 § 6).

### 5.4 The run, and what it leaves

```text
SCF at R₀ (every atom present, the held ones too)
for a in FC.First .. FC.Last:
    for α in (x, y, z):
        for s in (−, +):
            R = R₀;  R[a, α] += s · δ
            SCF at R, started from the reference density  →  forces F_b on EVERY atom b
            one row per atom b  →  <SystemLabel>.FC
```

$1 + 6\,n_{\mathrm{free}}$ force evaluations (§ 5.7 counts what they cost). What is left:

| file | what it is | measured on a two-atom run (2026-09-23) |
|---|---|---|
| `<label>.FC` | one header (`n_atoms`, $\delta$ in Å), then $6\,n_{\mathrm{free}}\,N$ rows of three numbers in **eV/Å²**, in the order displaced atom → direction → side (−, +) → atom | the file's two sides average to 41.713 for the H₂ bond; two single points displaced by hand give $-\Delta F/2\delta$ = 41.713 |
| `<label>.FCC` | the same with the held atoms' force rows zeroed (SIESTA's "constrained" variant) | the free block is identical, so the reader takes `.FC` and slices |
| `<label>.XV` | the **last displaced** geometry, not the input | the last free atom `FC.Displacement` off along z |
| `<label>.DM`, the usual outputs | the reference (undisplaced) step's density (`save_density_matrix.F90` writes it at the first step only), the `.out` with the version line and, with *Compute IR* on, after every step's SCF, every atom's Hirshfeld and Voronoi charges (§ 5.3) | |
| `<label>.spectra.json` | **the result**, at the run's root (`run-<n>/`), not in this call's folder: written by this call's record step and again by the finish (§ 5.5, § 5.11 § 3) | |

The run script `prep` writes (the wrapper, `runwrap.py`) treats the run like
any SIESTA run: it activates the env, runs SIESTA on the ranks and cores per
rank `prep` stated — the launch starts it with them, `-np` / `-omp` from the
job (`submit._run_sh_args`), and computes none — and logs. **Around the engine it runs the run's own Python** — the job's
python runs `mb_vibration.pyz` beside the deck, its plan step before SIESTA
and its record step after it (§ 5.5, § 5.11 § 3) — and marks the call
concluded with the record step's exit status: a failure of the step itself,
never whether a result was written. Whether the run computed its modes is
the run's state, read over its index (§ 5.11 § 7).

### 5.5 The finish — the job derives its own modes

**A vibration calculation ends with its result, on both engines** *(user,
2026-09-28: "if the task is designed for vibration calculation why ... do we
need another step manually to get this information"; "summarize is something
that summarizes results, not another step to get the result")*. PySCF's deck
writes `<label>.spectra.json` itself. A SIESTA vibration's job walks its run
in phases (§ 5.11): the force-constant call's **record step**,
`"$_mb_py" mb_vibration.pyz record fc <deck> <output> --exit <rc>` in the
call's folder with the job's own python, is **the finish's analysis**. It
reads what the call left and the deck that ran it, derives the modes through
the harmonic path the PySCF deck carries (§ 4.5), and writes
`<label>.spectra.json` at the run's root; the run's finish rewrites it once
every call has ended (§ 5.11 § 3). **Its failure is the job's**: the wrapper records the finish's exit
status in `-runN.concluded`, naming the failed finish, and exits with it; the
session log holds its error; and `run_status` reads the job as **failed**
although SIESTA's output ended (`execution/project-layout.md` § 1.6.3) — a
force-constant run without its spectrum has not finished. While the finish
works, and if the job is stopped inside it, the attempt says so too: an ended
output with no marker yet, beside the session log's `finish started:` line,
reads **running**, and **failed** once the monitor saw the process go
(`execution/running-a-job.md` § 4.2). **And it is asked before the run is
paid for**: once the run index is known and before SIESTA starts, the wrapper
asks the bundle whether it loads on the job's python (`$_mb_py mb_vibration.pyz
loads`, which imports the finish with numpy and netCDF4 (§ 6.10.6);
`runwrap._finish_check_block`); a job env that cannot
run it stops there, with a marker saying so (`rc=1 at …; finish cannot load
(mb_vibration.pyz)`), and the attempt reads failed with no run lost. A dry run
has exited before the question. `jobset summarize` derives no run's result
for a vibration: it only summarizes results that exist — here, the
force-constant stages of a displacement sweep (§ 5.9).

**Every input is a record the attempt already holds, read by the reader that
owns it** — nothing is re-derived, and nothing molbuilder resolves on the host
is guessed at on the node:

| input | read from | through |
|---|---|---|
| `SystemLabel`; the species (atomic number per label); the atoms in the deck's order (held first, § 5.2); `FC.First` / `FC.Last` — the atoms the run nudged, read through the one door for engine numbering (`engine_atom_index`); the held atoms are the rest, which in the held-first copy is exactly the set `Geometry.Constraints` names (I6), and a range that is not the trailing run is refused | the deck | the fdf reader, `parse/fdf.py` |
| the axis kinds — which whole-body turns survive (R3) — and the box the run had, its cell | the deck's `engine-offset` block | the block's one reader, `deck_record.extract_engine_offset` — the grammar of every molbuilder block and its one JSON reader, below `script_emit`, which writes them; the box through `deck_record.box_of_run`, the one rule the structure the result states is built with |
| the stationarity criterion (the tolerance the reference geometry was relaxed to, § 5.5), the person's statement (`already_relaxed`), the thermochemistry's temperature (`temperature_K`), the stage's name, and the molbuilder that prepared the deck | the deck's `vibration` block, written by `prep` | its reader beside that one, `deck_record.extract_vibration_record` |
| the force constants | `<label>.FC` | `parse/engines/siesta_fc.py` |
| the reference geometry and its forces (FC step 0); SIESTA's version | the run's output | SIESTA's one reading pass, `siesta_reader` |
| the order back to the user order | the deck's ATOM-METADATA block, its `atom_order` (none for a deck in the user order — [`model/overview.md`](?doc=model/overview.md) § 2.2a, I2, I5) | the block's reader and the order record's, `deck_record.extract_atom_metadata` and `order_record_of` |
| the masses | every atom's, as `prep` stated them — `chemistry.atomic_mass`, the isotope-averaged standard weights (I3) | the deck's `vibration` block, its reader `deck_record.extract_vibration_record`; written into the result as `equilibrium.masses_amu` (§ 6.2) |

**When the finish fails**, the session log names the reason and the
attempt reads failed. Fix the cause — most often the job env, which needs
numpy and netCDF4. The analysis can be run by hand in the force-constant
call's folder, with the job's env active, by that env's own python —
`"$CONDA_PREFIX/bin/python3" mb_vibration.pyz finish <deck> <output>` — which
writes the spectrum beside that call's deck in seconds; the run still reads
failed: its finish never ended, and its root holds no result (§ 5.11 § 7). To run the force constants again, launch the stage again cold
(`--cold`): launched again warm, the run takes over every call its last run
finished (§ 5.11 § 6); cold, every call runs again from the deck, the force
constants from `FC.First`
([`job-system.md`](?doc=execution/job-system.md) § 5.4, *A stage launched
again*); to change it first, go back to the state saved before its group's
first launch and prep it anew. An attempt prepared before 2026-09-28 carries
neither the bundle nor the deck's `vibration` block: prepare the stage again
(never launched), or from the state saved before its group's first launch ([`execution/job-system.md`](?doc=execution/job-system.md)
§ 5.0), and launch it. **A benchmark
trial of a force-constant stage is not finished**: it measures how long a
setting takes under capped SCFs, and modes derived from those would be a
spectrum of nothing, so its job carries no finish and its deck no `vibration`
block.

**What travels, and the env it needs.** The finish runs where molbuilder is
not installed, so its modules travel beside the job as ONE file,
`mb_vibration.pyz` (`runwrap.VIBRATION_BUNDLE`, built from
`runwrap.VIBRATION_COMPANIONS` by the builder the monitor's bundle uses): each
module's own file, imported two ways — from the package, or from the bundle —
like the monitor's readers ([`execution/run-reports.md`](?doc=execution/run-reports.md)
§ 2.3). Its imports are the standard library, **numpy** and **netCDF4** —
which the SIESTA job envs carry for it (`envs/recipes.py`) — and the other
members; the
masses it weights by are the deck's own statement (§ 5.3), so it reads no
mass table *(it read ASE's until 2026-10-10)*.
`prep` writes the bundle beside each phase's deck and marks the stage's
job with it (`Job.finish`, from the deck's own spec); each call's run script
runs its steps; every call is brought a copy (§ 5.11 § 2).

**The API, in layers** — each lower one knows nothing of the one above it:

| layer | module | what it answers |
|---|---|---|
| the math | `spectra/normal_modes.py` | the harmonic path and the steps after it -- the wavenumbers, the display form, the thermo sums and their grid (§ 4.5, § 4.7) -- which both routes call: the PySCF deck imports them from `mb_pyscf.pyz` |
| the analysis, engine-neutral | `spectra/vibrational_analysis.py` | a second-derivative block over the free atoms, the masses, the geometry with its held set and frame → the result (`SpectraResults`): frequencies, both eigenvector forms, the removed motions, the vibrational thermochemistry, the stationarity verdict, and every per-atom row back in the user order through the recorded permutation. Any route that has a block calls it |
| the PySCF route | `spectra/pyscf_vibration.py` | the rules the PySCF script calls inside its run, imported from `mb_pyscf.pyz`: the HOMO and its orbital window, the Hessian with $d\mu/dR$ (`dipole_derivatives`), the GPU array bridge |
| the SIESTA route | `spectra/siesta_vibration.py` | `read_force_constant_run(deck, output)` gathers the inputs above from the attempt the deck sits in; `finish(deck, output)` runs the analysis and writes the file; `main` is the bundle's entry |
| the prose | `spectra/methods.py` | the route's Methods paragraph (`siesta_methods_text`), beside every other Methods sentence |

```text
fdf      = the deck, through parse/fdf.py; its molbuilder blocks through deck_record
perm     = order_record_of(extract_atom_metadata(deck))            # held-first, stated by the deck
fc       = read_fc(<label>.FC)                                     # (n_free, 3, ±, N, 3), eV/Å²
ref      = the first step of the output with forces (FC step 0)    # sorted order, as the deck
H_AA     = mean over ± of fc[a, α, ·, b, β], a, b ∈ FREE           # the central difference
H_AA     = ½ (H_AA + H_AAᵀ);  × Bohr²/Hartree_eV                   # symmetrise; → Hartree/Bohr²
λ, L, patterns = vibrational_modes(H, masses_amu, R_ref, held, axis_kind, cell)
rows → user order through Permutation.map_rows_to_user_order          # the deck's order record, inverted once
write <label>.spectra.json at the run's root:  engine 'siesta', the SIESTA version from the output,
    intensities null, the MO block null, thermo = the vibrational sums (§ 4.7's second regime),
    hessian_scope 'free'|'all', removed_motions, relaxation (the verdict below and the `relax` call's record),
    engine_metadata {fc_file, fc_displacement_ang, fc_range_engine_1based, fc_asymmetry_max_ev_ang2,
    reference_force_criterion_ev_ang}, config {engine, calculation, stage,
    deck {name, sha256, keywords, blocks, pseudopotentials}}
```

The finish refuses — as the job's failure, with the reason in the session log —
a `.FC` whose row count is not $6\,N\,n_{\mathrm{free}}$, a range whose length disagrees
with the free set, an order record that cannot be read (its two directions
not inverse bijections, its key not a sort's), a deck whose free atoms are not one trailing run, a `vibration`
block that states no `temperature_K` (format v1, prepared before 2026-09-28 —
prepare the stage anew from the state saved before its group's first launch) or no
`masses_amu` (format v2, prepared before 2026-10-10 — the same) or a mass count other than its atoms', and an output whose reference step describes other
atoms. The modes are at **Γ** (R3) — the
centre of the Brillouin zone, $q = 0$, where every cell moves in phase: a
force-constant run over the cell as given is the Γ matrix; a phonon
dispersion over $q \neq 0$ is a different feature and is not this one. The
thermochemistry is summed at the template's `temperature_K`, carried in the
deck's `vibration` block (§ 5.3), and records no pressure it did not use: a
SIESTA result has no translational term (§ 4.7). Measured end to end through jobset on this
workstation (`tests/test_siesta_vibration_e2e.py`, SIESTA 5.4.2): CO₂ listed
O O C with the carbon held, so the held-first deck is a 3-cycle of the input →
four modes — a degenerate bend pair moving the oxygens in the (x, y) plane, two
stretches along z — two motions removed (the turns about the carbon, about x
and about y), the oxygens reported as atom indices 0 and 1 and the carbon as atom index 2, the user order — the
spectrum written by the launch itself.

**What the finish judges (built 2026-09-24; in the job since 2026-09-28).** Stationarity: SIESTA
evaluates the undisplaced geometry as its FC step 0 before the first nudge,
and its forces are the first `siesta: Atomic forces` block of the run's
output — printed because every SIESTA deck writes `WriteForces` and
`WriteCoorStep` true, which no form or stage may turn off (`role`,
[`template.md`](?doc=engines/template.md) § 6.4; SIESTA prints neither by
default, and a finish without them failed after the whole force-constant run). The finish reads them (SIESTA's one reading pass, `siesta_reader`, one pass over the run's output for the version, the geometry and the forces)
and writes `relaxation.max_force_eh_bohr` — the largest over the **free**
atoms (R5) — beside `max_force_all_atoms_eh_bohr`, and `converged` judged
against **the force the reference geometry was relaxed to** — the stage's
`relax_force_tol`, the one the `relax` phase runs to, and the template's
when the structure is stated relaxed — resolved by `prep` and
recorded in the deck's `vibration` block, so `prep` and the finish say one
thing; `engine_metadata.reference_force_criterion_ev_ang` records the
number used. *(Until 2026-09-24 the catalogue's general default stood in for
it.)* The positions the projection and the artifact use are the run's own
reference geometry, its FC step 0 (§ 5.11 § 3). The judged
number is the **largest absolute Cartesian component** over the free atoms, the
convention the PySCF deck's own check uses (§ 4.3). Above it the block carries a warning naming the number and
the one remedy for how the geometry was reached — **one text,
`vibrational_analysis.nonstationary_remedy`, which `prep` (§ 5.8) and the
PySCF deck (§ 4.3) write too**, each leading with the step that can always
be taken first — if this stage has been launched, go back to the state saved
before its group's first launch (what a launched stage computes is not prepared again,
[`job-system.md`](?doc=execution/job-system.md) § 5.0; one never launched is
prepared again): when the run relaxed it (the result's `relaxation_stage`,
this stage, whose `relax` call it was), give that stage more relaxation steps
(`relax_steps`) and prepare its group again — its run relaxes first — or
launch it again cold (`--cold`): every call again, the relaxation from the
structure at the steps it has (§ 5.11 § 5, § 6);
when the person stated it relaxed, relax first — or keep the run knowing that.
The finish says it in the session log, and the viewer's relaxation row shows
the force. And the block's honesty about its own numerics:
`engine_metadata.fc_asymmetry_max_ev_ang2` is $\max\lvert H_{ij} - H_{ji}\rvert$ over the
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

**Raman on SIESTA is the optical phase of the one vibration run** (§ 5.11),
the method's § 3
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)):
asked by *Compute Raman* — the setting both engines share (§ 3.1) — for
the modes *Raman for* picks, every mode unless a frequency range is given:
SIESTA's optical calculation at $\pm\delta Q$ along each, and the strength
derived at a laser the person names after the run (§ 5.11 § 10). What its
numbers mean in a cell holding metal — the value at the laser defined, the
static one not — is the method's § 3.1
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)). **The infrared comes from the force-constant run's own
charges** (§ 5.10 ②), asked by *Compute IR*: every atom's Hirshfeld and Voronoi charge at every
nudge, from which the spectrum viewer gives the infrared of the atoms the
person names — the dipole of a molecule in the junction by its atoms' charges
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)
§ 2). Born effective charges from the Berry-phase polarisation are the route
for an insulating crystal, and SIESTA computes them for an insulator only; a
junction is metallic. The molecular-orbital block is absent because a periodic
system has a Fermi level, not a HOMO. The force-constant call relaxes
nothing — the run's `relax` call does, before it, unless the box is ticked
(§ 2.2, § 5.11).

**Why the SIESTA route exists at all — consistency of the potential-energy
surface.** A junction study is a chain: relax → find the vibrations →
displace along one → compute transport. Done in one program, every step sees
the same pseudopotentials, the same numerical orbitals, the same functional,
the same slab and k-points; done in another, the mode displaced along is not a
mode of the system the transport runs through. And the modes differ in the way
that matters — a mode of BDT on gold includes the Au–S interface stretching,
`Au ⟷ S — C₆H₄ — S ⟷ Au`, which is what modulates the electrode–molecule
coupling and which an isolated-molecule calculation cannot produce.

Two levels of that connection — the first built (a mode's frame set,
§ 5.10 ③, and transport at every frame,
[`engines/transport.md`](?doc=engines/transport.md) § 2a.9), the second not:

- **Level one — displace along a mode and look.** $\mathrm{positions}(Q) = R_0 + Q\cdot L$
  swept from negative to positive gives real geometries caught at successive
  points of one vibration; transport at each gives how the current-carrying
  ability changes across it (`Q < 0`: Au–S shorter, stronger coupling). The
  PySCF deck already does the displacement half for its electronic-structure
  probe (§ 4.8). **How far is $Q$**: its reference is physical — the
  zero-point amplitude $Q_{\mathrm{rms}} = \sqrt{\hbar/2\omega}$ and its thermal growth, exactly the
  amplitudes [`web/spectra.md`](?doc=web/spectra.md) § 4.1 derives for the
  animation — and the extent a frame set samples is the person's, stated
  beside that reference (§ 5.10 ③); both pair with the **canonical**
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
density-difference map $\Delta\rho_\nu(r) = \rho(r, +Q_\nu) - \rho(r, -Q_\nu)$ from SIESTA's
density grid at the two points — the discussion's intermediate quantity
before any oscillator strength, which says whether a mode polarises the
molecule, moves charge across the Au–S bond or drives the metal's screening
(V1.26); and the projected density of states, resonance energies and widths
along the mode, each a run of its own kind on that pair, with $T(E, Q)$ the
transport kind on it. Each is a feature to design as one — and how they
combine into a ranking of modes for the transport step is proposed in § 5.10
(drafted 2026-09-28; its decisions are recorded there).

### 5.7 What it costs, by construction

$1 + 6\,n_{\mathrm{free}}$ force evaluations, each a whole-system SCF started from the
reference step's density, and the memory of one SIESTA SCF whatever $n_{\mathrm{free}}$ is.
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
is vibrational-only, that every atom's charge is recorded at every nudge
when *Compute IR* is on (the infrared of the atoms the person names,
§ 5.10 ②) and that Raman comes from the optical calls *Compute Raman* asks
for (§ 5.11 § 10); *Raman for* at `range` with neither bound, or with the
minimum above the maximum, is refused — a range stated nowhere, two bounds
that conflict; the precondition
(§ 2.2): with the box unticked, an info line that the run relaxes first, its
`relax` phase, to
the template's force tolerance, before the force constants; with it ticked, a
**warning** that the frequencies will be off if the structure is not relaxed
at this level of theory, and that the finish measures the reference-step
forces against that tolerance (§ 5.5); the relaxation-record findings of
§ 2.2 when the structure carries one; and the
unconsumed-region-label notice every kind carries. With the box unticked
the relaxation's outcome is the run's own: the finish judges the largest
force on the moved atoms at the force constants' reference step — the
geometry the `relax` call reached — against this calculation's tolerance
(§ 5.5), as information when within it and as a **warning** when
above it, naming the one remedy (`vibrational_analysis.nonstationary_remedy`).
The SIESTA engine validator defers its own *held during relaxation* line on
this kind — one fact, one finding ([`science/validation.md`](?doc=science/validation.md) § 7).

### 5.9 The displacement sweep — how converged the force constants are

**Why** ([`science/normal-modes.md`](?doc=science/normal-modes.md) § 4b.6 C).
A force constant is a finite difference of forces: too small a nudge drowns
in the SCF's noise, too large picks up anharmonic terms. The check is to take
the force constants at two or more nudges and see them agree — the
frequencies, $\omega(\delta) \approx \omega(\delta/2)$, and the matrix they come from, $H(\delta) \approx H(\delta/2)$,
with the matrix's own symmetry $H_{ij} \approx H_{ji}$ as each run's internal
diagnostic. § 9's H₂ ladder is why a finer-mesh stage belongs beside the δ
stages: there, most of the drift was the grid.

**Describing it.** A sweep is a ladder with more than one stage, each with
its own `fc_displacement` (or `mesh_cutoff`) as a stage override. **Every
stage is one run in phases** (§ 5.11); the rule that a later stage takes the
first stage's `relax` call over at its launch, so every stage's force
constants are taken at one relaxed geometry (§ 5.11 § 1, § 6), is **not
built** (plan § 1ae): each stage relaxes in its own run, and two stages at
different geometries the summary refuses, below. Each stage's job writes its own result in its own run
(§ 5.5) — its `<label>.FC`, its output, its `<label>.spectra.json` — the raw
data at that displacement, never merged or moved. **The hierarchical layout only**: the description's reader refuses a
SIESTA vibration described flat (§ 5.11 § 1).

**The summary: `jobset summarize task`** — `summarize` summarizes results that
exist (§ 5.5). On a SIESTA vibration with two or more stages it
reads each stage's newest attempt through the layout's doors, and each
spectrum through the one typed reader. It writes
**`<label>.fc-sweep.json`** at the calculation root (schema
`molbuilder/fc-displacement-sweep@2`, `spectra/displacement_sweep.py`) and
prints the same record as text; the Results tab opens the record at the
calculation root (the `fc-sweep` presenter), which draws every key below. A
stage without a result says why in the words `run_status` gives its attempt
(`execution/running-a-job.md` § 4.2): not launched, queued or running is
**pending**, never a failure; a run that failed, one that finished without
its spectrum (an attempt prepared before its job finished itself), or a
spectrum the reader refuses is **failed**, with the reason. A PySCF vibration
is refused by name — its second derivatives are analytic, so there is no
displacement to sweep.

**Nothing is lost and nothing is copied.** Each stage's files stay where its
run wrote them, and the record names each by its path from the calculation
root; it adds only what the comparison derives:

| the record's key | what it holds | how |
|---|---|---|
| `stages[]` | per stage: its name; the paths of its attempt, its spectrum and its `.FC`; what it varies (its own overrides of the template); the displacement SIESTA used (`fc_displacement_ang`, from the stage's spectrum, which read it from the `.FC` header) and the atoms it nudged, `fc_range_engine_1based` — `FC.First … FC.Last` in SIESTA's numbering, the deck's held-first order (§ 6.2); its asymmetry diagnostic; its stationarity verdict with the largest free-atom force and the criterion; its mode and removed-motion counts; SIESTA's version | read from the stage's spectrum |
| `modes[]` | per mode of the reference stage (the first with a result): its frequency at every stage, the matched mode's index there, the shapes' overlap, the change from the reference, the spread across stages, and a flag when a tolerance was given | modes matched **by shape**: the overlap of the mass-weighted eigenvectors over the free atoms, assigned one-to-one so the total overlap is largest (`scipy.optimize.linear_sum_assignment`) — ranks swap between displacements when two modes are near-degenerate, and matching by rank would compare different motions |
| `force_constants[]` | per stage against the reference: the largest change of any force constant over the block both nudged, in eV/Å², and that change relative to the largest constant | $H(\delta)$ from each stage's raw `.FC`, through the one reader (`parse/engines/siesta_fc.py`) |
| `tolerance_cm1` | the threshold a person gave (`summarize task --tolerance-cm1 X`), or `null` | a mode is flagged when its spread exceeds it; without one nothing is flagged — the numbers are stated and the judgement is the person's |
| `pending[]` | stages whose result is still to come: the stage, its attempt, its state (described; prepared and not launched — *ready to launch* or *waits for*, the ready door's; `queued`, `running`) and the detail of the door that answered | |
| `failed[]` | stages whose run ended without a readable result: the stage, its attempt, its state (`failed`, `finished` without the spectrum, `unreadable`) and why | |

`stages[]` also carries `varies_units` — the catalogue's unit for each value
a stage varies (`fc_displacement` is in Bohr, the Å beside it is the
displacement SIESTA used). A sweep across stages that describe different
atoms, share a directory, or **were measured at different geometries** — their
results' structure hashes (§ 6.2) differ — is refused by name, the last with
how far an atom moved between them: a relaxation's change reported as the
displacement's would be a wrong answer that looks right. *(Schema 9 compares
the structures each stage states, every atom within 10⁻⁶ Å — § 6.10 § 3,
decided 2026-10-14.)* **Two modes a few
wavenumbers apart can mix between stages**: a low overlap there is the pair
turning within its own plane, not a changed motion, and the presenter says
so beside the table.

### 5.10 From a mode to the current — the response screen *(drafted 2026-09-28; ③ built 2.35.0, ② built 2.36.0, the rest decided and not built)*

> **The design for the next programme, decided and not yet built.** The user's
> answers to its decisions are the table at its end (plan **W42**): D1–D5, D7
> and D8 decided, D6 taken as recommended, E3 still owed. Of it sisl in
> every SIESTA env (D5) and the null pressure of a vibrational-only result (E2)
> are built, with M2b‴, the frame set (③) with 2.35.0 and the charges' infrared (②) with 2.36.0. It is written into the contract so that the pieces it
> adds can be checked against the ones that exist; as each is built it moves
> into the section it changes, and this banner goes with the last. The science
> — why the ranking is by the conductance and not by the HOMO–LUMO gap — is
> [`science/normal-modes.md`](?doc=science/normal-modes.md) § 4c.

**The question** *(user, 2026-09-28: "we fix most of the metal atoms, and
leave the molecule and the anchoring Au for vibrational mode calculation with
siesta, but need a way to find the best vibration mode for next step of
transport calculation")*. A junction's force-constant run gives 30–50 modes,
and transport along a mode costs a TranSIESTA device SCF per displaced
geometry, so a few modes are chosen. Science § 4c says what to choose them by —
the change of the conductance a mode's motion causes, $\langle\Delta G\rangle_\nu/G = \tfrac12\,T''_\nu\,\sigma_\nu^2/T_0$
at a stated temperature — and that cheaper electronic numbers can screen for
it. This section says how the tool computes each number. Every piece reuses one
that exists:

| exists | reused for |
|---|---|
| both eigenvector forms and the zero-point amplitude of every mode (§ 6.3) | the frames and the modes' character |
| the frame set — one multi-frame pair for one mode, the mode in the structure's `customized` rows and each frame's node and weight in its own (③; `engines/transport.md` § 2a.9, V1.25) | the one interface between a vibration and every run along its modes |
| the axis rule — a stage carries a sub-level for each axis it varies over (`engines/transport.md` § 2a.11) | a level per frame inside one run, the shared runs once |
| the job's own finish, a bundle that travels (§ 5.5, I22, I23) | each frame's electronic result, written by its own job |
| `summarize` over results that exist — nothing lost, nothing copied, matching by shape (§ 5.9, I25) | the response of each mode across its frames |
| the PySCF probe (§ 4.8) | the free molecule's answer (V1.27) |

**The pieces, cheapest first.** Each is one addition with its own decision.

**① The character of each mode — derived when the result is read** (no
calculation, and no new label). **The anchor is the held atoms** *(user,
2026-09-28: "the anchor is the fixed atoms. we don't specify the anchor, the
atoms is labeled already with fixing or unfixed")*: the free atoms are what
the person let vibrate — the molecule and the gold left free with it — and
the result file already carries which atoms are free, their elements and the
geometry (§ 6.2). Served by `/api/spectra/load` beside
`motion_share_by_element` and never stored (I16):

- the element shares among the free atoms — `motion_share_by_element`, as
  the viewer shows it today (`web/spectra.md` § 4.2);
- `rigid_share` — the part of the free atoms' mass-weighted motion that is
  whole-body: its projection onto the free part's three slides and three
  turns about its centre of mass — the extended molecule sliding or rocking
  as a unit against the held anchor, the molecule's tilt among it;
- `bond_stretches` — for each bond from a free atom to a held one, and each
  bond between two elements inside the free part (Au–S), by the viewer's
  yardstick of a bond (`web/spectra.md` § 4.2): its length, its change per
  unit $Q$, $(L_a - L_b)\cdot\hat e_{ab}$ (a held atom's $L$ is zero), and its r.m.s.
  stretch at the thermal amplitude, in Å.

It says what kind of motion each mode is — internal, at the contact, the
free part rocking against the anchor, the gold moving — a filter and not a
ranking (science § 4c.3, the antisymmetric stretch).

**② The charge each mode moves — from the force-constant run itself**
*(built 2.36.0)* (D3). The force-constant deck asks for the population analysis at
every step when *Compute IR* is on — `Charge.Hirshfeld` and
`Charge.Voronoi`, the 5.4.2 spellings of the deprecation table (§ 5.3).
**Measured on the toy junction** (`claude-transport-walk/frequency/h-vib-q`,
22 H, two free, 2026-10-24): 13 force-constant steps, 13 Hirshfeld and 13
Voronoi blocks — one of each per step, every atom in each — each block
summing to zero within 1.2 × 10⁻⁵ e; a free atom moved off the chain's axis
(along x or y) leaves the charges unchanged both ways, moved along it (along
z) changes them with opposite signs, as the chain's symmetry asks; molbuilder's reader reads the
output without a warning, and the modes are the earlier run's to the last
digit. **Both partitions are read, every
number carries the partition it came from, and the viewer shows Hirshfeld
unless the person picks Voronoi** *(user, 2026-10-24: "both on"; "default to
hirshfeld")* — how the two differ and why Hirshfeld is shown first is
[`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)
§ 1a, and the viewer's switch is [`web/spectra.md`](?doc=web/spectra.md)
§ 3.2.

**The finish stores the charges; what is asked of them is derived when
asked.** The finish reads each step's two blocks through SIESTA's one reading
pass (each step's `charges`, [`model/parse.md`](?doc=model/parse.md) § 4a) and
writes them into the result whole, in the user order through the deck's
order record — `partial_charges` (schema 11, § 6.2): every atom's charge at
every step, by partition, beside each step's displaced atom, axis and side,
and the step $\delta$. Nothing a person chooses is baked into the file. What a person
asks of the charges — for a set of atoms (a label of the structure the
person names, the molecule's say), light polarized along a direction $\hat e$, a
partition — is derived on request by **one function**,
`spectra.infrared.infrared_view`
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)
§ 2): per mode, the set's dipole change $\partial\mu/\partial Q$ (a vector, so any $\hat e$ is a dot
product after the run), its infrared strength along $\hat e$, $A(\hat e)$ — or, where
the result holds no atom, the orientation average, a free molecule tumbling
(science § 2.3) — and its rank,
and the charge the mode moves into or out of the set, $\partial q_{\mathrm{set}}/\partial Q$. Two doors
ask it: the spectrum viewer's load (`web/spectra.md` § 3.2) and a person's
script (`molbuilder.postprocess.infrared_view`). A result written before
2.36.0, or from a deck that asked for no charges — prepared before 2.35.2,
or with *Compute IR* off (§ 3.1), its `phase_ir` *not requested* — states
none — `partial_charges` null — and the view says why, in the result's
own facts (`spectra.infrared.no_charges_reason`); a PySCF result
states none, its infrared being the dipole's own (§ 4.6). It is the periodic
analogue of PySCF's $\partial\mu/\partial R$ (§ 6.4): a molecular dipole is undefined in a
periodic metal (science § 4b.7), the charge on a set of atoms, by a stated
partition, is not. *(Found while drafting, fixed
2026-09-28: the SIESTA deck's commented hint for the population analysis
spelled the retired `WriteMullikenPop`; it names 5.4.2's `Charge.Mulliken`
and `Charge.Mulliken.Format` now, `siesta/input.py`.)*

**③ The frame set — the generator** *(built 2.35.0)* (V1.25; D6, D7; one
mode per set, the user 2026-10-09; its extent the user's, 2026-10-24). One generator writes
**one** multi-frame pair for **one** mode *(user, 2026-10-09: "i would rather
let one multi-frame focus on one mode ... each vibration mode will have a
separate prep and run as different task")*: frame 0 the geometry the Hessian
was taken at, then the Gauss–Hermite nodes of a Gaussian along the mode —
three by default, five on request ([`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) § 5),
each pair of nodes nearest first and its positive side first:

$$
\begin{aligned}
R_A(q_j) &= R_A^0 + q_j\,L_A , & q_j &= \sqrt{2}\,\sigma\,x_j \quad (x_j \text{ the rule's nodes})\\
\sigma &= q_{\mathrm{out}} \big/ \bigl(\sqrt{2}\,x_{\mathrm{out}}\bigr) , & q_{\mathrm{out}} &= d_{\max} \big/ \max_A \lvert L_A\rvert\\
R_F &\quad \text{unmoved}
\end{aligned}
$$

**Its extent is the person's: the maximum displacement $d_{\max}$**, in Å — the
largest atom's move at the outermost frames *(the user, 2026-10-24: max
displacement, over a temperature; and 2026-10-09: "max displacement is a
parameter we feed to generate the multi-frame data and is included in the
meta data")*. It sets the spread $\sigma$; each frame's weight is the rule's own —
$\tfrac23,\ \tfrac16,\ \tfrac16$ for three — so the frames average over a Gaussian of the spread the
person chose. **No temperature is stated**: $\sigma$ sits beside the mode's
zero-point amplitude $Q_{\mathrm{zp}}$, which every result states — $\sigma = Q_{\mathrm{zp}}$ is the
zero-point motion's spread, and the thermal one at a temperature $T$ is
$Q_{\mathrm{zp}}\sqrt{\coth(\hbar\omega/2k_BT)}$ (science § 2), so a person who wants the thermal
average at $T$ chooses the $d_{\max}$ that gives it. $d_{\max}$ is stated by the
person every time: a value stated nowhere is refused, never defaulted.

**What it reads, each from its owner:**

| what | from |
|---|---|
| the mode — `index_1based`, its frequency, $Q_{\mathrm{zp}}$, the canonical eigenvector $L$ over the free atoms (§ 6.3's pairing rule) | the vibration result, through its reader (§ 6.8); one that states its masses (schema 7 or later) |
| frame 0 — the Hessian's geometry in the user order, its labels, held atoms and box | the result's `structure` (§ 6.10.1) |
| each atom's mass | the result's `equilibrium.masses_amu` |
| frame 0's `info.calculation` — the settings a transport citation takes its own from (`engines/transport.md` § 3.1, § 2a.7) | **the record of the relaxation frame 0's geometry came from** *(the user, 2026-10-24, on the recommendation)*: where the vibration's own run relaxed it — its `relax` call (§ 5.11 § 1) — that run's settings, the vibration's own deck's record, through the run door's view (`runs.view_of`, `parse.contract.contract_of`), the one template every phase of the run reads; where the structure was stated already relaxed (§ 2.2), the record that structure states — its `info.calculation`, written when it was saved from its optimization — read from the calculation's own structure beside it. A stated-relaxed structure that states none gives a set with none, and the generator says so: such a set is no transport citation until its relaxation is recorded. A PySCF deck records none either, and the same is said. *(Measured on AuBDTAu, 2026-10-24: its vibration ran at a 1×1×1 k-grid and 300 Ry, its relaxation at 5×5×1 and 400 Ry; a transport citing a set stamped with the vibration's settings defaulted to the cheaper ones.)* |

**What it writes** — through the structure's own doors
([`model/structure.md`](?doc=model/structure.md) § 2.2d–2.2f: `with_frames`,
`set_channel`, `set_customized`, `set_info`), in § 2.2f's vocabulary, which
this section does not repeat: the mode, its frequency, $Q_{\mathrm{zp}}$, $d_{\max}$ and $\sigma$
on the structure's rows, with the rule the frames sample it by —
`sampling_rule` `gauss-hermite`, `sampling_nodes` — and the masses' source,
`mass_source` `standard atomic weights`; each frame's displacement, largest
atom move, $q$, $q/\sigma$ and weight on its own; **every row with its
meaning as its `note`**, the vocabulary's words, so the pair says what its
weights average and what its coordinate is measured with wherever it is read
([`model/structure.md`](?doc=model/structure.md) § 2.2f); every atom's mass on
the `mass_amu` channel; the vibration run — its folder from the projects
tree — and its result's sha256 in `info.vibration`. One pair, `.xyz` and `.molstruct.json`, through the
structure's one writer; **a pair already at that path is refused**, never
written over: a transport calculation citing it is pinned to it.

**It keeps what a transport citation checks**: the same atoms, order, labels
and box in every frame, and the held atoms where frame 0 has them — $L$ is
zero on every held atom. **An electrode atom left free moves**, and its
frames are then no transport citation (`engines/transport.md` § 2a.9): the
set is written, and the generator says which electrode atoms move and that
transport refuses such a set.

**Two doors, one generator** (`frameset_from_mode`):

- **the command line** — `molbuilder frameset from-mode RUN --mode N
  --max-displacement D [--nodes 3|5] --out PAIR.xyz`: `RUN` the vibration
  run's folder and `PAIR.xyz` the pair, each read from the projects tree as
  `--bundle` is, the run read through the run door; it prints each frame's $q/\sigma$,
  largest atom move and weight, and where the pair is;
- **the spectrum viewer** ([`web/spectra.md`](?doc=web/spectra.md)) — on the
  selected mode, *Write frame set…*: the maximum displacement (with the mode's
  largest zero-point atom move beside it, from the result's
  `zero_point_displacement_ang`, and the reference at the temperature the
  load door computed it at, [`web/spectra.md`](?doc=web/spectra.md) § 4.3),
  three or five frames, and where to save it in the tree; the same generator
  answers, and the pair opens where a structure opens.

**Another mode is another set**, cited by its own transport calculation; how
the modes' results are weighed against each other is the person's *(user,
2026-10-09: "it is cleaner and more focused to let user to post-process the
results from different vibrational mode and decide how they can be weighed to
give an overall impact")*.

**④ The electronic response on a frame set — a run of its own kind** (D4, D5).
§ 5.6 already says so: "each a run of its own kind on that pair". Proposed: a
calculation kind `electronic` on SIESTA — one SCF per frame, in the cell and at
the settings of the vibration the frames came from, so the modes and the screen
share one model (taken the way transport takes its shared values from the
relaxation it cites, `engines/transport.md` § 2a.7), the frames on the axis
rule's sub-level inside one run (`01_scf/run-0/f000/`, `f001/…`, as transport's
frames sit, `engines/transport.md` § 2a.11). Each frame writes its
own result through a finish — § 5.5's pattern, its bundle under I23 — as
`<label>.electronic.json`:

| key | holds |
|---|---|
| `fermi_energy_ev` | $E_F$ |
| `charges` | the free atoms', and each element's among them: the Hirshfeld, Voronoi and Mulliken charge (e) |
| `pdos` | the DOS projected on the free atoms, and per element, on its energy grid relative to $E_F$, per spin — the raw curves — with the broadening they were computed at |
| `resonances` | the peaks of `pdos` in a window around $E_F$: position − $E_F$, width, weight, and each element's share of the peak; the nearest below and above $E_F$ named |
| `levels` | the free atoms' own states (the extended molecule's MPSH, science § 4c.2) in a window around $E_F$: energy − $E_F$, each state's weight per element, and its coefficient vector, so a level can be followed across frames — *only with sisl in the job env, D5* |
| `frame` | the frame's token and its `customized` rows (③), copied from the pair, so the file says where it stands |

Optionally the density grid for $\Delta\rho_\nu(r)$ (V1.26), which then rides on the same
frames. The Fermi level and the charges are text in the output, read by the
SIESTA reader the finish already carries, and `.PDOS` is XML the standard
library reads; only the levels need the Hamiltonian and overlap, SIESTA's
binary `.HSX`, whose reader is sisl (D5) — without it the record carries
the PDOS peaks and the charges, and no levels. Every new
keyword — `%block ProjectedDensityOfStates`, `PDOS.kgrid.MonkhorstPack`,
`SaveHS`, `Charge.*` — is checked against the manual-derived table before it
is written (I21).

**⑤ The response across the frames — `summarize`** (§ 5.9's pattern). **On
transport it is the transport record's** — each frame's curve beside its rows,
the mode's average and its change from the base frame, one mode a set
(`engines/transport.md` § 2a.12, built with Q17, plan § 5z), and every frame's
values in the data file, from which a slope or a ranking is the analysis's;
what follows is the `electronic` kind's, when it is built — its derived
numbers (the slope, the curvature) to be revisited before it is, under the
ruling that the record saves each frame's values aligned with its rows and
leaves what is derived across them to the analysis (user, 2026-10-10).
`jobset summarize task` on an `electronic` calculation whose structure is a
frame set writes `<label>.mode-response.json` at the calculation root — a summary of results that exist, each frame's file named by its path and
none copied (I25's rule). It reads each frame's $Q$ and weight from the frame
rule, so the same writer serves both kinds and neither needs to know what a
mode is:

- per mode and per scalar $X$ — $E_F$; the free atoms' charge and each
  element's; $\mathrm{PDOS}_{\mathrm{free}}(E_F)$; each followed level's $\varepsilon - E_F$ (~~on transport
  $T(E_F)$ and $G$~~ — transport's values are the transport record's, each
  frame's aligned with its rows, and nothing is derived across them there,
  `engines/transport.md` § 2a.12; struck 2026-10-18) — the value at every node, the slope $X' = (X_+ - X_-)/2h$, the curvature
  $X'' = (X_+ + X_- - 2X_0)/h^2$ with $h$ the node's $Q$, the thermal average
  $\langle X\rangle = \sum_j w_j X_j$ and $\langle X\rangle - X_0$, and the three-node against the five-node
  average where both exist;
- the levels followed across frames one-to-one by the overlap of their
  coefficient vectors in the overlap-matrix metric — § 5.9's assignment, on
  orbitals instead of modes — so $\varepsilon_{\text{HOMO-like}}(Q)$ follows one state even where
  two cross;
- ~~on transport, the thermally averaged $T(E)$ of each mode, and per frame,
  where one resonance fits, its Breit–Wigner $\varepsilon_r$ and $\Gamma$~~ and ~~the
  rigid-shift estimate $T'' \approx g_\varepsilon^2\,\partial^2 T_0/\partial E^2\big\rvert_{E_F}$ wherever a transmission at
  the base exists~~ — struck 2026-10-18: a transport quantity across frames is
  the analysis's (`engines/transport.md` § 2a.12, science § 8); the resonance
  fit and the rigid-shift estimate, if wanted, are brought as an analysis of the
  data file, never as rows of this summary;
- frames without a result as `pending`, never a failure; flags only against a
  tolerance the person gives.

**⑥ One table, joined by the mode.** The response record's presenter is the
ranking table: per mode the frequency and the character (① — read from the
vibration run the frame rule cites) beside the level shift at $\sigma$ (meV), $g_\nu$
(meV) and $\langle\Delta G\rangle/G$ at the stated temperature, marked *explicit* or
*estimated*; sorted by any column, with no priority label unless the person
gives a threshold (§ 5.9's tolerance rule). A SIESTA mode then gets what a
PySCF mode has on its *Electronic structure* tab — the levels at $-Q$, $0$,
$+Q$, joined state by state — drawn from the response record instead of the
probe. The spectrum viewer itself shows ① and ②.

**The PySCF arm** (V1.27, D8). The probe moves to the same nodes — physical
amplitudes at the stated temperature, three points — so its SCFs give each
orbital's slope and curvature, and $g_\nu$ is written. V1.24's matching maps the
free molecule's modes onto the junction's by the overlap of their shapes over
the molecule's atoms, so the isolated-molecule screen (the discussion's first
stage) points at junction modes.

**What it costs** (R8). ① and ② nothing beyond a read and an output switch.
④ for every mode at three nodes: $2\,(3\,n_{\mathrm{free}} - n_{\mathrm{rigid}}) + 1$ whole-junction
SCFs, the order of the force-constant run's own $1 + 6\,n_{\mathrm{free}}$ (§ 5.7) — so
screening every mode is affordable wherever the modes were. The transport
frames: one device SCF each, for the modes chosen.

**Validation, before anything is claimed.** H₂ first, where the answer is known
independently: SIESTA's molecular levels of H₂ in a box against PySCF's
orbitals along its one stretch. Then the carbon-chain junction of the transport
walk; then Au–BDT–Au. Every record carries three built-in checks: the
three-node average against the five-node one; a symmetric junction's
antisymmetric modes, which must give $X_+ = X_-$ — their $X'$ is the record's
noise floor, as the asymmetry is the force constants' (§ 5.9); and the
rigid-shift estimate against the explicit curvature.

**Decisions — the user's** (plan W42 carries them):

| # | the question | recommended | why |
|---|---|---|---|
| ~~D1~~ | which atoms every projection is over | **decided 2026-09-28**: the free atoms — the anchor is the held atoms, and nothing new is labelled *(user: "the anchor is the fixed atoms. we don't specify the anchor")*; per element within them | the structure already says which atoms are held, and the result already carries it |
| ~~D2~~ | how the result knows the regions | **not needed**: with the projection over the free atoms, the result already carries everything ① reads — the free and held atoms, the elements, the geometry | |
| D3 | the charge each mode moves (②) | **decided 2026-09-28: yes** — on for every force-constant run once SIESTA is measured to print the populations at every step, *"and result presentation should include the correct data and UI design to present it"*; **the switch on since 2.35.2, measured 2026-10-24; stored and shown since 2.36.0** (②) | one switch and no extra SCF |
| D4 | where the electronic frames run | **decided 2026-09-28: yes**, a calculation kind of its own, `electronic` — *"make it clear the difference between pySCF and siesta, from setup and calculation and presentation"* | what § 5.6 already says; it serves $\Delta\rho_\nu$ (V1.26) too, and needs no leads |
| D5 | sisl in the SIESTA job envs | **decided 2026-09-28: in every SIESTA env**, through the env system's recipes — *"so it is always available when we need them"* (installed with M2b‴, the milestone that retired the unit-height spectrum — § 10's sisl row) | the levels are the junction's HOMO and LUMO, and sisl is SIESTA's own reader of the file they come from |
| D6 | one frame set for many modes | ~~yes — V1.25's "for a named mode" becomes "for named modes"~~ — **reversed by the user, 2026-10-09: one mode per set** (*"i would rather let one multi-frame focus on one mode ... each vibration mode will have a separate prep and run as different task"*); how modes are weighed against each other is the person's post-processing | ~~the base, the leads and the seed once~~ — each mode's set its own transport calculation |
| D7 | the amplitude rule — where the displaced structures the electronic and transport runs compute on sit (not the animation) | **decided 2026-09-28, its input the user's 2026-10-24**: the Gauss–Hermite nodes of a Gaussian along the mode, three by default, its spread set by the person's maximum displacement — the largest atom's move at the outermost frames — and no temperature stated (§ 5.10 ③) | the same frames give the curvature and the average (science § 4c.5); the zero-point and thermal spreads are what a person compares $\sigma$ with |
| D8 | the PySCF probe — how far its per-mode orbital check pushes the molecule (not the animation) | **decided 2026-09-28**: the same nodes; `electronic_structure.amplitude_ang` gives way to the node record | one amplitude rule on both engines, and a curvature needs frames above the noise |

### 5.11 One run in phases — relaxation, force constants and Raman, each SIESTA call a point of the run

Built 2.42.0–2.44.0: the run and its walk, its records, its view and
export. **Not built**: a displacement sweep's later stages taking the first
stage's `relax` call over (§ 6), and a transport sweep's one start and one
finish (§ 7a).

> **The user, 2026-10-25**: *"i don't see why pyscf can do all this in one
> run, while you struggle so much. there is no intrinsic difference"*; *"i
> choose B"* — one run for the relaxation, the force constants and Raman,
> PySCF's shape; *"consolidate how these steps are joint, and how output
> record/log are correctly unified to save in one place for parser"*; *"it
> should share the same structure as pySCF results … keep all consistency"*;
> *"make sure the monitor probe etc should be integrated correctly"*; and
> before that: warm start and continue, status, every SIESTA datum in a
> structured NetCDF with its meaning and parameters beside an index, the
> shell and Python each doing their part, every run's normal records kept,
> each deck with the right labels and held atoms, the broadening chosen when
> the result is presented. The science is unchanged:
> [`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md)
> § 3. The relaxation is the run's first phase, as PySCF's Phase 0
> is (§ 4.2). **The shape is hierarchical only**: three SIESTA calls in one
> directory would write over each other's `.DM` and `.XV`, so the kind fixes
> it (its SIESTA part in the kind table, `execution/architecture.md` § 3.3): the description's reader refuses another shape,
> and Task setup's shape card shows it fixed, in one sentence
> ([`web/task-setup.md`](?doc=web/task-setup.md) § 4).

#### 1. The shape — PySCF's run, in SIESTA's calls

**One stage, `freq`, on both engines: one prep, one launch, one run, one
job.** PySCF's run is one script whose phases call PySCF inside its own
process (§ 4.1). SIESTA is a separate program, so each of its phases is one
or more SIESTA calls, **each call a point of the run** — the folder shape and
the readers a run of points has (`engines/transport.md` § 2a.11) —
walked in order by the run's walk, with molbuilder's Python between the
phases. Prep fixes every setting in each phase's deck; the job works out only
coordinates, as PySCF's script does (`mol.set_geom_`), and SIESTA reads them
from a file beside the deck (`%block AtomicCoordinatesAndAtomicSpecies <
coordinates.fdf`, libfdf's own syntax — verified, § 12).

| phase | PySCF, in its process | SIESTA, its calls, each a point | runs when |
|---|---|---|---|
| `relax` | Phase 0, geomeTRIC (§ 4.2) | one relaxation — `relax_type`, `relax_steps`, `relax_force_tol`, the held atoms in `Geometry.Constraints` | the structure is not stated relaxed (`already_relaxed`, § 2.2) |
| `fc` | Phase 1–2: the equilibrium SCF, the Hessian (§ 4.3–4.5) | one force-constant run (§ 5.3–5.4), every step's charges with it when *Compute IR* is on; then Python: the modes (§ 5.5's analysis) and the infrared's charges | always |
| `optical` | Phase 3, the polarizability sweep (§ 4.6) | one single point with the optical calculation per frame: the equilibrium, then $+\delta Q$ and $-\delta Q$ along each mode *Raman for* picks | *Compute Raman* is on (§ 9) |
| the finish | the script writes the result | Python: the result pair and the data file | always |

The settings are one stage's, so every phase of **one calculation** computes
with one model — the relaxation at the basis, functional, mesh and k-grid of
the modes it serves (L8's mismatch on AuBDTAu, relaxed at 5×5×1 and 400 Ry
and vibrated at 1×1×1 and 300 Ry, cannot arise in one stage). A
**displacement sweep** (§ 5.9) is the one case that measures several models
at one geometry, on purpose: its later stages are to take the first
stage's `relax` call over — **not built**: each stage relaxes in its own
run, and § 5.9's summary refuses two geometries (§ 6).

#### 2. On disk — the stage, and one run holding every record

Every name is the run's grammar's (`job-contracts.md` § 2.2a): the stage
segment is the token, `01_freq`; the walk's files are a run of points'
(`runfiles.GroupNames.of_points`, `<stem>-sweep`).

```text
<calculation>/
  task.json  job-request.json  job-set.json  prepared-with.json
  01_freq/                                  THE STAGE -- prep's, copied into each run's points
    relax/    <label>_01_freq.fdf  <label>_01_freq.run.sh  coordinates.fdf  <label>_01_freq.molwatch.log
    fc/       <label>_01_freq.fdf  <label>_01_freq.run.sh  coordinates.fdf  …
    optical/  <label>_01_freq.fdf  <label>_01_freq.run.sh  coordinates.fdf  …
              each phase's deck, every setting, by the one deck writer; its run script, rendered
              for the phase; the structure's coordinates (§ 4); the progress-log seed; and
              mb_vibration.pyz -- the job's Python: plan, record, finish -- and mb_monitor.pyz
    <El>.psml                               the pseudopotentials, brought into every call
    launch/   <label>_01_freq-sweep.sbatch  the header (prep); the walk script and its log (launch)
  01_freq/run-0/                            THE RUN -- one job; a run of points' container,
    run.json                                  its one launch record
    <label>_01_freq-run0.phases.json        THE INDEX: every call -- what, from what, how it ended;
                                              the finish's entry
    notified.json                           the run's notification record (§ 7, § 7a)
    <label>_01_freq-run0.monitor.log        the run's own monitor log: the notify verb's lines,
                                              the run speaking as one run (§ 7a)
    <label>.spectra.json  <label>.spectra.nc the result: the force constants' record step's, then the finish's (§ 8)
    <label>.optical.nc                      every optical call's values, rewritten as each ends (§ 8)
    relax/                                  A POINT -- a run folder, as a transport point is:
      calcdir.json                            its place record (run, below run-0)
      <label>_01_freq.fdf  coordinates.fdf    the deck (the stage's copy), its own coordinates
      <label>_01_freq.run.sh  mb_monitor.pyz  its run script and monitor (the stage's copies)
      <label>_01_freq-run0.out  …scf-timing.log   its output and SCF timing
      …runwrap-<stamp>.log  …monitor.log  …util.csv  …-run0.concluded   its own records
      <label>.XV  <label>.DM  <label>.CG …    SIESTA's
    fc/                                     … <label>.FC  <label>.FCC  <label>.DM
    optical/                                a container (stamped), the optical calls below it:
      f000/  f001/  f002/  …                … <label>.DM (the seed)  <label>.EPSIMG  <label>.contrib.EPSIMG
```

**Each call has its own folder** because SIESTA names its files by the
label: two calls in one folder would write over each other. **Each call is a
run folder** (`calcdir.json`, role run, below the run's `run-<n>/` — the
level `calcdirs.where` reads as a point; `optical/` between them a container,
stamped as every container is, invariant 6b), so everything a run has, it
has: its own run script, monitor, session log, SCF timing log, retry record
and conclusion. **Once per run, at its root**: the launch record, the index,
the result pair and the data file. `coordinates.fdf` is a fixed name on no
label: a row of the catalogue (`runfiles.fixed`), so the file card and the
checkpoint know it.

**Who makes each folder and file** *(the user, 2026-10-25: "who take
charge in construction of the f* dirs")* — **every hand-over is launch's,
decided on this machine and recorded in the ledger, as every hand-over is**
(`job-system.md` § 5.4, *What a run takes from other runs is one
hand-over*); the node makes only what cannot be known before the force
constants exist:

| who | runs on | makes |
|---|---|---|
| **prep** | this machine | `01_freq/` and one folder per phase the stage's settings turn on (`vibration_phases.phases_on`: `relax/` unless the structure is stated relaxed, `fc/`, `optical/` when *Compute Raman* is on) — each with that phase's deck, its run script, the structure's `coordinates.fdf` and the progress-log seed; the bundles; `launch/` with the header; the stage's first run, `run-0/`, opened and stamped as every prepared stage's is (`materialize.open_sweep_run`: the run, its `optical/` container and the calls every run of it holds — `relax/` when on, `fc/` — each with its phase's files, as a transport prep opens its points) |
| **launch** | this machine | `run-<n>/` — the first launch takes the `run-0/` prep opened, a launch again opens the next through the same opener (`materialize.open_sweep_run`) — and its `run.json`; the walk script and its log in `launch/`; **the call folders it can know, through the one opener** (`open_run` + the copier, § 3): `relax/` and `fc/` on every launch, and on a warm launch every call taken over from the last run (its index names them, the optical ones included) — each taken-over call copied whole, its `.continued-from` written; a `relax` call not done, its warm files copied in (§ 6); the index's first entries, these calls |
| **the plan step** — `mb_vibration.pyz plan <phase> <deck>`, Python, before each call's engine | the node | the call's coordinates where they are another call's result — the force constants' from the `relax` call's output — and its index entry; the run's `start` (§ 7a) |
| **each call's run script**, bash | the node | in its own folder: the output, the SCF timing log, the session log, the monitor log and `util.csv`, the retry record, the conclusion; SIESTA's own files |
| **the record step** — `mb_vibration.pyz record <phase> <deck> <output> --exit <rc>`, Python, after each call's engine, success or failure | the node | the call's ending and facts into the index; the force constants' analysis — the result pair at the run's root — and the optical calls planned, each folder through the opener's travelling core (`calcdirs.record`, the copier) from the stage's `optical/` folder with its `coordinates.fdf`; an optical call's $\varepsilon_2$ and facts into `optical.nc`; the finish, once no call is left to run |
| **the finish** — run by the record step that leaves no call to run | the node | the result pair, final; its own entry in the index — started, ended, exit; the run's `finish` (§ 7a) |

**Where each datum is stored** — nothing derived in any file; `raman_view`
derives from `optical.nc` on request (§ 10):

| data | file | where |
|---|---|---|
| SIESTA's raw files of a call | `.out`, `.FC`, `.DM`, `.EPSIMG`, … | the call's folder, never moved |
| what each call is, from what, how it ended; the finish | the index `<label>_01_freq-run<N>.phases.json` | `run-<n>/` root |
| the notifications sent for the run | `notified.json` | `run-<n>/` root |
| modes, charges, the relaxation record, Raman's citation | `<label>.spectra.json` + `.nc` | `run-<n>/` root |
| every optical frame's $\varepsilon_2(\omega)$, every parameter, its mode, side, $\delta Q$, positions, SCF facts | `<label>.optical.nc` | `run-<n>/` root |
| the walk's log | `<label>_01_freq-sweep.log` | `01_freq/launch/` |

#### 3. How the phases are joined — the walk, shell and Python

```text
launch/<label>_01_freq-sweep.run.sh       the run's WALK (bash), launch's -- the walk script every
                                          run of points has (`submit._walk_script`); it activates no
                                          environment and runs no python (the two-layer model,
                                          `job-system.md` § 6): the job's Python runs INSIDE each
                                          call's run script, with the job's own python
  export MB_JOB_STARTED (run here: now)   the run's start, once, for every call (§ 7a)
  run_member freq@relax run-0/relax bash <label>_01_freq.run.sh --run 0 …
  cp run-0/relax/<label>.DM run-0/fc/ && run_member freq@fc run-0/fc …
                                          each call's SEED, copied by the walk just before it runs
                                          (the `along` mechanism a bias sweep uses), from the
                                          PHASES row: fc from relax, every optical call from fc --
                                          never by a Python step
  for d in run-0/optical/*/; do           THE LOOP STEP -- the walk's step for the calls planned in the job: the
      [taken over]  && continue             folders launch could not know, walked as they stand;
      cp run-0/fc/<label>.DM "$d"/ && run_member "freq@optical@$(basename "$d")" "$d" …
  done                                    needs fc; one call failing stops no other
```

**Each call's run script** is prep's SIESTA run script for its phase — the
env, mpirun with the probed ranks and binding, MaxWalltime from what is left
of the job's wall, the output teed with its SCF timing log, its own monitor
(a member's, § 7a), the ending judged (`_mb_ending`), an SCF that did not
converge retried in place from its own density, its conclusion — rendered
with the phase's `resumes` and **the phase's two Python steps**, run with the
job's python beside the deck as a finish is (`runwrap`, a wrapper's
`steps`; `Job.finish` names the bundle, `mb_vibration.pyz`):

| step | runs | does |
|---|---|---|
| `plan <phase> <deck>` | before the engine (after the bundle's load check); its failure starts no engine, and the record step is told it | the call's coordinates where they are another call's result — `fc`'s from the `relax` call's output; an optical call's were written when it was planned (below) — and the call's index entry: its deck's and coordinates' sha256, where the coordinates came from, its seed; `notify start`, which the verb sends only when the run's notification record holds no start (§ 7a) |
| `record <phase> <deck> <out> --exit <rc>` | after the engine, **success or failure** (after the retries are spent; its failure is the job's, as a finish's is) | the call's ending, exit and verdict (what its output says it converged, § 6) into the index; `relax`: the relaxation record; `fc`: the finish's analysis, the selection (`select_modes`), each picked mode's $\delta Q$, and **the optical calls planned** — each folder opened through the opener's travelling core (`calcdirs.record`, `planned.bring` from the stage's `optical/`, the `optical/` container stamped), its `coordinates.fdf` ($R_0 \pm \delta Q\cdot L$, `normal_modes.displaced`), its index entry — then the result pair at the run's root, written once: `phase_raman` *running* where calls were planned; where none were, *not requested* with *Compute Raman* off and *complete* where Raman was asked and its range kept no mode — the Methods paragraph naming the range; an optical call: its facts into the data file (§ 8); **the finish** once no call the walk may still run is left — every call listed has ended, or needs a call that did not end done (§ 5) — the result pair final where the force constants ran, the finish's entry, `notify finish` with the run's state from the one rule (§ 7) |

No step knows whether it is first or last: the index and the notification
record decide, so a warm relaunch that runs three optical calls finishes
after its third and sends one `finish`. **The shell runs SIESTA, always
through a run script prep rendered; Python decides coordinates and writes
records.** No setting is chosen in the job: each is in the deck prep wrote.
The walk's steps name what each needs (`needs`): `fc` needs `relax`, the loop
step needs `fc`, no optical call needs another (§ 5) — rendered from the
`PHASES` table; a bias sweep's walk names its needs the same way (each
point, the point before it in its frame).

| phase | its coordinates | its seed | `record` adds to the index |
|---|---|---|---|
| `relax` | the structure, the held-first copy — prep's file, copied with the folder | none; launched again not done, its own warm files (§ 6) | its relaxation record — steps, largest force, the fingerprint of the geometry reached — from its output (`parse.contract.relaxation_of_output`); the geometry itself the `fc` call's plan step reads from that output |
| `fc` | the relaxed geometry, from the `relax` call's output; the structure itself when stated relaxed | `relax`'s `.DM`, copied by the walk | the number of modes (the analysis, § 5.5, writes the modes and the charges into the result at the run's root); one entry per optical call it plans, for the modes *Raman for* picks (`select_modes`, § 3.1, § 5.11 § 4), each with its mode, side and $\delta Q = \delta / \max_k \lvert L_k\rvert$, $\delta$ the `.FC` header's displacement (one source, the one the finish reads) |
| `optical` | frame 0 the force-constant run's reference geometry; then $R_0 \pm \delta Q\cdot L$, the displacement rule `mode_frames` uses (one function for both) | the force-constant run's `.DM` — its reference density, from which every displacement of that run started (§ 5.3; verified: one SCF iteration at frame 0, § 12) — copied by the walk's loop step | $\varepsilon_2(\omega)$ and the output's facts into the data file |

**Data, not code**: the phases table `siesta/vibration_phases.PHASES` —
each row its name, its deck part, the item that turns it on, its
coordinates' source and its seed (which call's file), what makes a call done
(§ 6), its warm-files section and the phases it needs.
Prep renders the decks and run scripts from it, launch writes the walk and
opens the known calls from it, the record step and this machine's done door
judge a call by it, and the state is read by it; what each phase's record
step writes is its own branch of the verb (`record_call`).

**What the plan and record steps call** *(the user, 2026-10-25: "is that a
unified api or this task specific code?")* — every door the finish and prep
use, in the bundle; nothing of their own but the verb that walks the table:

| what it does | module | who else calls it |
|---|---|---|
| reads the previous call's output — the relaxed geometry; the force-constant run's reference geometry, charges, version | `parse/engines/siesta_reader.py`, SIESTA's one reading pass | every reader of a SIESTA output |
| reads `.FC` → the Hessian, and $\delta$ | `parse/engines/siesta_fc.py` (`read_fc`, `hessian_from_fc`) | the displacement sweep (§ 5.9) |
| the modes | `spectra/vibrational_analysis.py` → `normal_modes.vibrational_modes`, the analysis the finish runs | the PySCF route, its `normal_modes.vibrational_modes` (§ 4.5) |
| which modes | `spectra/selection.select_modes`, the one selector, on *Raman for* as the deck's `vibration` block carries it (§ 4) | the PySCF route; it travels in both bundles |
| the frames $R_0 \pm \delta Q\cdot L$ — $L$ over every atom, zero on held, $\delta Q$ from the largest atom's move | `normal_modes.displaced` | the frame generator (`mode_frames`, § 5.10 ③) |
| the coordinates file | `siesta/deck_lines.coordinates_rows` | every SIESTA deck (`siesta/input.py`) and TranSIESTA's (`transport/transiesta.py`) |
| a call folder: made, stamped, the phase folder's files copied | the opener's travelling core — `calcdirs.record` with `paths`, and the copier `planned.bring` | `materialize.open_run` and `bring_files` on this machine |
| every file's name — the seed, the output, the index | `runfiles.compose` / `RunNames` | every writer and reader of a run's files |
| the phases table, the index, the state rule | `siesta/vibration_phases.py` — data | `status`, launch, prep, the Results tab (§ 7) |

The verb itself — `plan` in `spectra/siesta_vibration.py`, the SIESTA
route's job-side module, where the finish lives — reads the call's `PHASES`
row and calls these doors: the counterpart of the PySCF script's phase code
(§ 4.1), one module reading a table rather than text generated per deck.

#### 4. The one deck writer, every frame — nothing is a template

Each phase's deck is written at prep by `siesta/input.spec_for(…,
calculation="vibration", part=)`, the writer of every SIESTA deck, from the
held-first copy (§ 5.2): the labels, the held atoms in
`Geometry.Constraints`, the atom order record and the cell and placement in
the ATOM-METADATA and ENGINE-OFFSET blocks — **the same for every geometry
of one structure**, which is why one deck serves every call of a phase. What
differs per call is its coordinates, in `coordinates.fdf`, and what the call
is, in the index and the data file — as PySCF records its displacements in
its result, not in its script. **`part` chooses the sections** (each part is
one `DeckSpec`, the three decks sharing the stage's one name,
`RunNames.name(".fdf")`, each in its phase's folder):

| part | sections | the record blocks |
|---|---|---|
| `relax` | the optimization layout — the relaxation items (`relax_*`, `restart`), the vibration's catalogue narrowing | ATOM-METADATA, ENGINE-OFFSET; no `vibration` block; the bundle named, for its plan and record steps |
| `fc` | § 5.3's: the force-constant section and block — its charges when *Compute IR* is on — the start state | all three; the `vibration` block, format **v5**: v3's keys and **`raman`** — *Raman for* and its range, `null` with *Compute Raman* off — the one fact no keyword states ($\delta$ is `FC.Displacement`; $\hat e$ the optical deck's `Optical.Vector`: nothing is stated twice, § 6.10.1); the finish named |
| `optical` | the vibration's settings; a single point (`MD.TypeOfRun CG`, `MD.Steps 0`); the start state (`DM.UseSaveDM .true.`, `MD.UseSaveXV .false.`); the optical section from the catalogue items (§ 9), each checked against SIESTA's manual before it is written (I21) | ATOM-METADATA, ENGINE-OFFSET; no `vibration` block; the bundle named, for its plan and record steps |

The block's reader (`deck_record.extract_vibration_record`) refuses no key;
the version is the format comment, and **the record step after the force
constants refuses a block without `raman`** (formats v3 and v4, prepared before *Raman for*) naming *prepare
the stage again*, as the finish refuses one without `masses_amu` (§ 5.5). **Every reader of a deck's file follows the
include**: `parse/fdf.py`'s `read_deck` inlines `%block Label < Filename`
from beside the deck as libfdf's `fdf_read` does, and every reader of a
deck's file reads through it (`read_fdf`, given text that still holds an
include, refuses it naming `read_deck`) — `deck_atoms`, the engine atom index, `same_calculation`, the deck's
geometry difference, prep's own read-back check, the run script's "what the
engine will read" echo. Prep writes **the structure's** `coordinates.fdf`
beside every phase's deck, so each is a whole deck prep's two gates check;
every call's folder gets its own file, which is what SIESTA reads.

#### 5. When a call does not end normally

- **What builds on it does not run**: no force constants after a failed
  relaxation; no optical call after failed force constants; an optical call
  that failed leaves the others to run — each builds only on `fc`. The
  phases' dependency is the `PHASES` table's, rendered into the walk (§ 3).
- **A relaxation that ended without converging** — at its step cap after
  its run script's retries — **is done** (§ 6: a `relax` call's done rule is
  *finished*, converged or not): the force constants run on it, and the
  finish's stationarity verdict says it, a warning (§ 5.5). To relax further: `--cold`, or the state saved
  before the first launch and a prep with more steps.
- **The finish runs whatever ended**, and the result states what exists: the
  modes when `fc` finished; a mode's Raman when both its sides did, else the
  reason.
- **The run's state** (§ 7): *failed*, naming the first call not done — or
  the finish — and why.

#### 6. Launched again — warm takes over every call done

**The rule a run of points keeps** (`engines/transport.md` § 2a.11,
*Launched again*), read at the call, **decided by launch on this machine**
(§ 2). Every launch is a new run. **Warm** — the default — takes over each
call **done** in the stage's latest run: its folder whole but not the
engine's scratch (`warmfiles.is_scratch`; an optical call's
`.contrib.EPSIMG`, `[optical]`'s `scratch`, § 8),
its `.continued-from` naming the run that computed it (one hop). **Done is
the phase's rule** (`PHASES`): `relax` — finished, converged or not (§ 5);
`fc` — finished, its SCF converged, its `.FC` beside it (the analysis refuses one whose
row count is not $6\,N\,n_{\mathrm{free}}$, § 5.5); an
optical call — finished, its SCF converged, its `.EPSIMG` beside it; and no call
whose record step failed. **One rule on both sides**: this machine's done
door (`continuation.done`, for `status` and launch's take-over) and the
job's own record step (`siesta_vibration._rows`, for what the finish
writes and the run's state it sends) ask the same pieces — whether the
phase's verdict counts (`PhaseRule.verdict_counts`), the verdict read from
the call's output by one reader (`_run_ending.verdict_of`; the record step
writes it into the index), and the sentence for a finished call not done
(`_run_ending.not_done_why`: *finished, not converged (SCF NO)*, *finished
without `<label>.FC`*) — so a call the job counts done is one `status`
counts done, and with `scf_must_converge` off an SCF that did not converge
is no call done on either. A call **not done** runs in the new run from
its phase's start, by its row's warm-files section — each phase's run script
is rendered with its own section's `resumes` (`Rung.wrapper_at`), and launch
carries a `relax` call's files by its row's section (`warm_kind`):

| phase | a call not done, launched again | its section |
|---|---|---|
| `relax` | continues from its own last geometry, density and optimiser history — copied into the new run's `relax/` by launch, a hand-over (`.continued-from`), as a relaxation launched again does (`restart`, `continue` by default) | `[optimization]` |
| `fc` | again from its first displacement — SIESTA cannot resume one | `[vibration]`, `resumes = false` |
| `optical` | again, from this run's `fc/` density, as the loop seeds it; a retry in place continues its SCF from the density the try left (`DM.UseSaveDM`) | `[optical]` — `.EPSIMG` inventory; `.contrib.EPSIMG` the engine's scratch, beside the call (`scratch` with no `scratch_dir`, `job-contracts.md` § 4.2a); `resumes = true` |

The optical calls taken over are listed to the walk's loop step, which
passes their folders over; a planned optical call **not done** runs again
from the stage's `optical/` files and the coordinates its planning wrote —
launch opens its folder with the last run's call folder's inputs (its
`coordinates.fdf`), never its outputs — and the loop seeds it from this
run's `fc/`. **Cold** (`--cold`): nothing is taken over. Launch shows, and
asks once: *it takes over the relaxation, the force constants, frame 0, the
equilibrium done in run-0; runs the optical calls below run-1/optical/ (10
to run again)* — each call taken over in its words, the rest counted.
A warm launch with every call done **and the finish done** is refused,
naming `--cold`; with every call done and the finish not done (its index
entry missing or its exit not 0), the last call runs again and its record
step finishes the run — launch reads the index's finish entry, not the
calls alone. **A displacement sweep** (§ 5.9) is to have each later
stage's first launch take the first stage's `relax` call over — launch's
hand-over across stages, the one case of § 1's note — so every stage
measures at one geometry; **not built**: each stage relaxes in its own run,
and § 5.9's summary refuses two geometries. `ladder.independent` is false for the kind (a sweep's stages
never continue from each other's `.XV`, which is a force-constant run's last
displacement, § 5.3).

#### 7. The records, the state, and what the parser reads

**Every file is read by its own door**, a call as any point of a run of
points is read:

| what | read by | as |
|---|---|---|
| a call's folder | `calcdirs.where` → a point below the run; `runs.run_of`; `run_status` with the run's launch record (`launched_in` = `run-<n>/`) | a transport point is read: its own output, session log, monitor log, conclusion |
| a call's output | the SIESTA output parser — the trajectory viewer for `relax`, the force-constant reader for `fc`'s `.FC`, the `.EPSIMG` reader (`parse/engines/siesta_epsimg.py`, registered — [`model/parse.md`](?doc=model/parse.md) § 5f) | as any run's output |
| the run's state | **the run of points' rule** (`continuation.points_state`, `engines/transport.md` § 2a.11), its rows the index's calls and its words **one rule over the index, `vibration_phases.summary`** — it travels (stdlib), so the finish's `notify finish` and this machine's `status` count in the same words. *pending* for a call the index lists whose folder is not opened; *queued* while no call the run walks itself has run and its job is not over; *running* while a call runs, or the walk is between two (the job going, `joblist`), or every call has ended and the job still goes — its last call's record step, or the finish, at work; *finished* when every call is done **and** the finish's entry says exit 0 **and** the result is on disk; *failed* naming the first call not done, by its words and why where its record step said, or the finish — every call ended, the job gone, and no finish entry that ended with exit 0 in the index (the finish never ran, or was stopped inside it: its entry is written once it has ended) is *failed*, *the finish did not end*. The index lists `relax` and `fc` from launch, so the window before the first call reads *queued*, never *finished, 0 of 0*; the optical calls appear as the `fc` call's record step plans them, so the count says *optical 3 of 10* once they exist — *(4 taken over)* beside it where a warm launch took calls over — and *optical: after fc* before — only where the run walks the optical phase (the index's `phases`) | `status`, Task setup, the Results ladder, the walk's monitor line |
| a call, in words and on the wire | the vibration's own point type, `vibration_phases.Call` — phase, name, mode, side; `rel`, `words`, `to_wire` — the shape `transport.stages.Point` has; the readers typed on it (`points_summary`, `run_folders`, `_sweep_status`, `stage_run_state`, `every_run`'s wire form, the ladder's run rows) ask the rung's row | *relax done · fc done · optical 7 of 11 (4 taken over)* |
| the index `<label>_01_freq-run<N>.phases.json` | a row of `runfiles.WRITTEN` with its one reader (`vibration_phases.read_index`) and a registry parser, so the Results tab lists it as an aspect of the run | the phases the run walks (its job's row at launch); per call: its phase and folder; what it is (the relaxation; the force constants; frame 0, or mode $\nu$'s side at its frequency and $\delta Q$); its deck's sha256; its coordinates' sha256 and where they came from; its seed and where from; its start, end and exit; its verdict, as its output states it (`{"of": "SCF", "converged": …}`, § 6); its record step's failure, where it failed (the call then not done); computed here or taken over from run-N; the selection as asked and as applied; **the finish's entry** (started, ended, exit — written once the finish has ended). Written by launch (the calls it opens) and by the run's own Python (the plan step fills its call's entry, the `fc` call's record step adds the optical calls, each record step completes its call's, the finish adds its own) — rewritten whole, atomically |
| the run's notification record `notified.json`, at the run's root beside its launch record (`runrecord.notified_path`) | a row of `runfiles.WRITTEN`, the monitor's own (`run-reports.md` § 2.6), its parser registered (`parse/sidecars/notified.py`): every event sent for the run, when — written by `mb_monitor.pyz notify` and by a member monitor for every event it sends (`scf_converged`, `periodic`, `wall_near`); read by every member monitor for its N-hour clock and by the `notify` verb, which sends no second `start` | kind-blind; a transport run of points writes none — its points send their own events (§ 7a) |
| the run folder `run-<n>/` | `runs.folder_answer`, of the one folder: a container answers no state and no record. **A run of points' root** is read as its run of points by the view floor 6 composes (`runstatus.points_view`, [`execution/architecture.md`](?doc=execution/architecture.md) § 3.2): the run's state — its points' — its calls, the call at work and its record, and what the run opens: `.spectra.json` at the root once it is written, else the call at work's result; `run_answer` for a file at its root answers the run's, so a viewer of the result polls while the run goes | the Results tab follows the run |

#### 7a. The monitor — one per call, the run reporting as one run

*(The user, 2026-10-25: "make sure the monitor probe etc should be
integrated correctly with this new design".)* **Each call's run script
starts its own monitor**, as every run script does (`run-reports.md` § 2.3,
§ 2.4): it watches that script's PID, reads that call's output and SCF
timing log through the readers shipped beside it, writes that call's
`.monitor.log` and `util.csv` — `[MACHINE]`, `[STATUS]`, `[WALL]`,
`[UTIL-SUMMARY]` — and is stopped by the script's trap at the call's end
(SIGTERM) or before a retry in place (SIGUSR1). A call's folder reads
exactly as a transport point's does.

**What a member monitor takes from the run** (`monitor.py`;
`run-reports.md` § 2.1b: *a run of points' points count nothing*): a
monitor reads the job's wall from the launch record **in its own folder**,
which a point has none of, and keeps its own N-hour clock and
`start`/`finish`. For a member of a walk:

- **the wall**: the run script's monitor block, rendered for a member,
  passes `--launched-in <the run's folder>` (what `Run.launched_in` answers
  for a point), so the monitor reads the run's one launch record and counts
  the time left as a run's does; `MB_WALL_S` and `MB_JOB_STARTED` reach the
  engine as for any run;
- **the run's start**: on a queue the header sets `MB_JOB_STARTED` once; run
  here, **the walk exports it once at its start** (`running-a-job.md`
  § 3.6) — so every call's `MaxWalltime` counts what is left of the run's
  wall, and a run here is told its card's wall for the whole run, not per
  call;
- **where the run is** *(decided 2026-10-28, not built)*: the run's
  position and the call's own, from the run's index through its one
  reader (`vibration_phases.summary`, `Call.words`); the run's clock and
  the call's, each named; the pace of the phase underway's finished calls
  and the calls left ([`run-reports.md`](?doc=execution/run-reports.md)
  § 2.8);
- **the events** — the run speaks as **one run**, never as 121 calls:

| occasion | sent by | how |
|---|---|---|
| **it started** | the plan step of whichever call runs first, once | `mb_monitor.pyz notify start --dir <the run>` — a verb of the monitor's bundle: one event through the same channels and card the monitor uses, with the run's identity; recorded in the run's notification record (§ 7), which is what says it was sent |
| **a step finished** (`notify.on_scf_converged`) | the call's monitor, for each step its engine states: the `relax` call's geometry steps, the `fc` call's displacements (`run-reports.md` § 2.2) | an optical call is a single point and states no step, so its monitor sends none |
| **every N hours** (`notify.every_hours`) | the call's monitor | **from the run's notification record**: the clock counts from the last event sent for the run — read from the record by the member monitor, which writes its own there — so a run of many short calls speaks every N hours, and never once per call |
| **its wall is near** | the call's monitor | from the run's launch record (`--launched-in`), once for the run, recorded |
| **it ended** | the record step that completes the index, once | `mb_monitor.pyz notify finish --dir <the run> --state … --detail …` with the run's state from the one rule over the index (§ 7) |

So each call's monitor is started as a **member** (`--member`, with
`--launched-in`): no `start`, no `finish`, its clock and its wall the run's.
**This rule is a run of points' rule**, not Raman's; it is built for the
vibration's calls, whose Python steps send the run's events. **Not built for
a transport sweep**: its points send their own `start` and `finish`, one
pair per point — its walk runs no Python, and the step that would send the
run's one pair is not designed (plan § 1ae). A call's run script carries
`--member` however it is started, so run by hand it still reports as a
member of its run.

**The walk's own log** (`launch/<label>_01_freq-sweep.log`) holds one line
per call — start, end, exit — and the Python steps' lines; a monitor's log
is its call's. **The Results tab** follows a call's folder by that call's
state (`run_answer`) and the run's folder by the run's state (§ 7); the
ladder lists every call under its stage, each row named by its run and the
call in the page's words — *run-0 · the force constants*, *run-0 · mode 3
(2548 cm⁻¹), +δQ* (`lib/results/point-words.js`, [`web/results.md`](?doc=web/results.md)
§ 2.5). **Not built**: the Run panel naming the phase and the call that is
running — the run's folder is served its state and no record, so the panel
stays hidden there.

#### 8. The result — the structure PySCF's has, SIESTA's values in it

**One result type, two routes** (§ 6): a SIESTA result is a `SpectraResults`
like a PySCF result, and the viewer reads both **by role, never by the
engine's name** ([`web/spectra.md`](?doc=web/spectra.md) § 3.3, § 9b.3) —
**from the result's own record**, which both engines write the same way:
**whether a strength was asked** by its phase — `phase_ir`, `phase_raman`
*not requested* is a run that never asked (*Compute IR*, *Compute Raman*
off), on either engine; **which way it was computed** by its route —
`ir_route` `charges` and `raman_route` `optical` are SIESTA's, derived on
request from what the run stored (the atoms named, the laser typed),
`analytic`, `finite-difference` PySCF's. A PySCF run still sweeping states
its route `none` and its phase `running`, so it reads as asked. **The
probe** is PySCF's alone: by its `es_mode_selection`, which a SIESTA
`config` never holds, so on SIESTA it is *not computed on this route* — the
one capability one route lacks. A SIESTA result's `config` carries the
engine, the kind, the stage and the force-constant deck it was computed
from — its name, its sha256, every keyword and block it states as written
but its geometry (the functional, the basis, the mesh, the k-points, the
species) and each species' pseudopotential by name and sha256 (§ 6.2) — *Raman for* as asked and the modes
it picked are the optical citation's and the index's; its `phase_es` stays
`not requested`.

| field | PySCF | SIESTA, this run |
|---|---|---|
| `phase_relaxation` | `running` → `complete`; `complete` under `already_relaxed` | the `relax` call: `complete` in every result the run writes — the first is written after the force constants; `not requested` when the structure was stated relaxed |
| `phase_frequencies` | `running` → `complete` | the `fc` call and its analysis |
| `phase_ir` · `ir_route` | the dipole sweep; `analytic` / `finite-difference` | `complete` once the charges are stored (`partial_charges`), `ir_route` `charges` — the infrared derived on request from them (§ 5.10 ②); `not requested` with *Compute IR* off, `ir_route` `none` |
| `phase_raman` · `raman_route` · `raman_fd_step_ang` | the polarizability sweep; `finite-difference`, `RAMAN_FD_STEP_ANG` | `running` while the optical calls run, `complete` when all ended — `complete` with no call and `raman_route` `none` where *Raman for*'s range kept no mode — `not requested` with *Compute Raman* off; `raman_route` **`optical`** — SIESTA's optical calculation at $\pm\delta Q$ along each mode; `raman_fd_step_ang` = $\delta$ |
| a mode's `raman_activity_a4_amu` | the Placzek activity, stored | **null in the file**: $R(\hat e)$ depends on the laser and the width the person asks (§ 10), so it is derived on request and served beside the result — the load door's `raman` view, as its `infrared` view serves the charges' infrared; the viewer's one accessor `_ramanOf(m, r)` reads the view's value where the door derived one, else the file's (PySCF's) — `_irOf`'s pattern — and the modes table's Raman column and the chart's Raman lane read it |
| `optical` | — | the data file's name and sha256, $\hat e$, $\delta$, the selection as applied, and per mode its $\delta Q$ and the state of its two sides — the citation a reader of the result needs; $\varepsilon_2$ lives in the data file alone |
| `partial_charges`, `engine_metadata`, `relaxation` | as § 6 states | as § 6 states; `relaxation` from the `relax` call's own record |

**The live artifact** (§ 4.9's rule, on SIESTA): the result pair is written
after the `fc` phase — the modes, the charges, `phase_raman` `running` — and
again by the finish, so the viewer follows the run as it follows a PySCF
run; and **`<label>.optical.nc`** is rewritten after each optical call, so
it is the record of the Raman phase so far. Each optical call's record step
puts its frames in the user order by the reference the run's `fc` call
gives — its structure, atom index map and masses — read from that call's
own files as the finish reads them, whether this run computed it or took it
over (§ 6): a run that took its force constants over has no result at its
root until its finish.

**Its variables** — the calculation in words, its title: *SIESTA's optical
calculation: ε₂(ω) for a field along ê, first-order perturbation over
independent states, at the equilibrium geometry and at ±δQ along each
selected mode*; every SIESTA parameter the calls read under `/settings`
(`spectra/optical_data.py`, schema
`molbuilder/optical-data@2`, `@1` — `delta_ang` an attribute — read as it
is; the framework's rules are
[`model/datafiles.md`](?doc=model/datafiles.md), every group titled):

```text
/                     attributes: title -- the calculation in words, schema,
                      molbuilder_version, source -- SIESTA's version and the
                      run, Conventions, unit_meanings, label, run,
                      siesta_version
/settings             delta_ang -- angstrom, of no dimension
                      attributes: keywords (JSON: each Optical.* key and
                      OpticalCalculation as written), direction, selection
                      (JSON: asked, applied), deck_name, deck_text,
                      deck_sha256
/structure            the result's own layout (spectra.datafile.
                      write_structure_group): atom(atom) -- the user index,
                      element, position_ang, mass_amu, held, cell_ang,
                      axis_kind, user/engine_atom_index; structure_json
/frames               frame(frame) -- 0 the reference, then each mode's +dQ
                      and -dQ  atom(atom)  xyz(xyz)
                      photon_energy_ev(photon_energy_ev) -- eV, as written
                      spin(spin) -- unpolarized, or up and down
                      call(frame) -- optical/f000 ...  mode(frame) -- -1 at 0
                      frequency_cm1(frame)  side(frame)  dq_amu12_ang(frame)
                      position_ang(frame, atom, xyz) -- the user order
                      state(frame)  computed_in(frame)  scf_iterations(frame)
                      total_energy_ev(frame)  fermi_level_ev(frame)
                      engine_elapsed_s(frame) -- the call's session log
                      f_sum(frame, spin)  drude_wp2_ha2(frame, spin)
                      eps2(frame, photon_energy_ev, spin) -- raw, <label>.EPSIMG
```

A frame not run yet is a row of fills beside its state `pending`; a frame
whose spectrum lies on another grid of photon energies than the others' is
refused by name — the calls share one deck. New axis: `photon_energy_ev`,
the light's energy $\hbar\omega$ (a concept other than `energy_ev`, which is relative
to the leads' Fermi level) — a row of `model/datafiles.md` § 2a. Its size is the resolution's (§ 9). **The cost a
person owns**: at 0.002 Ry each `.EPSIMG` is 2.7 MB of text, 320 MB for a
junction's 121 frames, carried with every done call and saved by every
checkpoint; `.contrib.EPSIMG` — each band pair's share at each bin
(`Src/optical.F`) — is the engine's scratch (`[optical]`'s `scratch`),
beside the call, never carried and never saved; prep says the sizes beside
the count (§ 9).

#### 9. The settings — one stage's card, a section per phase

**The roles are the phases'.** The kind table names the vibration's three
roles, `relaxation`, `force_constants`, `optical`, each computing an SCF
([`execution/architecture.md`](?doc=execution/architecture.md) § 3.3), and
the `PHASES` row carries its `role`; with one stage, an item's
`stages = ["relaxation"]` names **the phase whose deck renders it**, and the
stage's one resolved configuration feeds every phase's deck, each part
taking its items (§ 4): a SIESTA vibration rung carries every role (its
part's `stage_roles = "every"`; `template.stage_roles`, a tuple — `reads` is
any-of), a PySCF rung `force_constants`; the named set `"scf"` holds
`optical`; a stage override applies to every phase; the stage table offers
one row.

| phase | items |
|---|---|
| every phase | the vibration's settings: basis, functional, mesh, k-grid, SCF — one model |
| `relax` | the relaxation items: `relax_type`, `relax_steps`, `relax_force_tol`, `relax_max_displ`, `restart`; `already_relaxed` turns the phase off |
| `fc` | `fc_displacement`; the charges (`Charge.Hirshfeld`, `Charge.Voronoi`) written at every step when *Compute IR* is on, neither when it is off — the infrared's source (§ 5.10 ②) |
| `optical` | **planned when *Compute Raman* is on**, for the modes *Raman for* picks — `raman_modes` (`all` every mode · `range` within `raman_freq_min_cm1`–`raman_freq_max_cm1`), the setting both engines share (§ 3.1), applied by the one selector in the job (`select_modes`); `optical_direction` (`Optical.PolarizationType polarized`, `Optical.Vector`; z); `optical_mesh` (`Optical.Mesh`; none for a molecule — SIESTA takes Γ alone for a cell it judges a molecule — the SCF's k-grid for a cell joined across its boundary); `optical_broaden` (`Optical.Broaden` — **the run's resolution** (§ 13, R-c): SIESTA spreads each transition as a Gaussian of this width on a grid a tenth of it and writes only the spread spectrum, `Src/optical.F:287`; the width a person reads is applied when the result is presented, § 10); `optical_energy_max` (`Optical.Energy.Maximum`, 3 Ry — a cutoff on the pairs of states the calculation couples, SIESTA's own 10 Ry reaching only its basis's artefacts, measured in the method's § 3.1, *The cutoff*; SIESTA's source reads `Optical.EnergyMaximum`, `Src/reoptical.F:107`, one label to fdf, which compares labels without `.`, `_`, `-` or case, `libfdf utils.F90 labeleq`); `optical_bands` (`Optical.NumberOfBands`, every band) |

Prep states the count and the sizes beside the wall, a line of its
request on both doors (`vibration_phases.run_size_words`, the rung's `said`):
the relaxation up to
`relax_steps` geometry steps a try, its retries the run card's, said beside
it; the force constants $1 + 6\,n_{\mathrm{free}}$ SCFs; Raman $1 + 2\,N$ single
points, $N$ at most the modes the run reports — every one under `all`, those
inside the range under `range`, which only the run can count —
$3\,n_{\mathrm{free}}$ less the whole-body motions the held atoms leave (the
rank rule prep's check states, `science/normal-modes.md` § 3.1), and each
`.EPSIMG`'s size from its photon energies (`Optical.Energy.Maximum` over a
step of `Optical.Broaden / 10`), one spin channel's and what a second adds.
The run's one job is worded by the points declaration (`prep.walk_words`):
its calls known at prep — where the stage runs its optical phase, the rest
planned in the job — on the stage's one wall. The wall stays the person's.

**A SIESTA vibration described before 2.45.0** states no *Compute IR*,
*Compute Raman* or *Raman for*: its `es_mode_selection` decided the Raman,
and its charges were always written. Prep refuses it naming
`jobset migrate` — the selector's rows are retired on SIESTA
([`engines/template.md`](?doc=engines/template.md) § 3) — and migrate states
the three from what it said: `skip` → *Compute Raman* off; `all` →
*Compute Raman* on, *Raman for* `all`, or `range` with the window's bounds
where it gave one; *Compute IR* on — so the deck it renders computes what
it computed before. An `explicit` list is refused by name: mode numbers have
no range to become, so the person chooses *All modes* or a range.

**A calculation described before** — a `relax → freq` ladder — is read
as it is wherever it stands: its runs are runs, read by the run door, and a
launched one is never prepared again. **Before any verb writes** for such a
description — `prep`, a Save from Task setup — `jobset migrate` turns the two
stages into the one (the `relax` stage's overrides onto the relaxation
items, `freq`'s onto the rest, a conflict refused by name) and prep refuses
an unmigrated one naming migrate, as it refuses a retired item
(`template.RETIRED_ITEMS`). The retired stage — its name, the role it
played, 2.42.0 and why — is the kind table's, on SIESTA's part of the
vibration (`execution/architecture.md` § 3.3); both the refusal and the fold
read it. `status` and the Results ladder read an old
calculation's two stages by their folders.

#### 10. Raman, presented — at the laser and the width a person asks *(the user, 2026-10-26: "agree with your M4 recommendations … need export, and api to support netcdf file reading")*

**What it is**: after a SIESTA vibration whose optical phase ran, each picked
mode's Raman strength at a laser the person names, derived when asked from
the $\varepsilon_2$ the run stored (`<label>.optical.nc`) — no new calculation; a new
laser is a new number in a second. Nothing derived is written into the
calculation: the result keeps a SIESTA mode's `raman_activity_a4_amu`
null (§ 8), and the view is served beside it.

**The one derivation** — `spectra/raman.raman_view(result, optical, *,
laser_nm, broaden_ev)`, the pattern of `infrared_view` (§ 5.10 ②), over the
data file read through its one reader (`spectra.optical_data.read_optical_data`,
through `ncdata.reading`); `raman_view_of(path, *, laser_nm, broaden_ev)`
reads the result at `path` and the data file it cites beside it, and names
both by their sha256 — the one door the load door, the export and
`postprocess` ask. **The physics, each formula, is the method's § 3.2a**
([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md));
here, the numerical choices and the refusals. The laser and the width are
each a positive number (`read_ask`); a laser whose photon energy lies
outside the stored spectrum is refused, naming its ends; a data file written
before any optical call ended — its photon energies none yet — is refused,
saying so. Per frame:

1. **The spectrum**, the spin channels summed, on the run's photon-energy
   grid.
2. **Broadened to the width asked** — a Gaussian of
   $\sqrt{\sigma^2-\sigma_{\mathrm{run}}^2}$, since two Gaussians make one,
   in SIESTA's own convention of `Optical.Broaden` — $\sigma$ the $s$ of
   $e^{-(E/s)^2}$, its full width at half height $2\sqrt{\ln 2}\,s \approx 1.665\,s$;
   exact for a transition a few widths above zero (§ 12), and not below:
   SIESTA stores only the part of each line above zero, so a line within a
   few widths of zero — a metal's near-Fermi pairs — keeps less weight when
   widened after the run than a run at the wider width would;
   $\sigma_{\mathrm{run}}$ the optical deck's, from the optical keywords the data
   file's `/settings` keeps, read by the deck's reader (`parse_fdf_params`) —
   a deck stating none is refused. The run's own width is the narrowest there
   is: a smaller one is refused, saying the run's.
3. **$\varepsilon_1$ by Kramers–Kronig** (the method's step 3), at the laser's
   energy (step 1, `constants.HC_EV_NM`) and at zero: the principal value
   taken by subtracting the pole's value — and its slope at a grid point —
   read from a cubic spline through the spectrum, and adding its
   closed-form integral, the trapezoid on the run's uniform grid up to its
   `Optical.Energy.Maximum`; the static integrand from the grid's first
   energy above zero.
4. **$\alpha$** (step 4) in Å³, $V$ the reference structure's cell volume —
   a data file whose structure has none is refused; complex at the laser,
   its imaginary part read from the same spline; static, real.

Per picked mode, from its two frames, $\partial\alpha/\partial Q$ and
$R(\hat e) = \lvert\partial\alpha/\partial Q\rvert^2$ (steps 5–6), at
the laser and static — the squared modulus of the complex polarizability's
change, as a Raman tensor at a laser is [Gillet2013] (the user, 2026-10-26:
*"Full complex"*); ranked by the laser's against the strongest real mode,
no rank at or below the Raman channel's band floor
(`activity.CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU`, $10^{-3}$ Å⁴/amu — set on the
activity's scale, $45a'^2+7\gamma'^2$, 45–79 times the orientation-averaged
$R$ (the method's § 3.2); as a gate of whether there is a band at all it serves $R$ too).
Frame 0's $\alpha$, at the laser and static, is said once. A mode with a side
not done is a row naming the side and its state — `pending`, or `failed` —
never a number from one side. **It answers**: the laser, the width asked and
the run's, $\hat e$, $\delta$, the equilibrium $\alpha$, and per mode its
frequency, $\delta Q$, $\partial\alpha/\partial Q$ and $R$ at the laser
and static, its rank and its state.

**What the person sees** (the decisions, § 13): **the laser is the person's
to name** — no default; until one is typed the Raman lane says *type the
laser's wavelength — the strength depends on it*, with 532, 633 and 785 nm
offered (M4-b); **the width starts at the run's own** — any other starting
number would be one molbuilder invented (M4-a); **the quantity is named as
what it is** — *Raman strength, light along z, 633 nm, 0.0272114 eV
(Å⁴/amu)*, the static value beside it, *Raman strength, static, light along
z, 0.0272114 eV* — never under PySCF's *Raman activity*, the tumbling
molecule's $45a'^2 + 7\gamma'^2$, a different number (the method's § 3.2,
M4-c); **strengths, not intensities** — the laser-frequency and temperature
factors of a measured intensity are not applied, on either engine (M4-d).
Every number states its width, and the laser's its laser: near a resonance,
and for a metal's near-Fermi transitions, the width changes $\alpha$ — the
static one too, since the static integral reads the smoothed spectrum. **The
Methods paragraph**: the result's, written by the run, says the route — its
optical calls, the strengths derived on request at a laser — or that none
were requested (`methods.siesta_raman_sentence`); a view shown adds its
sentence at load — the route, $\hat e$, $\delta$, the laser, the width, and
*independent states, no local field* (the method's § 3.3,
`methods.raman_view_sentence`) — and its citation joins the result's
`bibliography_keys`.

**The doors** — one derivation, asked four ways:

| door | asks | answers |
|---|---|---|
| the load door, `POST /api/spectra/load` | `raman: {laser_nm, broaden_ev?}` beside `infrared`, read by `spectra.raman.read_ask` | `results.raman`, the view; a view the result cannot give — no optical route, no optical call finished yet, a deck stating no width, a width below the run's, a laser outside the stored spectrum, a data file with no cell, a result opened without its folder (an upload: the data file beside it cannot be read) — `results.raman_said`, the result still loading |
| the export, `GET /api/spectra/export` and `molbuilder jobset export` | `format` `netcdf` or `csv`; `laser_nm`, `broaden_ev` (the page's boxes; the command's `--laser-nm`, `--broaden-ev`) | the calculation's vibration data files as they are, and the view composed now (below) |
| a script, `molbuilder.postprocess` | `raman_view(path, laser_nm=, broaden_ev=None)`; `read_optical_data(path)`; `read_raman_data(path)` | the same view; the optical calls' data file as data, with its atom index map; an exported view as data |
| a notebook, with no molbuilder | `xarray.open_dataset(path, group="frames")` | every data file reads as NetCDF-4 alone (`model/datafiles.md` § 0) |

**The export of a vibration calculation** — one composition,
`spectra/export.export(calc, fmt, *, folder=None, laser_nm=None,
broaden_ev=None)` of the calculation (`calculation.at` of the path asked)
and the result it holds — the one in `folder`, the run the viewer shows;
else its stage's newest run's (`result_path`) — through the one composer
both exports share (`ncdata.exported`, one `Exported` shape for both) —
**what is already there, and the view at the
laser named**: `<label>.spectra.nc` and, on a SIESTA run with an optical
phase, `<label>.optical.nc`, each copied as its writer wrote it, and, when a
laser is named, `<label>.raman.nc`, the view composed now through the
data-file framework ([`model/datafiles.md`](?doc=model/datafiles.md) § 1,
`molbuilder/raman-view@2`), citing the two files it was derived from by name
and sha256. **NetCDF**: the files; **CSV**: each file's tables and README
(`ncdata.as_tables`) — flat for one file, a folder each for several. A
download is one zip
(`<label>.vibration-netcdf.zip`, `<label>.vibration-csv.zip`); the command
writes into the folder named, outside the calculation, refused where the
name is there already. No laser named on a SIESTA result with optical calls:
the export holds the data files and says the view is not in it. **Refused by
name**: a form not one of the two; a calculation of another kind, or with no
run or no result yet; a displacement sweep named by its calculation, whose
several stages each hold a result; a width named with no laser; a laser
named for a result whose Raman is not the optical route (a PySCF result
stores its activity); the view's own refusals (above); a data file that does
not read, in its reader's words. Nothing is written into the calculation, and the
ledger has no line of it.

**`<label>.raman.nc`** (`spectra/raman.py`'s variable table; written only by
an export):

```text
/            laser_nm  photon_energy_ev  broaden_ev  run_broaden_ev
             delta_ang  cell_volume_a3  band_floor_a4_amu -- each of no
             dimension, its unit stated
             attributes: title -- the Raman strengths of a SIESTA
             vibration's picked modes at a laser, derived from its optical
             calls' eps2, and how; schema, molbuilder_version, source --
             the files it was derived from and the SIESTA that computed
             them, Conventions, unit_meanings, label, direction, spins (the
             channels summed), run, siesta_version, derived_from (JSON:
             each data file's name and sha256)
/equilibrium alpha_static_a3  alpha_laser_real_a3  alpha_laser_imag_a3 --
             frame 0's, angstrom^3
/modes       mode(mode) -- the result's numbering
             frequency_cm1(mode)  dq_amu12_ang(mode)
             dalpha_dq_static_a2_per_amu12(mode)
             dalpha_dq_laser_real_a2_per_amu12(mode)
             dalpha_dq_laser_imag_a2_per_amu12(mode)
             strength_static_a4_amu(mode)  strength_laser_a4_amu(mode)
             relative(mode)  state(mode)  why(mode)
```

A view exported before 2.47.0 (`raman-view@1`, its numbers attributes) is
refused by its reader with the way back: the view exported again.

**Not designed**: the named set's share (D-e) — its nearest published
counterpart, junction theory's polarizability projected on the molecule
(the method's § 3.4).

#### 11. Where each piece lives

| piece | where |
|---|---|
| **one stage on both engines** | `pyscf/stages.vibration_stages`; the shape SIESTA's fixes, its part in the kind table; the kind table's vibration roles, the phases' (§ 9); `ladder.independent` false for the kind (§ 6) |
| **a rung of points, declared** | one `PointsOf` per pair that runs per point, named by the pair's `points` in the kind table — `jobset/vibration_points.py`, `jobset/transport_points.py` — the type and its doors (`points_of`) in `jobset/points.py`: its points as prepared (transport: the bias item × the cited set; the vibration: `relax` and `fc`), the points a run holds (the vibration's optical ones, planned in the job, those its index lists), its point type (`transport.stages.Point`, `vibration_phases.Call`), what makes a point done, what a run needs beyond its points (the vibration's finish), its containers, one wall or one per point; prep, launch's walk, `materialize`, `continuation` (`point_rows`, `points_state`, `done`), `runstatus` (`run_folders`, `_sweep_status`, `stage_run_state`, `every_run`), `ready`, `prepared_with` and the run door ask it |
| **the phases** | `siesta/vibration_phases.py`: `PHASES`, `Call`, the index's writer and reader (a `runfiles.WRITTEN` row, a registry parser), `summary` (the state's words); `runfiles.fixed`'s `coordinates.fdf` row |
| **the run door and the view** | `runs.folder_answer` of one folder; the run of points' view (`runstatus.points_view`, floor 6) at its root and its calls' folders, and `run_answer` for a file at its root, the run's state (§ 7); `materialize.open_sweep_run`; the opener's travelling core (`calcdirs` + `paths` in the bundle, `bring_files`' copier); the progress-log seed copied with the deck |
| **prep** | each phase's deck and run script in the stage's phase folder (`spec_for(part=)`, § 4); the structure's `coordinates.fdf` beside each; the count and sizes beside the wall; `Job.phases` the phases the run walks; each phase's run script with its own section's `resumes` |
| **launch** | the phased walk (`submit._walk_script`: the loop step, `needs`, `MB_JOB_STARTED` exported once for a run here — `running-a-job.md` § 3.6); `run-<n>/`, its `optical/` container and the known calls opened (`materialize.open_sweep_run`); the calls a warm launch takes over, and a `relax` not done's warm files, through `materialize.HandOver`; the seeds as walk copies; the refusal reading the index's finish entry; `warm-files.toml`'s `[optical]` section |
| **the monitor** | `--member` and `--launched-in`; the `notify` verb; the run's notification record (`notified.json`); the N-hour clock and `wall_near` from that record and the run's launch record (`run-reports.md` § 2, § 2.1b, § 2.6) |
| **the job's Python** | `spectra/siesta_vibration.py` (`plan`, `record`, `finish` — the two steps run inside each call's run script, `runwrap`'s `steps`), the `.EPSIMG`'s one reader (`parse/engines/siesta_epsimg_reader.py`, travelling) and its registered parser (`parse/engines/siesta_epsimg.py`), the coordinates writer (`siesta/deck_lines.py`), the displacement rule's one function; `selection`, `calcdirs`, `paths`, `vibration_phases` in the bundle |
| **the result** | `spectra/results.py`: `optical`, `raman_route` `optical`, `raman_fd_step_ang`, `phase_raman` on SIESTA (schema 13) — `phase_ir` and `ir_route` `charges`; `spectra/datafile.py`; `spectra/optical_data.py` (its `Var` table); `ncdata.AXES`; the record step's refusal of a v3 or v4 `vibration` block |
| **the reader of decks** | `parse/fdf.py`: `read_deck` inlines `%block Label < Filename`, as libfdf reads it, for every reader of a deck's file |
| **the view** | `spectra/raman.py` (the derivation, its ask's reader, the `raman.nc` variable table); the load door (`raman` view, its asks); `spectra/export.py` and `GET /api/spectra/export`, `jobset export` for a vibration; `postprocess.raman_view`, `read_optical_data`, `read_raman_data`; `methods`; the viewer — `_ramanOf`, the two boxes, the Export buttons, and every gate read by route (§ 8) |
| **the catalogue** | the optical items; *Compute IR*, *Compute Raman* and *Raman for* on both engines, under their results' cards (§ 3.1) |
| **the sweep** | `displacement_sweep.collect_sweep` reads each stage's `fc` call below its run; `jobset migrate` turns a description of the earlier `relax → freq` ladder into the one stage (§ 9) |

#### 12. Verified — the proof of concept *(2026-10-25, H₂ in a 10 Å box, SIESTA 5.4.2 of the packaged env, by hand in scratch, outside jobset)*

One shell script ran the three phases through one call function, each
SIESTA call in its own folder, every deck's coordinates read from
`coordinates.fdf` beside it; molbuilder's own readers between the phases
(`siesta_reader`, `read_fc` + `hessian_from_fc`, `vibrational_modes`). 46 s
in all on 2 ranks:

| checked | measured |
|---|---|
| `%block AtomicCoordinatesAndAtomicSpecies < coordinates.fdf` | honoured: every output's first geometry is its coordinates file's, to the printed digits |
| the phases chained by the density | `fc` read `relax`'s `.DM` (*Attempting to read DM from file... Succeeded*); frame 0 read the force-constant run's `.DM` and converged in **one** SCF iteration — that file is the reference step's density (§ 5.3) — a displaced frame in four |
| the relaxation | 6 steps, 25 SCF iterations, 11 s; the bond 0.775 Å |
| the force constants | 13 steps, 41 SCF iterations, 19 s; 13 Hirshfeld and 13 Voronoi blocks; one mode, 4271 cm⁻¹ |
| the optical single points | 3.5–5.4 s each; `.EPSIMG` 100,001 rows (5.3 MB) at `Optical.Broaden` 0.001 Ry, 10,001 (0.53 MB) at 0.01, 1,001 at 0.1; `.contrib.EPSIMG` 30 kB; `Optical.NumberOfBands` defaulted to every band (10) |
| the polarizability | $\alpha_{zz}(0)$ = 1.32 Å³ at frame 0, 1.36 Å³ at 633 nm (the molecule's measured 0.80 Å³ mean, 1.0 Å³ along the bond — independent states, no local field, § 3.3 of the method) |
| the Raman strength | $\partial\alpha_{zz}/\partial Q$ = 4.84 Å²/amu^½ static, 5.08 at 633 nm; $R(\hat z)$ = 23.4 and 25.8 Å⁴/amu — non-zero for H₂'s one mode, which is Raman-active and infrared-silent |
| the broadening composed after the run (§ 10, step 2) | **exact**: the 0.001 Ry run's $\varepsilon_2$ convolved with a Gaussian of $\sqrt{\sigma^2 - \sigma_{\mathrm{run}}^2}$ gives, at 0.01 Ry, $\alpha(0)$ 1.3208 and $\alpha(633\ \mathrm{nm})$ 1.3552 Å³ against the direct 0.01 Ry run's 1.3208 and 1.3552; at 0.1 Ry, 1.3290 and 1.3648 against 1.3290 and 1.3648; the integral of $\varepsilon_2$ 0.3290 either way; the f-sum 1.0027. (The width itself moves $\alpha$ — 0.6 % between 0.0136 and 1.36 eV — which is why every number states it) |

Not measured here: a junction's optical cost and its `.contrib.EPSIMG` size
(every transition at every energy bin above 10⁻⁵) — measured on one junction
frame, on the machine that runs it (the method's § 5.1).

#### 13. Decisions — the user's

| # | question | recommended |
|---|---|---|
| R-a | how SIESTA's form asks for Raman | **decided 2026-10-25, as recommended**: the selector alone — `skip` no optical phase; `compute_raman` stays PySCF's (its Cartesian route). **Revised 2026-10-27** (the user: "raman and ir option is engine neutral, so why don't you make it explicit and clearly have it user accessible"; "there is a 'all mode' option? i think that should be by default"): `compute_ir`, `compute_raman` and *Raman for* (`raman_modes`, all modes by default, or a frequency range) are one setting each for both engines, drawn once before the engine is chosen, infrared on and Raman off by default on both; PySCF's mode selector stays the probe's ([`engines/template.md`](?doc=engines/template.md) § 6.2a; § 3.1) |
| R-c | the run's resolution, `optical_broaden` | **decided 2026-10-25, as recommended**: 0.002 Ry (0.027 eV): 15,001 energies up to the 3 Ry cutoff — 0.8 MB of `.EPSIMG` text a frame, 0.12 MB in the data file per spin; a junction's 121 frames, 96 MB and 15 MB; any presentation width from 0.03 eV up |
| R1 | the measurement (plan § 1ac) | decided 2026-10-25: through the built run, on the toy junction, two `optical_mesh` values. **Withdrawn 2026-10-27**: the toy junction's free atoms are links of its metal, and the published work settles what the static value beside a metal is (the method's § 3.1, § 7); the Raman's science checks are the person's, on their machine |
| R-e | a run of points' monitors report as one run (§ 7a) — transport's points lose their per-point `start`/`finish` | taken: one run, one start, one finish — built for the vibration; a transport sweep's points keep their pair until its walk has a step that sends the run's (§ 7a) |
| D1 | `.contrib.EPSIMG`, SIESTA's scratch beside an optical call, where the vocabulary named scratch only with a folder of the run's own | **decided 2026-10-26, as recommended**: declared where it lies — `[optical]`'s `scratch` names it, and `scratch_dir` is stated only where a kind's scratch has a folder of the run's own (`job-contracts.md` § 4.2a) |
| D2 | every call done and the finish not: the last call runs again, its record step finishing the run — or a verb that runs the finish alone | **decided 2026-10-26, as recommended**: the last call again (§ 6) |
| D4 | `[optical] resumes`: a retry in place continues the SCF from the density the try left (`DM.UseSaveDM`) — or runs the call again from its seed | **decided 2026-10-26, as recommended**: `true` |
| C4 | prep stating the count and the sizes beside the wall (§ 9) | **decided 2026-10-26, as recommended**: built 2.43.0, beside the `.EPSIMG` size |
| M4-a | the width the page starts at | **decided 2026-10-26, as recommended**: the run's own (`Optical.Broaden`, 0.027 eV by default) — a value the run states; wider is a keystroke, narrower impossible |
| M4-b | the laser before the person names one | **decided 2026-10-26, as recommended**: none — no Raman numbers until one is typed, 532, 633 and 785 nm offered |
| M4-c | what SIESTA's Raman column shows | **decided 2026-10-26, as recommended**: $R(\hat e)$ at the laser, named *Raman strength, light along …, the laser, the width*, the static value beside it — never under PySCF's *activity* |
| M4-d | the measured intensity's laser and temperature factors | **decided 2026-10-26, as recommended**: not applied — strengths on both engines; **an option later** (the user, 2026-10-26: "add the Stokes shift factor as an option later"): the measured intensity's factors, the method's § 3.2a, applied to the strength on request, on both engines, the plain strength kept beside it |
| M4-e | an export and a script's reading of the Raman data | **asked by the user, 2026-10-26** ("need export, and api to support netcdf file reading"): the vibration calculation's export (§ 10) and `postprocess`'s three readers |

#### 14. Tests — each with the silent failure it catches

| test | category | catches |
|---|---|---|
| CO₂, carbon held, not stated relaxed, *Compute Raman* on for all modes, through `jobset init → prep → launch` — the SIESTA vibration e2e | e2e | a phase not joined to the one before (the index's `fc` coordinates' sha256 is `relax`'s output's); a call not made through its run script (its output, session log, monitor log and conclusion in its folder, each read by the run door as a point); the index, the result and the data file disagreeing; the data file not reading (its reader and `xarray.load_dataset`); the result's phases and route wrong for the run (`phase_relaxation` complete, `phase_raman` complete, `raman_route` `optical`); the run's state read *queued* before the first call and *finished* only with the finish's entry; one `start` and one `finish` in the index's notification record; **the symmetry**: the stretch whose oxygens move together reads $R(\hat z)$ ≈ 0 against the one whose oxygens move apart — CO₂'s mutual exclusion |
| the same, launched again warm after the person stopped it once an optical call started (`molbuilder jobs stop`) — a copy of it launched again cold, stopped, then launched warm | e2e | a relaunch running a done call again (`relax` and `fc` taken over, each `.continued-from` and index entry naming the run that computed it, its start and end that run's; the launch runs only the optical calls); an optical call's record standing on what only a run computing its own force constants writes |
| `raman_view`'s Kramers–Kronig on a Lorentz oscillator's $\varepsilon_2$ (closed-form $\varepsilon_1$); Gaussian sticks spread at $\sigma_{\mathrm{run}}$ then to $\sigma_{\mathrm{view}}$ equal the same spread at $\sigma_{\mathrm{view}}$ | API | the principal value, the units or the broadening's composition wrong |
| the CO₂ run exported at a laser — NetCDF and CSV — its files read back through `postprocess.read_optical_data` and `read_raman_data` and through `xarray`: the view read back equals the load door's for the same laser and width, `derived_from` names the two files by their sha256, and the stretch whose oxygens move together reads $R(\hat z)$ ≈ 0 | e2e (the CO₂ run) | an export that writes another number than the page shows; a data file a notebook cannot open; a view cut from its source |
| a deck with `%block … < coordinates.fdf` read through the fdf reader equals the same deck written inline; `deck_atoms` on it | API | the include not read: every deck reader would see no atoms |
| a SIESTA vibration described flat through `molbuilder describe` — and a PySCF one (`tests/test_describe.py`); the same saved flat through Task setup's Save, after its prep (`tests/data/prep_protocol.toml`) | API | a SIESTA vibration's calls run in one directory, each writing over the last one's `.DM` and `.XV`; the shape card offering *flat* and Save saying another sentence than the card |
| the CO₂ run's calculation opened in the Results tab: the ladder's rows (`tests/test_siesta_vibration_e2e.py`) | e2e | every call's row reading `run-0` alone — eight rows a person cannot tell apart |

---

## 6. The result file — `<label>.spectra.json`

One file, every engine, **written by the job itself** — by the PySCF deck,
and on SIESTA at the run's root by its own Python, the force-constant call's
record step and then the finish (§ 5.5, § 5.11 § 3); read by `sidecars.spectra.parse_spectra_json` →
`SpectraResults.from_dict` (`spectra/results.py`), and served to the browser by
`POST /api/spectra/load` with one derived field added at load (§ 6.6) and the
views a person asks beside it — the infrared of named atoms, the Raman at a
laser (§ 5.10 ②, § 5.11 § 10). It is
registered in [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 6.1.

### 6.1 Schema history

| version | date | change |
|---|---|---|
| 1 | — | one eigenvector per mode, used for both the animation and the Raman projection — recorded as a correctness defect when the two uses were separated |
| 2 | — | `eigenvector_canonical` ($\sum m\lvert L\rvert^2 = 1$) and `eigenvector_display` ($\max\lvert L\rvert = 1$) per mode; the per-mode reader still accepts a v1 file's single vector as both, best effort — unreachable today behind the version gate below |
| 3 | 2026-05-21 | `fixed_atom_idxs` → `frozen_atom_idxs`; no backward compatibility |
| 4 | 2026-05-22 | `runtime_info` (CPU, threads, GPU, host) |
| 5 | 2026-08-20 | the optional `relaxation` and `thermo` blocks and `phase_relaxation` — additive, a v4 file reads whole |
| 6 | 2026-09-23 | `removed_motions`, `hessian_scope`, `n_atoms_in_hessian`, `hessian_density_fit`; the equilibrium block optional as a whole; `eigenvector_display` derived when a file carries only the canonical form — additive |
| **7** | 2026-10-10 | `equilibrium.masses_amu` — every atom's mass, as the deck stated it and the analysis weighted by it (I3); the derived per-mode keys — the activity classes, the zero-point amplitude and displacement — **written by both engines** through one function and checked on read (§ 6.6), where until now the SIESTA finish wrote them and the PySCF deck did not. A v4–v6 file reads as before: no masses, its derived keys computed on read |
| **8** | 2026-10-14 | the SIESTA route's `engine_metadata` names its force-constant range in SIESTA's numbering, `fc_range_engine_1based` — the deck's held-first order, an engine fact and not an atom's identity ([`model/overview.md`](?doc=model/overview.md) § 2.2a). A v4–v7 file reads as before, its range read under v8's name (`SpectraResults.from_dict`) |
| **9** | 2026-10-18 | **the result states its structure whole** — `structure`, `Structure.to_dict()` of the equilibrium structure (§ 6.10.1): the geometry, the box, the labels with the held atoms as `frozen_atoms`, the identity columns, `info`; `structure_hash`, `frozen_atom_idxs` and `equilibrium.{elements, positions_ang}` are read from a v4–v8 file and never written (`frozen_atom_idxs` still in a PySCF file written before any geometry exists, § 6.10.2); `relaxation.record` carries the relaxing stage's record whole. The reader serves every file as this shape |
| **10** | 2026-10-23 | **the result states its atom index map** — `atom_index_map` ([`model/overview.md`](?doc=model/overview.md) § 2.2b): listed by user index, each atom's engine index its stages' deck order (held atoms first for SIESTA's; the user order for PySCF's) — always written, each column where it is known; a v4–v9 file reads, its map made on read from its structure (the user column) and its stages' deck (the engine column) |
| **11** | 2026-10-24 | **the force-constant run's charges** — `partial_charges`: every atom's Hirshfeld and Voronoi charge at every step, in the user order, beside each step's displaced atom, axis and side (§ 5.10 ②); `null` on PySCF and from a deck that asked for none. A v4–v10 file reads as before, `partial_charges` null |
| **12** | 2026-10-24 | **each mode's dipole change, whole** — `modes[].dipole_derivative_d_per_ang_amu12`, $d\mu/dQ$ in D/(Å·√amu), three components, on PySCF where infrared was computed (§ 4.6); `null` on SIESTA, whose infrared is derived from the charges (§ 5.10 ②). A v4–v11 file reads as before, the key absent |
| **13** | 2026-10-26 | **the optical calls' citation** — `optical`, on a SIESTA run in phases whose optical calls were planned (§ 5.11 § 8): the data file the calls are recorded in, `<label>.optical.nc`, and its sha256; the light's direction; $\delta$; the selection as asked and as applied; frame 0's state and, per mode the selection applied, its frequency, $\delta Q$ and each side's state. The Raman strength itself stays derived on request (§ 5.11 § 10); `null` where no optical call was planned and on PySCF. A v4–v12 file reads as before, `optical` null |

`READABLE_SCHEMA_VERSIONS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}`; an older or unknown version is
refused by name.

### 6.2 The keys, and who writes each

| key | written by | meaning |
|---|---|---|
| `schema_version` · `engine` · `engine_version` · `molbuilder_version` · `timestamp` | both | provenance; `engine` is `pyscf` or `siesta` |
| `structure` | both, from v9 (§ 6.10.1) | the equilibrium structure whole, `Structure.to_dict()`: the geometry the Hessian is taken at, in the user order, its box, its labels — the held atoms as `frozen_atoms` — its identity columns and `info`. A v4–v8 file's is built on read from its old keys. `structure_hash` (v4–v8) is read, never written |
| `n_atoms_total` · `free_atom_idxs` · `frozen_atom_idxs` | both | 0-based, in the **user** order (SIESTA: mapped back through the record); the two lists partition `range(n_atoms_total)` and the reader refuses otherwise. From v9 the held list is the structure's label, and `frozen_atom_idxs` is written only before any geometry exists (§ 6.10.2) |
| `atom_index_map` | both, from v10 | which index the result's atoms are listed by — `user_index` — and each atom's user and engine index (`model/overview.md` § 2.2b); the SIESTA finish from its deck's order record, the PySCF deck the identity |
| `equilibrium.{scf_energy_eh, mo_energies_eh, homo_idx}` | PySCF | the reference SCF and its orbital ladder — the **energy subgroup** of the equilibrium block, which travels whole or not at all: **`null` in all three on SIESTA** (a periodic engine has a Fermi level, not a HOMO), and `null` in one slot only is a broken file. The two geometry keys below are separate and always present |
| `equilibrium.{elements, positions_ang}` | v4–v8 | the geometry the Hessian is taken at (§ 4.2); from v9 it is the structure's, which the animation draws from |
| `equilibrium.masses_amu` | both, from v7, wherever the geometry is | every atom's mass in amu, in the user order — the masses the analysis weighted by, as the deck states them (I3): one positive finite number per atom, refused otherwise (§ 6.7). What a mode's frame set's `mass_amu` channel is copied from ([`model/structure.md`](?doc=model/structure.md) § 2.2f) |
| `modes[]` | both | ascending frequency; every entry a vibration (§ 6.3) |
| `removed_motions.{count, patterns}` | both | what the harmonic analysis removed before diagonalising: the count and the orthonormal Cartesian patterns over the free atoms; `len(modes) = 3·n_free − count` by construction |
| `hessian_scope` · `n_atoms_in_hessian` · `hessian_density_fit` | both | `free` (second derivatives for the free atoms only) or `all`; how many; whether the Hessian itself was density-fitted (`false` on the reduced route, `null` on SIESTA) |
| `ir_route` · `ir_fd_step_ang` · `raman_route` · `raman_fd_step_ang` | both | which route produced each strength and the step of a difference (§ 4.6) — on SIESTA `ir_route` `charges` (the infrared derived from the stored charges, § 5.10 ②) and `raman_route` `optical` (§ 5.11 § 10); `none` when not computed; an older file reads `""` — absence of a record, never a claim (`raman_*`: § 10) |
| `phase_relaxation` · `phase_frequencies` · `phase_raman` · `phase_ir` · `phase_es` | both | `empty` · `running` · `complete` · `not requested` (§ 4.9), and `phase_ir` `""` in a file written before 2026-09-28; the SIESTA writer writes `complete` for the frequencies, `not requested` for the probe, and for Raman `running` once its optical calls are planned and `complete` at the finish — `not requested` with *Compute Raman* off, `complete` with no call where the range kept no mode (§ 5.11 § 8); for the relaxation `complete` when the run's `relax` call ran (its record's `n_steps` in `relaxation.n_steps`, `enabled` true) or `not requested` when the box was ticked |
| `relaxation.{enabled, already_relaxed, n_steps, max_force_eh_bohr, max_force_all_atoms_eh_bohr, converged, warning}` | both | the tracked precondition; the judged force is over the free atoms, in Eh/Bohr (§ 4.3), and `converged` is its verdict against the criterion on every route — after the relaxation, under the person's statement, at SIESTA's reference step (§ 4.2, § 4.3, § 5.5); SIESTA writes `enabled` true when the run's `relax` call ran — its relaxation record, from the run's index (§ 5.11 § 3), gives `n_steps` — and false when the box was ticked, `already_relaxed` as the person's statement, the judged force and verdict of § 5.5, and the warning the viewer shows in the phase's row |
| `thermo` | both | `regime`, the headline temperature — and pressure for `rrho`, `null` for `vibrational-only` — with `zpe_eh`, `h_eh`, `s_eh_k`, `g_eh`, `n_modes`, `n_imag_excluded`, `n_rigid_removed`, `note`, and `grid` (§ 4.7) |
| `selected_mode_idxs_1based` | PySCF | the modes that got the electronic-structure probe |
| `config` | both | how it was computed: what the description held (the PySCF config as a dict); on SIESTA the engine, the kind (`calculation`) and the stage, and `deck` (since 2.47.0) — the force-constant deck: its `name` and `sha256` (the deck file's alone); its `keywords` and `blocks`, every keyword and block it states as written, by their spelling, but its geometry (`parse.fdf.GEOMETRY_KEYS`: the coordinates and the lattice, which the result states from the structure) — the functional, the basis, the mesh, the k-points, the species; and its `pseudopotentials`, each species' `<label>.psml` SIESTA opened beside the deck, by name and sha256 (since 2.47.1) |
| `methods_text` · `bibliography_keys` | both | the composed paragraph and the keys it cites (§ 4.10) — at load, a view shown adds its sentence (§ 5.10 ②, § 5.11 § 10), the Raman's with its citation among the keys |
| `engine_metadata` | both | engine-specific facts; SIESTA: `fc_file`, `fc_displacement_ang`, `fc_range_engine_1based`, `fc_asymmetry_max_ev_ang2`, `reference_force_criterion_ev_ang` (§ 5.5). `fc_range_engine_1based` is `FC.First … FC.Last` in SIESTA's numbering — the deck's held-first order, 1-based — named as the engine's here and in `fc-sweep.json`'s stage rows: an engine fact, not an atom's identity ([`model/overview.md`](?doc=model/overview.md) § 2.2a) |
| `optical` | SIESTA, from v13 | `{data_file, data_sha256, direction, displacement_ang, selection, reference, modes}` — the optical calls' citation (§ 5.11 § 8): `data_file` the calls' data file at the run's root and `data_sha256` its bytes', `direction` the light's three components (`Optical.Vector`), `displacement_ang` $\delta$, `selection` `{asked, applied}`, `reference` frame 0's state, `modes[]` `{mode, frequency_cm1, dq_amu12_ang, plus, minus}` — each side's state `finished`, `failed` or `pending`, by the run's one done rule. `null` where no optical call was planned, and on PySCF; refused when a key is not one of these or a state not one of the three |
| `partial_charges` | SIESTA, from v11 | `{displacement_ang, steps, charges_e}` — the force-constant run's steps in SIESTA's order (§ 5.4): `steps[i]` is `{atom, axis, side}`, the displaced atom by user index, `x`/`y`/`z`, −1 or +1, step 0 the reference `{null, null, 0}`; `charges_e.hirshfeld[i]` and `charges_e.voronoi[i]` every atom's charge at step `i`, in e, positive where electrons left the atom, in the user order (§ 5.10 ②). `null` when the deck asked for none (*Compute IR* off), and on PySCF; a deck that asked and an output that printed none is the record step's error, the force-constant call then not done |
| `runtime_info` | PySCF | CPU, threads, GPU, host |

### 6.3 A mode

| key | meaning |
|---|---|
| `index_1based` · `frequency_cm1` | a negative wavenumber **is** an imaginary mode (a saddle, not a minimum), reported, never dropped; `has_imag` says so |
| `eigenvector_canonical` | `(n_free, 3)` Cartesian, $\sum m_k\lvert L_k\rvert^2 = 1$ in amu — the science form: intensities, the physical amplitudes and the element shares pair with **this** one |
| `eigenvector_display` | the same, rescaled so its **largest absolute Cartesian component** is 1 (not the largest per-atom length: an atom moving off-axis swings up to $\sqrt{3}$ of it) — the animation's exaggerated form only; derived from the canonical form by the reader when absent |
| `ir_intensity_km_mol` · `raman_activity_a4_amu` | `null` when the channel was not computed (not requested, or the engine has none); **`0.0` is a measured absence**, a symmetry-forbidden band's residue |
| `dipole_derivative_d_per_ang_amu12` | each mode's dipole change $d\mu/dQ$, three components x, y, z in D/(Å·√amu) (v12, PySCF where infrared was computed; `null` otherwise): the vector `ir_intensity_km_mol` is the orientation average of, and every light direction's strength is computed from (§ 4.6) |
| `electronic_structure` | the probe of § 4.8: `amplitude_ang` (the push $A$, kept with the probe so the coupling's denominator travels with its numerator — an older provenance table listed it as a top-level mode key; the code has always kept it here), the orbital windows at $-A$, $0$, $+A$, the SCF energies, `homo_index_in_window`; `null` when the mode was not selected |
| `ir_active` · `raman_active` · `activity_class` | **derived**, and from v7 **written by both engines** through one function and checked on read (§ 6.6) |
| `zero_point_amplitude_amu12_ang` · `zero_point_displacement_ang` | **derived**, and from v7 **written by both engines** through one function and checked on read (§ 6.6): the zero-point amplitude of the mode, $Q_{\mathrm{zp}} = \sqrt{\hbar/2\omega}$ in amu^½·Å — $\sqrt{\texttt{ZERO\_POINT\_Q2\_AMU\_ANG2\_CM1}/\tilde\nu}$, the constant 16.858 amu·Å²·cm⁻¹ derived in `constants.py` from the two constants the wavenumber conversion uses plus the Bohr radius — and the Cartesian displacement of every free atom at that amplitude, $Q_{\mathrm{zp}}\cdot L_{\mathrm{canonical}}$ in Å: the **mass-calibrated displacement** a vibration-coupled transport step moves the structure along (the thermal amplitude is this times $\sqrt{\coth(\hbar\omega/2k_BT)}$, [`web/spectra.md`](?doc=web/spectra.md) § 4.1). `null` for an imaginary mode, which has no amplitude |

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
| `equilibrium.mo_energies_eh` | `mf.mo_energy` | a non-finite entry written as null (`sidecars.spectra.finite_or_none`) | Hartree |
| `equilibrium.homo_idx` | `mf.mo_occ` | **derived** — `homo_index` (§ 4.3) | index |
| `equilibrium.positions_ang` | the relaxed (or asserted) geometry | the Hessian's geometry, `COORDS_EQ_ANG` | Å |
| `equilibrium.masses_amu` | — | **stated** by molbuilder at prep — `chemistry.atomic_mass`, the isotope-averaged standard weights — in the deck (the PySCF deck's `MASSES_AMU`, the SIESTA deck's `vibration` block), and written as the analysis used them (I3) | amu |
| `modes[].frequency_cm1` | the eigenvalues of § 4.5 | $\operatorname{sign}(\lambda)\cdot\sqrt{\lvert\lambda\rvert}\cdot 5140.487$; imaginary → negative wavenumber, never dropped | cm⁻¹ |
| `modes[].eigenvector_canonical` | the eigenvectors of § 4.5 | the canonical normalisation, from the one path on both engines (an older table called this column *dimensionless*; for $\sum m\lvert L\rvert^2 = 1$ in amu the unit is 1/√amu) | 1/√amu |
| `modes[].eigenvector_display` | — | **derived** — per-mode rescale to $\max\lvert L\rvert = 1$ | dimensionless |
| `modes[].ir_intensity_km_mol` | $d\mu/dR$ (analytic response, or dipoles at displaced geometries) | **derived** — $d\mu/dQ = \sum (d\mu/dR)\cdot L_{\mathrm{canonical}}$ (`einsum('kai,ka->i', DMU_DR, L_canonical)`), then $42.2561\cdot\lvert d\mu/dQ\rvert^2$ | km/mol |
| `modes[].dipole_derivative_d_per_ang_amu12` | the same $d\mu/dR$ | **derived** — $d\mu/dQ = \sum (d\mu/dR)\cdot L_{\mathrm{canonical}}$, the vector itself, before it is squared (§ 4.6) | D/(Å·√amu) |
| `modes[].raman_activity_a4_amu` | polarizabilities at displaced geometries, in Bohr³ | **derived** — central differences, the Placzek scalar, one global $(\mathrm{Bohr}/\text{Å})^6 \approx 0.021959$ | Å⁴/amu |
| `modes[].zero_point_amplitude_amu12_ang` · `zero_point_displacement_ang` | the mode's frequency and canonical vector | **derived** by the one function both writers call (§ 6.6) — $\sqrt{16.858/\tilde\nu}$, times $L_{\mathrm{canonical}}$ per free atom (§ 6.3) | amu^½·Å · Å |
| `relaxation.max_force_eh_bohr` (SIESTA) | the first `siesta: Atomic forces` block of the run's output, its FC step 0 | **read** by the finish (§ 5.5) — the largest over the free atoms, eV/Å → Eh/Bohr; `converged` against the tolerance the reference geometry was relaxed to (§ 5.5) | Eh/Bohr |
| `engine_metadata.fc_asymmetry_max_ev_ang2` (SIESTA) | the `.FC` file | **derived** — $\max\lvert H_{ij} - H_{ji}\rvert$ over the free block before symmetrisation (§ 5.5) | eV/Å² |
| `engine_metadata.reference_force_criterion_ev_ang` (SIESTA) | the tolerance the reference geometry was relaxed to (§ 5.5) | **read** by the finish from the deck's `vibration` block — the criterion the verdict used, so the verdict carries its provenance (§ 5.5) | eV/Å |
| `modes[].electronic_structure.amplitude_ang` | — | the push $A$ of § 4.8, molbuilder's own choice (`displacement_amplitude_ang`), recorded with the probe it produced | Å |
| `modes[].electronic_structure.mo_energies_*_eh`, `scf_energy_*_eh` | `mf.mo_energy`, $E$ at $\pm A$ | a non-finite entry written as null (`sidecars.spectra.finite_or_none`); the shift and the coupling $\Delta E/(2A)$ are the viewer's arithmetic | Hartree |
| `removed_motions` | — | **derived** by the one rule (§ 4.5), on both engines | count · (count, n_free, 3) |
| `thermo` | PySCF's `thermo.thermo` (rrho) or the one home's vibrational sums | the deck computes, the viewer draws; the headline is a row of the grid (§ 4.7) | Eh, Eh/K |
| `relaxation.max_force_eh_bohr` (PySCF) | the nuclear gradient at the judged geometry | the largest force over the **free** atoms; the all-atom figure beside it | Eh/Bohr |
| SIESTA's $H_{AA}$ | `.FC` rows in eV/Å² | mean of the two sides, symmetrised, converted (§ 5.5) | Hartree/Bohr² |
| `partial_charges.charges_e` (SIESTA) | each step's *Hirshfeld* and *Voronoi Atomic Populations* blocks of the run's output | **read** by the finish through SIESTA's one reading pass, put in the user order through the deck's order record; the infrared of a set of atoms is **derived** from them when asked (§ 6.6) | e |

**Two rules this table enforces** *(2026-09-09)*: a number molbuilder only
passes through is not ours to test — a test asserting the magnitude of
`scf_energy_eh` asserts PySCF; what is ours is that it reaches the right key in
the right unit, unrounded. And a number molbuilder **derives** must have its
rule callable, not embedded in script text: `homo_index`, the harmonic path,
the thermo sums and the hash are imported by the deck from `mb_pyscf.pyz`, so
one implementation runs and is tested. The IR and Raman scalars are still inline, because neither
has a branch — the trigger `siesta/makov_payne.py` states: copy a branchless
formula if you must, make it a callable that travels as its module once it has a branch; the same ruling
covers the sign convention for imaginary modes and the free/held index
bookkeeping, both branchless. If any of them grows a branch — a second
polarizability convention, a per-mode prefactor — it moves to a callable the
way `homo_index` did.

### 6.5 What a SIESTA file looks like beside a PySCF file

| | PySCF | SIESTA |
|---|---|---|
| `equilibrium.scf_energy_eh`, `mo_energies_eh`, `homo_idx` | numbers | **`null`, all three** |
| `modes[].ir_intensity_km_mol`, `raman_activity_a4_amu` | numbers, or `null` when not requested | **`null` on every mode** |
| `modes[].dipole_derivative_d_per_ang_amu12` | the vector, where infrared was computed (v12) | `null` — the infrared of a set is derived from `partial_charges` |
| `modes[].electronic_structure` | present on selected modes | `null` |
| `ir_route` / `raman_route` | `analytic` · `finite-difference` · `none` | `ir_route` `charges` once the charges are stored, else `none`; `raman_route` `optical` where the run's optical calls were planned (§ 5.11 § 8), else `none` |
| `thermo.regime` | `rrho` or `vibrational-only` | `vibrational-only`, the vibrational contributions alone (no total energy is reported, so `h_eh` and `g_eh` are sums above an electronic minimum taken as zero) |
| `relaxation` | the tracked phase | the force-constant run relaxes nothing: `enabled` says whether the run's `relax` call did (§ 5.11 § 3), and the verdict is the force at FC step 0 (§ 5.5) |
| `hessian_density_fit` | `true` / `false` | `null` |
| `equilibrium.masses_amu` | the deck's `MASSES_AMU`, every atom | the deck's `vibration` block, every atom |
| `modes[].ir_active` · `raman_active` · `activity_class` | from the strengths computed | `partial` — no strength is computed (§ 6.6) |
| `engine_metadata` | `{}` | `fc_file`, `fc_displacement_ang`, `fc_range_engine_1based` (SIESTA's numbering), `fc_asymmetry_max_ev_ang2`, `reference_force_criterion_ev_ang` |
| `partial_charges` | `null` — the infrared is the dipole's own | every atom's charge at every force-constant step (v11): the infrared of a set of atoms along a direction is derived from them when asked (§ 5.10 ②), `ir_intensity_km_mol` staying `null` |
| what the viewer shows | the spectrum of the strengths computed, the modes table, the animation, the probe's level diagrams, the thermochemistry | **the mode positions** — one line of one height at each mode, picked like a stick, with no width ([`web/spectrumchart.md`](?doc=web/spectrumchart.md) § 6.2) — and, where the result states its charges, the infrared lane and columns of the atoms the person names ([`web/spectra.md`](?doc=web/spectra.md) § 3.2); on a run with optical calls, the laser and width boxes, the Raman column and lane at the laser named, and the export of its data files ([`web/spectra.md`](?doc=web/spectra.md) § 3.3); the modes table and the animation; the electronic-structure tab says the route's electronic response is the projected density of states along a mode, planned (§ 5.10); the thermochemistry as vibrational contributions ([`web/spectra.md`](?doc=web/spectra.md) § 2, § 9b.3) |

**A missing number is absent, never zero.** A key an engine cannot produce is
`null`, and a reader treats it as *not computed* — a different statement from
`0.0`, which is a measured absence (a symmetry-forbidden band). How each
surface draws an absent number is [`web/spectra.md`](?doc=web/spectra.md)
§ 9b.3: mode positions instead of a spectrum when no mode carries a
strength (a SIESTA result until its infrared is asked of a set of atoms or its Raman at a laser) — one line at each mode, no heights, no
broadening, no width control — no column for a channel the route cannot
compute, a `—` in a column it can, the `partial` colour on the rug, and a
write-up that names what was computed.

### 6.6 Derived values — written beside what they come from, checked on read *(user, 2026-10-10: "explicit is always better than implicit in data science")*

The activity classes (`ir_active`, `raman_active`, `activity_class`) and a
mode's zero-point amplitude and displacement are **derived** — from the
strengths, the frequency and the canonical vector — and from v7 **both
writers store them**, computed by one function both bundles carry
(`spectra/derived.py`, which the SIESTA finish reaches through
`SpectraResults` and the PySCF deck imports from `mb_pyscf.pyz`): a result
file is a full record a person reads without molbuilder (§ 6.8). The reader
computes them again with the same function and refuses a file whose stored
value disagrees (§ 6.7) — a file never carries a number its own inputs
contradict. A file before v7 reads as before, each computed on read — the
PySCF deck's stated none, and what the SIESTA finish's stated is computed
again in its place, unchecked.

**The activity class is a decision, so its rule is part of the file's
definition**: changing the rule below is a new schema version, never a silent
re-derivation of files already written. Whether a band is active is decided,
not read, because a symmetry-forbidden mode's stored strength is
floating-point residue (measured on CO₂: 4.8·10⁻⁹ to 7.5·10⁻⁸ beside bands of
32.85 and 613.04 km/mol). The rule: a mode whose channel was not computed is
`partial` — every SIESTA mode, whose infrared and Raman are derived at load
from what the run stored, never classified into the file (§ 6.5);
otherwise, per channel, divide every mode by the channel's strongest, sort on
a log scale, and cut at the widest gap between neighbours when that gap is at
least two decades wide and sits below a thousandth of the peak; when the
channel will not separate itself, cut at a millionth of the peak; and a
channel whose strongest value is under an absolute floor (10⁻³ km/mol,
10⁻³ Å⁴/amu) has no band at all. It is asked of the data because where the
residue sits is a property of the calculation (measured cuts 2.6·10⁻⁶ to
1.5·10⁻⁴ across four real runs) while the separation is a property of the
symmetry. Built 2026-09-11, confirmed as the rule 2026-09-23; pinned on CO₂ by
`tests/spectra/test_activity.py`.

**The element shares are a view, added at load**: `motion_share_by_element` —
each element's share of the mass-weighted motion, $m_i\lvert L_i\rvert^2 / \sum m_k\lvert L_k\rvert^2$
([`web/spectra.md`](?doc=web/spectra.md) § 4.2). It and a displacement
sweep's mode matching (§ 5.9) weight by **one rule**, `results.mass_of`: the
file's own masses (§ 6.2), and `chemistry.atomic_mass`'s for a file before
v7, which states none.

**The infrared of a set of atoms, and a mode's reference displacement, are
views too** — each asked with its inputs and derived by one function, never
stored: `spectra.infrared.infrared_view`, from `partial_charges`, for a set
of atoms, a direction and a partition (§ 5.10 ②); and
`mode_frames.reference_max_displacement`, the maximum displacement whose
frames sample the mode's spread at a temperature, from its zero-point
amplitude and its pattern (`science/vibrational-averaging.md` § 5). The load
door serves each (`web/spectra.md` § 3.2, § 4.3).

### 6.7 What the reader refuses

`SpectraResults` is built at the boundary and refuses, by name: a schema
version outside `{4, 5, …, 13}` (`READABLE_SCHEMA_VERSIONS`, § 6.1); an `optical` citation in a file older than schema 13, or one with a key not its own or missing, a mode row likewise, a direction not three numbers, no `displacement_ang`, or a frame state not `finished`, `failed` or `pending` (`checked_optical`, § 6.2);
`partial_charges` whose steps are not the force-constant run's — $1 + 6\,n_{\mathrm{free}}$
of them, step 0 the reference, then each free atom along x, y, z, side −1
before +1 — or whose charges are not one finite number per atom at every
step, or name a partition other than `hirshfeld` and `voronoi`; masses that are not one positive finite
number per atom, or that come without the geometry they belong to; in a v7
file, the geometry without its masses — `equilibrium.masses_amu` travels with
`equilibrium.positions_ang`, so an in-progress file not yet at its geometry
states neither — a mode that does not state one of its derived values, and a
stored derived value (§ 6.6) that disagrees with its derivation beyond a
relative $10^{-9}$; free and frozen lists
that do not partition
`range(n_atoms_total)` (a count-only check once passed `free = [0, 1, 5]` for
three atoms and the viewer silently dropped a displacement); a mode whose
eigenvector does not carry one row per free atom; an equilibrium energy
subgroup (`scf_energy_eh`, `mo_energies_eh`, `homo_idx`) with `null` in some
slots and numbers in others; and a key it does not know, at the top level and in every block it reads
field by field — `equilibrium`, each mode and its probe, `partial_charges`,
`optical` — because a misspelled
`ir_intesity_km_mol` used to serve a chart titled *not computed* with every
number present and thrown away, and a file claiming 10¹² atoms used to be
answered with a `MemoryError`. `/api/spectra/load` turns each refusal into a
typed error (missing → 404, wrong version → 422, malformed → 400) so the viewer
reacts without parsing a message.

### 6.8 Reading the file yourself

The result is **plain JSON** (UTF-8, `indent=2`, no `NaN`: a non-finite number
is refused before the file is written), so any language reads it, and it sits
where the run ran: `<attempt>/<label>.spectra.json` — `run-N/` under the
stage's folder in the hierarchical layout, the calculation folder in the flat
one. It stays one JSON file rather than a table because a mode is not one row:
its frequency sits with an `(n_free, 3)` displacement pattern and the
thermochemistry with its temperature curves, which a CSV would split across
files. *(Schema 9 writes a NetCDF beside it, the same result as structured
data for analysis and fast loading — § 6.10, decided 2026-10-14.)*
The units are in the key names (`_cm1`, `_ang`, `_eh`, `_km_mol`,
`_a4_amu`); § 6.2–6.4 say what each key is and where its number comes from.

```python
import json
d = json.load(open("siesta_PDT.spectra.json"))
for m in d["modes"]:                        # ascending; a negative wavenumber is imaginary
    print(m["index_1based"], m["frequency_cm1"], m["ir_intensity_km_mol"])   # None: not computed
L = d["modes"][0]["eigenvector_canonical"]  # one row per FREE atom, in the order of:
free = d["free_atom_idxs"]                  # atom indices, from 0, by the file's map:
atoms = d["atom_index_map"]                 # v10: atoms_listed_by "user_index", each atom's engine index
xyz = d["structure"]["positions"]           # every atom, the geometry the modes are taken at (v9)
m = d["equilibrium"]["masses_amu"]          # every atom's mass, amu -- what L is weighted by (v7)
q_zp = d["modes"][0]["zero_point_amplitude_amu12_ang"]   # sqrt(hbar / 2 omega), None if imaginary
```

With molbuilder importable, a script's one import is the post-processing
layer, `molbuilder.postprocess` ([`model/parse.md`](?doc=model/parse.md) § 3):
`read_spectra_data("<label>.spectra.nc")` the same result as data with its
atom index map, and `atom_index_map(path)` the map of either file — the
reader's answer, which adds what the run's deck states (F3) to what the file
states of itself (`SpectraResults.atom_index_map_stated`, a result before
v10 its user column alone). molbuilder's own read,
`molbuilder.sidecars.spectra.parse_spectra_json(path)`, returns the typed
`SpectraResults`, through the gates of § 6.7.

**A SIESTA run's optical calls and its Raman** (§ 5.11 § 8, § 10): the
optical calls' data file `<label>.optical.nc` beside the result, every
frame's $\varepsilon_2$ on the light's energies, and an exported view
`<label>.raman.nc`, each a NetCDF-4 file any reader opens:

```python
from molbuilder.postprocess import raman_view, read_optical_data, read_raman_data
v = raman_view("siesta_PDT.spectra.json", laser_nm=633)   # the run's own width
for m in v.modes:
    print(m.index_1based, m.frequency_cm1, m.strength_laser_a4_amu, m.relative)
opt = read_optical_data("siesta_PDT.optical.nc")         # every frame, its eps2
r = read_raman_data("siesta_PDT.raman.nc")               # an export, as data

import xarray as xr                                       # no molbuilder needed
frames = xr.open_dataset("siesta_PDT.optical.nc", group="frames")
eps2 = frames.eps2.sel(frame=0).sum("spin")               # frame 0, both spins
```

### 6.9 What each file of a vibration holds

Where the data of a vibration calculation is, file by file — what a person
opens, and what a program reads. Every name molbuilder composes comes from
the run-file catalogue (`runfiles.WRITTEN`), which Task setup's *what this
calculation writes* card lists; SIESTA's own `.FC` / `.FCC` are named by
SIESTA from the label (§ 5.4).

| file | where | holds | written by |
|---|---|---|---|
| `<label>.spectra.json` | the run's root on SIESTA (§ 5.11 § 2); the attempt on PySCF | **the result**: every mode's frequency and both eigenvector forms, the strengths the engine computes (`null` where not computed), the removed motions, the thermochemistry, the stationarity verdict, the steps of every finite difference (§ 6.2–6.4); read it as § 6.8 says | the run itself (§ 5.5) |
| `<label>.spectra.nc` | beside the `.spectra.json`, written with it at every write | **the same result as structured data** (§ 6.10.5): NetCDF-4 groups `/structure`, `/equilibrium`, `/modes` (`/modes/electronic_structure`), `/thermo` (`/thermo/grid`), `/partial_charges` where the result states them, every variable stating its units, long name, definition and kind, the scalar blocks' numbers as variables of no dimension and their texts, counts and flags as attributes, the structure whole as JSON text beside its arrays, the JSON's sha256 | the run itself, through the one writer (`sidecars.spectra.write_spectra_payload`) |
| `<label>.optical.nc` | the run's root, a SIESTA run whose optical calls were planned | **the optical calls as data** (§ 5.11 § 8): every frame's $\varepsilon_2$ over the photon energies, its mode, side, $\delta Q$ and geometry, its state and the run that computed it, its SCF facts, and the settings the calls read; the result cites it by name and sha256 | the force constants' record step, every frame pending; each optical call's record step; the finish (`spectra.optical_data.write_optical_data`) |
| `<label>.fc-sweep.json` | the calculation root | a SIESTA displacement sweep's comparison: each stage and what it varied (with units), every mode's frequency per stage matched by shape, how far the force constants moved, the paths of every stage's own files, and the stages without a result with their state (§ 5.9) | `jobset summarize task` |
| `<label>.FC` · `<label>.FCC` | each SIESTA force-constant attempt | **the raw force constants**: one header (atoms, the displacement in Å), then one row of three numbers in eV/Å² per displaced atom, direction, side and atom (§ 5.4); `.FCC` the same with the held atoms' rows zeroed | SIESTA |
| `<label>_<NN>_<stage>-run<N>.out` | each attempt | the engine's own account — every displacement's SCF and forces; FC step 0 is the reference geometry and its forces (§ 5.5) | SIESTA's stdout |
| `<label>_<NN>_<stage>-run<N>.pyscf.log` · `<label>_<NN>_<stage>.log` · `<label>_<NN>_<stage>_geom_optim.xyz` | the PySCF attempt — in the flat shape the last two carry the run's number too, `<label>_<NN>_<stage>-run<N>.log` (`engines/pyscf.md` § 2) | the engine's own account; PySCF's own log; geomeTRIC's trajectory | PySCF |
| `<label>_<NN>_<stage>.fdf` | each SIESTA attempt | the deck, and in its record the `engine-offset` and `vibration` blocks and the ATOM-METADATA block's order record — the order the deck is written in and the way back to the user order — which the finish reads (§ 5.2, § 5.3; `model/overview.md` § 2.2a, I5) | `prep` |
| `<label>_<NN>_<stage>.runwrap-<stamp>.log` | each attempt | the session log: the wrapper's lines, the monitor's start, and the finish's (§ 5.5) | the wrapper |

### 6.10 Schema 9 — the result states its structure, in a JSON and a NetCDF *(decided 2026-10-14, built 2026-10-18)*

> **Decided with the user on 2026-10-14** *(plan V1.39)*, built in 2.13.3,
> 2.14.0 and 2.15.0 — the reader, the consumers, both writers and the data
> file.  What remains is the person's: each machine's job environments
> repaired for `netCDF4` (§ 6.10.6, `molbuilder envs install`). The user's words: *"why
> not save the whole structure with labels all in that json rather than just
> having the fingerprint? the single json would then contain all the
> information … having one single json file storing everything removes the
> requirement of having a fingerprint of a structure that needs to be found
> somewhere that could be unreliable"*; *"this would also make the later
> planned step of generating a multi-frame structure data for series of
> transport calculation from a vibration result easy, because a single .json
> file would be enough for citation"*; and *"add both netcdf and json writing,
> make sure netcdf contains meta data and are structurally clean and
> organized with hierarchical structure"*. Revised the same day after two
> review agents read it against the code; the sections it names describe
> the files as written.

**Why.** A schema-7 result names the geometry it is about by
`structure_hash`, a hash of a structure kept elsewhere, carrying the job's
name — and its two writers hash different geometries: the SIESTA finish the
force-constant reference geometry, the PySCF deck its input atoms, before any
relaxation it runs (§ 4.3). And the file carries no cell, axis kinds, labels
or identity columns: a periodic SIESTA result cannot be opened as its
structure from the file alone, and the frame-set generator (§ 5.10 ③) would
read the labels — the transport partition among them — and the cell from
another file, trusting it to match. **A result that states its structure
whole needs no fingerprint, and the one file is the whole citation.**

#### 1. The structure, whole

`structure` is `Structure.to_dict()` of the **equilibrium structure** — the
one loss-free serialiser, read back by `Structure.from_dict`
([`model/structure.md`](?doc=model/structure.md) § 2.1):

| part | what it is |
|---|---|
| the coordinates | the geometry the Hessian is taken at (§ 4.2): on PySCF `COORDS_EQ_ANG` — the relaxed geometry when the deck relaxed, the input otherwise; on SIESTA the force-constant reference geometry, FC step 0 (§ 5.5) — **in the user order**, SIESTA's put back through its deck's order record (§ 5.2) |
| the labels | as the run was given them: the reserved `frozen_atoms` — **the atoms this calculation held**, the partial Hessian's fixed set ([`science/normal-modes.md`](?doc=science/normal-modes.md) § 1) — the transport partition, every other label |
| the box | the cell, the axis kinds, the stated engine offset and whose it is, the vacuum — the run's |
| the identity columns and channels | as the run was given them |
| `info` | as the run was given it: **the input's own run records stay behind** — its `relaxation` and `calculation` describe the run the input came out of, at the input's coordinates, not these (`pyscf/input.py::_sidecar_for`) |

**What is not in it, and where it is.** The masses the analysis weighted by
stay the result's own, `equilibrium.masses_amu` (§ 6.2): they are the
analysis's input; the structure's `mass_amu` channel is a mode's frame set's
maker's ([`model/structure.md`](?doc=model/structure.md) § 2.2f; the
frame-set generator copies them there). The relaxation that made the
geometry is the result's `relaxation` block (§ 6.2) — on SIESTA it now
carries the `relax` call's record **whole**, `relaxation.record`, from the
run's index (§ 5.11 § 3), where a schema-7 file kept only its step count.

**How each writer builds it — from what its deck already carries, once:**

- **PySCF**: the deck holds the run's structure as its sidecar payload,
  `_MB_SIDECAR` (`engines/pyscf.md` § 3); the structure is the Hessian's
  elements and coordinates with that payload applied by
  `molstruct.apply_to_structure` — the envelope's door, which takes the
  payload whole. `molstruct.py`'s import of the structure becomes two-way,
  so it loads in a bundle.
- **SIESTA**: the deck already states the labels, channels and customized
  rows (its ATOM-METADATA block) and the box (its ENGINE-OFFSET block), in the
  deck's held-first order; the finish builds the structure from them and the
  reference geometry, puts it back in the user order through the recorded
  permutation (§ 5.2), and takes from the `vibration` block **only what no
  block carries** — the title and the identity columns. Nothing is stated
  twice in the deck (`script_emit.emit_atom_metadata`: *"a key of its own
  would put the same fact in the generated script twice"*). **Checked**: the
  atoms outside `FC.First … FC.Last`, mapped back, are the structure's
  `frozen_atoms` — a deck whose two statements of the held set disagree is
  refused by the finish.
- `structure.py`, `sidecars/molstruct.py`, `persist.py`, `ncdata.py` and
  `spectra/datafile.py` join `mb_vibration.pyz`; `persist.py`, `ncdata.py`,
  `spectra/datafile.py` and `runfiles.py` join `mb_pyscf.pyz` (the structure
  and the sidecar codec travel there already).

#### 2. One home for each fact

| schema 8 | schema 9 |
|---|---|
| `structure_hash` | **retired** — read in a v4–v8 file, never written; the structure is stated instead |
| `equilibrium.elements` · `equilibrium.positions_ang` | the structure's elements and coordinates |
| `frozen_atom_idxs` | the structure's `frozen_atoms` label, once the structure is stated; **its own key until then** — a file written before any geometry exists (below) states its held atoms as this list, as every file before v9 does. The held set is stated, never derived from the count |
| `n_atoms_total` · `free_atom_idxs` · `n_atoms_in_hessian` · `hessian_scope` | **stated and checked** (§ 6.6): the structure's atom count; the complement of its held label in ascending order — the order of every mode's eigenvector rows; the count of free atoms on the reduced route; `free` when atoms are held. A file whose stated value disagrees with its structure is refused (§ 6.7) |
| `equilibrium.masses_amu` | **kept**, beside the structure (§ 1) |

**In-progress files.** A PySCF result is written before any geometry exists
(§ 4.9); `structure` appears with the first write that knows the geometry
the Hessian is taken at, and the checked values are checked from then on —
exactly as `equilibrium.positions_ang` and `masses_amu` travel today (§ 6.7).
Until then the file carries `frozen_atom_idxs`, and a file stating neither is
refused.

**The reader serves one shape.** `SpectraResults` reads both shapes and
answers with one: `to_dict()` emits the v9 shape for every file, a v4–v8
file's `structure` built from its old keys (its elements, coordinates and
held list; no box, no labels — stated absent, never invented). **The order
of migration**: the reader first; then every consumer onto it — the
viewer's animation and the summary (`lib/spectra/core.js` reads
`equilibrium.positions_ang` and `frozen_atom_idxs`), the spectra route's
element shares (`blueprints/spectra.py`), the sweep, the frame-set
generator; then the two writers.

#### 3. The sweep compares the structures

§ 5.9's *one geometry* becomes a comparison of what each stage states: the
same elements in the same order, and every atom within **10⁻⁶ Å** of the
reference stage's; otherwise the sweep is refused, naming the largest move,
as today. Each stage of a SIESTA sweep relaxes in its own run (§ 5.11 § 1)
until a later stage takes the first stage's `relax` call over (§ 5.11 § 6,
not built), so two relaxations are refused here; at one geometry the stages
agree exactly, and the tolerance only absorbs the float formatting of the two
files. A
v4–v8 stage is compared by its `equilibrium.positions_ang`, so a sweep across
the version change still reads. (The sweep is SIESTA's alone: PySCF's second
derivatives are analytic, § 5.9.)

#### 4. A frame set cites one file

§ 5.10 ③'s generator reads the result alone — the base structure with its
labels and box, the masses, the mode. `info.vibration.result_sha256`, the
sha256 of the result's JSON bytes, says which file it was made from; like
every row of a frame set it is the maker's, carried and shown, and no
calculation compares it ([`model/structure.md`](?doc=model/structure.md)
§ 2.2f; the user, 2026-10-18).

#### 5. The data file — `<label>.spectra.nc`

**The same result, as structured data**: NetCDF-4 — one of the data-file
framework's files ([`model/datafiles.md`](?doc=model/datafiles.md): each
variable stating itself, the one writer, the one door; this section lists its
groups and variables) — written by the result's one writer
(`write_spectra_payload`) at every write, from the payload the JSON is written
from — one source, two files, both through `persist`'s one atomic writer (a
temporary beside each, then a rename; plan K13), so the result file is no
longer written by a private writer (plan V1.37). The `.nc` states the sha256
of the JSON bytes written with it, which ties each to its pair at every
phase's rewrite. The JSON stays the result a person reads and the Results tab
loads (§ 6.8); molbuilder itself reads the `.nc` in its round-trip test and
for its atom index map (`model/overview.md` § 2.2b), as a person's script
does.
The NetCDF is for analysis and for loading a large result fast —
`xarray.load_dataset(path, group="modes")`: read whole and the file closed,
so a molbuilder read of the same file in the same kernel is never refused
(one handle at a time, [`model/datafiles.md`](?doc=model/datafiles.md) § 4).

Every axis is a coordinate of its own name, each a row of
[`model/datafiles.md`](?doc=model/datafiles.md) § 2a, so a reader selects by
value — `.sel(mode=7)`, `.sel(temperature_k=298.15)`.

```text
/                     each group its own dims -- atom, cell_vector and xyz in
                      /structure, free_atom, mode, removed and xyz in /modes,
                      mo in /equilibrium, es and window in
                      /modes/electronic_structure, temperature_k in /thermo/grid
                      ir_fd_step_ang  raman_fd_step_ang
                      relaxation_max_force_eh_bohr
                      relaxation_max_force_all_atoms_eh_bohr -- the scalar
                      blocks' numbers, each of no dimension
                      attributes: title, schema "molbuilder/spectra-data@8",
                      molbuilder_version, source -- the engine and its
                      version, Conventions, unit_meanings, json_sha256,
                      spectra_schema_version 13, engine, engine_version,
                      timestamp, label, methods_text, bibliography_keys; the
                      scalar blocks' texts, counts and flags -- phase_*,
                      ir_route, raman_route, hessian_scope,
                      n_atoms_in_hessian, hessian_density_fit,
                      relaxation_<key> for the relaxation block's;
                      config, engine_metadata, runtime_info, optical and
                      relaxation_record as JSON text
/structure            atom(atom) -- the user index  xyz(xyz)  cell_vector(cell_vector)
                      element(atom)  position_ang(atom, xyz)  mass_amu(atom)
                      held(atom)  cell_ang(cell_vector, xyz)  axis_kind(xyz)
                      user_atom_index(atom)  engine_atom_index(atom)
                      attributes atoms_listed_by "user_index",
                      atom_index_base 0 -- the atom index map
                      attribute structure_json: Structure.to_dict(), whole
/equilibrium          scf_energy_eh -- of no dimension  mo(mo)  mo_energy_eh(mo)
                      attribute homo_idx
/modes                mode(mode) -- 1-based, the JSON's index_1based
                      xyz(xyz)  removed(removed)  frequency_cm1(mode)
                      free_atom(free_atom)     -- which atom each row is,
                                                  by user index
                      eigenvector_canonical(mode, free_atom, xyz)
                      eigenvector_display(mode, free_atom, xyz)
                      zero_point_amplitude_amu12_ang(mode)
                      zero_point_displacement_ang(mode, free_atom, xyz)
                      ir_intensity_km_mol(mode)  raman_activity_a4_amu(mode)
                      dipole_derivative_d_per_ang_amu12(mode, xyz) -- v12, where kept
                      ir_active(mode)  raman_active(mode)  activity_class(mode)
                      removed_motion(removed, free_atom, xyz)
    /modes/electronic_structure   the probe of § 4.8, one row per selected mode:
                      es(es) -- the mode's 1-based index  window(window)
/thermo               temperature_K  pressure_atm  zpe_eh  h_eh  s_eh_k  g_eh
                      -- each of no dimension
                      attributes regime, n_modes, n_imag_excluded,
                      n_rigid_removed, note
    /thermo/grid      temperature_k(temperature_k) in K: every curve of § 4.7,
                      of no temperature where the result draws none; attribute
                      curves -- the ones it holds
/partial_charges      the force-constant run's charges (§ 5.10 ②), when the result
                      states them: fc_step(fc_step) -- 0 the reference  atom(atom)
                      displaced_atom(fc_step) -- by user index  displaced_axis(fc_step)
                      displaced_side(fc_step)  hirshfeld_charge_e(fc_step, atom)
                      voronoi_charge_e(fc_step, atom)  displacement_ang -- of
                      no dimension  attribute partitions -- the partitions it
                      holds
```

Every group states its title, and every number its unit
([`model/datafiles.md`](?doc=model/datafiles.md) § 2b–2c): the canonical
eigenvector in `amu^-1/2`, the dipole change in `D/(angstrom amu^1/2)`.

A file of an earlier version is read as it is: `spectra-data@1` and `@2`
name four axes' values `index_1based`, `free_atom_index`, `mode_index_1based`
and `temperatures_K`; `@3` names the temperatures `temperature`; `@1`–`@4`
hold no `/partial_charges`; `@1`–`@5` no `dipole_derivative_d_per_ang_amu12`;
`@1`–`@6` no `optical` citation; `@1`–`@7` keep the scalar blocks' numbers,
`/equilibrium`'s `scf_energy_eh` and `/partial_charges`' `displacement_ang`
as attributes, read from there, and state no title or source.

**The atoms are listed by user index, and the file says so** *(since
2.32.0 — [`model/overview.md`](?doc=model/overview.md) § 2.2b)*: `/structure`
carries the atom index map — each atom's user and engine index, the engine's
the stages' deck order (held atoms first for SIESTA's, the user order for
PySCF's, which sorts nothing) — made by the finish from the deck's order
record and handed to the one writer with the atoms; `free_atom` counts
by user index, marked so. The JSON — schema 10 — states the same map,
`atom_index_map`, at its top. A result written before 2.32.0 (schema 4–9,
`spectra-data@1`) is read as it is, its map made as it is read — the user
column its own (I1: a result lists its atoms by user index), the engine
column from its run's deck where the deck answers, and otherwise not known,
said — and never rewritten: its JSON's hash is named by its
data file and by every frame set made from it (`info.vibration`).

**The structure has one serialiser**: `/structure` holds the arrays an
analysis reaches for and `Structure.to_dict(view=False)` whole — the
structure's own fields, a view's mark left out ([`model/overview.md`](?doc=model/overview.md) § 2.2b, F4) — as JSON text — never
a second encoding of the labels, channels or `info` by hand
([`model/structure.md`](?doc=model/structure.md) § 2.1: *outside this class
nobody assembles or picks apart a structure dict*), so a field the structure
gains later reaches the `.nc` with no edit here. **What was not computed
says so once**, as in the JSON: a fill value where the JSON has `null`, and
the phase and route attributes that already state what ran (§ 6.5) — no
second flag beside them. **Every variable states itself** — `units`,
`long_name`, `definition` and its kind (given, raw, derived) — through the
variable record and the writing loop the transport data file uses
(`ncdata.Var`, `ncdata.write_group` — the record and the loop the transport
data file wrote with, now the module both data files and both job bundles
share); each file keeps its own table of variables
(`spectra/datafile.py`).  A key of the result, its `relaxation` or its
`thermo` block that the file's tables do not know is refused by name — a
fact the writer thinks it saved and the file would drop.
**Encoding**, so the round trip is exact: `null` is the fill value (a
number's NaN, a text's empty string, an integer's stated fill); a boolean is
an `int8` with `flag_values 0 1`; an empty list is a dimension of length 0; a
nested dict is JSON text. **Read back, the NetCDF gives the payload the JSON
holds**, and that round trip is the file's test.

#### 6. Where it is written, and what that needs

Both files are written **by the job**, beside each other in the attempt: a
run is not written into after it ends ([`execution/project-layout.md`](?doc=execution/project-layout.md)
§ 1.5), so nothing on this machine adds the NetCDF later. The writer imports
`netCDF4` in every job environment that writes a result, and each lists it in
its recipe — `molbuilder-pySCF`, `molbuilder-siesta` and
`molbuilder-siesta-gpu` (`envs/recipes.py`; approved by the user,
2026-10-14). The PySCF env's self-test imports it (`import pyscf, geometric,
netCDF4`), and the SIESTA envs' self-tests import numpy and netCDF4
before `siesta --version` (approved by the user, 2026-10-26), so an env that
lacks them fails when it is installed or checked. **A job is stopped early,
never after the expensive part**: the PySCF deck writes its first result
before any heavy work (§ 4.9), and the SIESTA job's `loads` check
(`runwrap._finish_check_block`, § 5.5) imports the finish with numpy and
`netCDF4` before the force-constant run starts — that check is what stops a
SIESTA job whose env lacks them.
The bundle rule — `configuration.md` § 2.3, I23 and the PySCF bundle's rule in
`runwrap.py` — holds `netCDF4` beside numpy, imported at load by `ncdata`,
the data-file framework the result is written through; I15 (one structure
hash) retires.

**Registered** where every file molbuilder writes is: `runfiles.WRITTEN`'s
`.spectra.nc` row, § 6.9's table and `execution/job-contracts.md` § 6.1.

#### 7. Decided with it

- ~~**The held set across a relaxation and its vibration.**~~ *Decided
  2026-10-14*: one held label, the vibration's structure stating its own set,
  and the relaxation's held set in its record — § 2.3a, built.
- ~~**`netcdf4` in the job environments**~~ *Approved by the user 2026-10-14*.

---

## 7. The invariants — what the code must keep true

These are the statements a code review checks the implementation against.
Each names where it holds and what pins it.

| # | invariant | where it holds | pinned by |
|---|---|---|---|
| I1 | **One derivation of the surviving motions** (R1): `rigid_motions` is the only place $n_{\mathrm{rigid}}$ is computed; no call site tabulates it, branches on `len(F)`, or asks whether a molecule is straight | `spectra/normal_modes.py`; the deck imports it; the count said before a run is `normal_modes.mode_count`, asked through `validation.spectra.modes_of` by both preflights, prep's size line and the Methods text; the SIESTA finish calls it | `tests/spectra/test_normal_modes.py` (every row of the science § 7 table, three mutations each); review |
| I2 | **One harmonic path** (R3, R4): both engines hand `vibrational_modes` the block, the masses and the geometry; there is no free-molecule branch and no engine branch after the block | the PySCF deck (imported from `mb_pyscf.pyz`), `spectra/vibrational_analysis.py` (the SIESTA finish) | the rank gate against PySCF on free molecules; the held-water and H₂ end-to-end runs |
| I3 | **One mass convention, stated once**: isotope-averaged masses, decided at prep by `chemistry.atomic_mass` and written into the deck — the PySCF deck's `MASSES_AMU`, which the deck checks against the masses PySCF's own routines use and refuses to run on a difference; the SIESTA deck's `vibration` block, which the finish reads — weighted by, and stated in the result (`equilibrium.masses_amu`); $\sum m\lvert L\rvert^2 = 1$ in amu; one wavenumber constant derived from its parts *(until 2026-10-10 the SIESTA finish read ASE's table itself and the PySCF deck PySCF's — equal numbers from two tables)* | `chemistry.atomic_mass`, the two deck renderers, `constants.CM1_PER_SQRT_HARTREE_BOHR2_AMU` | the BDT pair (C–H stretches equal to 0.001 cm⁻¹ free vs held); the end-to-end runs, SIESTA's and PySCF's, read the result's masses and every mode's $\sum m\lvert L\rvert^2 = 1$ in them |
| I4 | **The Hessian is over the free atoms**, and the run says so (R8): `hessian_scope`, `n_atoms_in_hessian`, `hessian_density_fit` | `spectra/pyscf_vibration.dipole_derivatives`; the FC range on SIESTA | the free-atom-block check in `tests/test_vibration_e2e.py`; `tests/test_siesta_vibration_deck.py` |
| I5 | **Stationarity is judged on the free atoms** (R5), and the number is recorded beside the all-atom one | `_vib_gradient_check`; the relax callback | the held-water run's `relaxation` block |
| I6 | **The held set has one source** — the structure's `frozen_atoms` region — and reaches every phase from it: the PySCF `$freeze` file, the free-atom Hessian, the SIESTA `Geometry.Constraints` and FC range | `VibrationConfigView.frozen_indices`; `fc_facts` | `tests/test_vibration_render_gate.py`; the wrapper's constraint banner reads the deck's one spelling |
| I7 | **The reorder is recorded once and inverted once**: the sorted copy's own `atom_order`, made by `sort_by` and stated by its deck, crossed back only by `Structure.restored()` and the record's own methods (`Permutation.map_rows_to_user_order`); the user order never reaches the engine, the sorted order never reaches a person (`model/overview.md` § 2.2a, I3, I5) | `atom_permutation.py` (the record and its one reader), `transport/sort.py`, `deck_record.py`, `spectra/vibrational_analysis.py` | `tests/test_siesta_vibration_e2e.py` (the road's CO2, its carbon last in the user order → the result's oxygens atom indices 0 and 1, the carbon 2) |
| I8 | **One start state per kind on SIESTA**: the density read, the geometry declined out loud; the vibration warm-file section names `.FC`/`.FCC` inventory-only; the optical section names `.EPSIMG`, its product, a re-run of an optical call resuming from its own `.DM` | `siesta/vibration_deck.start_state_lines`, `siesta/warm-files.toml` | `tests/test_siesta_vibration_deck.py`, `tests/test_warmfiles.py` |
| I9 | **Every mean field the PySCF deck builds is dressed by the one generated door** (`_mb_configure_scf`, `_mb_configure_theory`), on either method; no SCF knob is spelled twice | `pyscf/scf_setup.py`; every `_build_mf_at` | `tests/test_vibration_render_gate.py::test_the_level_of_theory_has_one_spelling_per_deck`; the honesty gate |
| I10 | **Every parameter the form shows is honoured by the render, or refused by name** | `tests/test_vibration_form_honesty.py`; `validation/spectra.py` | the same test |
| I11 | **The kind's science gate fails closed**: an engine config the dispatch does not name is refused, never given an empty verdict | `validation/__init__._validate_vibration_kind` | `tests/test_vibration_render_gate.py` |
| I12 | **A cost claim is read from the code** (R8): the advisories say what the code skips, and what it cannot | `validation/spectra.py`; § 4.4, § 5.7 | review |
| I13 | **The Methods paragraph is composed once**, route-neutral before the run, the route added at load; the count it states is R2's, and the level of theory `is_dft`'s | `spectra/methods.py`, `_loaded` in `web/blueprints/spectra.py` | `tests/spectra/test_methods.py`; the IR-only and solvated runs assert the text; `test_a_hartree_fock_deck_names_no_functional` the level of theory |
| I14 | **Absent is never zero** in the file, on both engines; the equilibrium energy subgroup travels whole or not at all | `SpectraResults.__post_init__`, `spectra/vibrational_analysis.py` | `tests/spectra/test_types.py`, the SIESTA fixture test |
| I15 | **One structure**: the result states its structure through the one serialiser (`Structure.to_dict()`), each writer building it from what its deck already carries once (§ 6.10.1) — never a hash of a structure kept elsewhere | `spectra/pyscf_vibration.py`, `spectra/siesta_vibration.py` | review; the reader checks the structure against the result's own count, coordinates and held list (§ 6.7) |
| I16 | **The derived per-mode values are written by one function, beside what they come from, on both engines, and checked on read**; the element shares at load, never in the file (§ 6.6) | `spectra/derived.py`, `SpectraResults` (the check), `_loaded` | `tests/spectra/test_activity.py`, `test_motion_share.py`; the end-to-end runs read them from each engine's file |
| I17 | **Every catalogue citation resolves** in `science/references.bib`, and every key the prose cites is an entry | the catalogue-refs test | `tests/spectra/test_methods.py` |
| I18 | **No second producer**: a deck is written by `prep` from a description, through `spec_for`; there is no engine verb and the tab renders nothing | [`engines/pyscf.md`](?doc=engines/pyscf.md) § 1 | review |
| I19 | **A vibration's topic is the kind table's** (`spectrum/`), as every run topic is its kind's; nothing places or refuses a calculation by its folder | `kind_table.toml`, `projects.py` | review |
| I20 | **Nothing of molbuilder's is spliced into a deck**: what the PySCF script runs it imports from `mb_pyscf.pyz`, whose members load with molbuilder absent (since 2026-10-05; until then the rules were spliced as source, and a name the source read from its module -- the constant left behind on 2026-09-21 -- was undefined in the deck) | `runwrap.PYSCF_COMPANIONS` | `tests/test_pyscf_bundle_runs_alone.py` and the end-to-end runs |
| I21 | **The SIESTA keyword set is re-verified on any SIESTA upgrade** against the manual-derived table, never `strings` | § 5.1 | review, on every upgrade |
| I22 | **The job writes the result, on both engines**: a vibration's `launch` ends with `<label>.spectra.json` in the attempt — the PySCF deck writes it, and on SIESTA the finish the wrapper runs after the force-constant run (§ 5.5), whose failure is the job's. Nothing after `launch` derives a run's result: `summarize` only compares the results of a displacement sweep's stages (§ 5.9) | `runwrap` (the finish step), `DeckSpec.finish` → `Job.finish` | `tests/test_siesta_vibration_e2e.py` (the launch alone leaves the spectrum) |
| I23 | **What travels is one declared set that imports only what the job env carries**: the finish's modules are `runwrap.VIBRATION_COMPANIONS`, each its own file imported two ways; at load, and in every function the finish calls, they import only the standard library, numpy and netCDF4 — the SIESTA job envs' own packages (`envs/recipes.py`; netCDF4 for the result's data file, § 6.10.5–6, imported at load so a job whose env lacks it stops at the `loads` check, before the engine runs) — and each other. A member's functions the finish never calls (the host's element shares, the host's typed reader) may import what the host has | `runwrap.VIBRATION_COMPANIONS`, `envs/recipes.py` | the SIESTA end-to-end run, whose job python cannot import molbuilder |
| I24 | **The force constants are taken at the relaxed geometry**: the run's plan step writes the `relax` call's last geometry as the force-constant call's `coordinates.fdf`, in that call's cell (§ 5.11 § 3); a sweep's later stage taking the first stage's `relax` call over (§ 5.11 § 6) is not built yet (plan § 1ae) | `spectra/siesta_vibration.plan_call`, `jobset/submit._phased_walk` | `tests/test_siesta_vibration_e2e.py` |
| I25 | **A sweep's summary loses nothing and copies nothing**: every stage's files stay in its attempt and the record names them; it adds only the comparison — modes matched by shape, the frequency changes, the force-constant changes read from the raw `.FC` — and flags only against a tolerance the person gave (§ 5.9) | `spectra/displacement_sweep.py` | `tests/test_siesta_vibration_e2e.py` (the sweep's record) |

---

## 8. The pieces, and how the data flows

### 8.1 The file map

| file | role |
|---|---|
| `molbuilder/spectra/normal_modes.py` | `rigid_motions`, `vibrational_modes`, `vibrational_thermo`, `vibrational_thermo_grid`, `THERMO_GRID_K` — the one harmonic path and the thermo sums; travels in both bundles, `mb_pyscf.pyz` and `mb_vibration.pyz` |
| `molbuilder/spectra/pyscf_vibration.py` | `homo_index`, `mo_window`, `dipole_derivatives`, `as_numpy` — the PySCF route's rules, which the vibration script calls inside its run, imported from `mb_pyscf.pyz` |
| `molbuilder/spectra/results.py` | `SpectraResults`, `ModeData`, `ModeElectronicStructure`; the schema and its history; the reader's gates; the activity classes at serialisation; `motion_share_by_element` |
| `molbuilder/spectra/activity.py` | the active/inactive decision per channel (§ 6.6); travels in both bundles |
| `molbuilder/spectra/derived.py` | the derived per-mode values — the activity classes and the zero-point amplitude and displacement — computed once for both writers and the reader's check (§ 6.6); travels in both bundles |
| `molbuilder/spectra/selection.py` | the mode selectors and the window (§ 4.8) |
| `molbuilder/spectra/methods.py` | `render_methods_md`, `with_ir_route`, `extract_citation_keys` (§ 4.10); `siesta_methods_text`, the force-constant route's paragraph (§ 5.5), and its Raman sentence, `siesta_raman_sentence`; the views' sentences at load, `with_infrared_view` and `with_raman_view` (§ 5.10 ②, § 5.11 § 10) |
| `molbuilder/spectra/infrared.py` | the infrared of a set of atoms from the run's charges, or of a held PySCF molecule from its dipole change — `infrared_view`, `read_ask` (§ 5.10 ②) |
| `molbuilder/spectra/optical_data.py` | the optical calls' data file, `<label>.optical.nc` — its variable table, writer and reader, `read_optical_data` (§ 5.11 § 8) |
| `molbuilder/spectra/raman.py` | the Raman at a laser — `raman_view`, `raman_view_of`, `read_ask`; `<label>.raman.nc`'s variable table and `read_raman_data` (§ 5.11 § 10) |
| `molbuilder/spectra/export.py` | a vibration calculation's export — `export`, `result_path`, `name_of` (§ 5.11 § 10) |
| `molbuilder/siesta/vibration_phases.py` | the phases of a SIESTA vibration run — `PHASES`, `Call`, the index and each call's state (§ 5.11 § 2, § 7) |
| `molbuilder/postprocess.py` | a script's doors — `infrared_view`, `raman_view`, `read_spectra_data`, `read_optical_data`, `read_raman_data`, `reference_max_displacement` |
| `molbuilder/spectra/vibrational_analysis.py` | the analysis, engine-neutral: `vibrational_analysis` — a free-atom block, masses, geometry, held set and frame → `SpectraResults` (§ 5.5) |
| `molbuilder/spectra/siesta_vibration.py` | the SIESTA run's Python: `plan` and `record`, the steps inside each call's run script, the record step after the last call running `finish`; `read_force_constant_run`; `main`, the entry of `mb_vibration.pyz` (§ 5.5, § 5.11 § 3) |
| `molbuilder/parse/engines/siesta_epsimg_reader.py` · `parse/engines/siesta_epsimg.py` | an optical call's `.EPSIMG`: its one reader, travelling, and its registered parser (§ 5.11 § 7) |
| `molbuilder/jobset/submit.py` · `jobset/materialize.py` · `monitor.py` | the walk over the phases (`_walk_script`); the run and its calls opened, a warm launch's take-over (`open_sweep_run`, `HandOver`); each call's member monitor and the run's `notify` (§ 5.11 §§ 3, 6, 7a) |
| `molbuilder/spectra/displacement_sweep.py` | a SIESTA displacement sweep's summary: `match_modes` (by shape), `stages_share_a_directory` (the layout rule `prep` refuses by), `collect_sweep`, `sweep_path`, `write_sweep`, `sweep_table_text` — `<label>.fc-sweep.json` (§ 5.9); which stages take part is `pyscf/stages.force_constant_stages` |
| `molbuilder/parse/engines/siesta_fc.py` | `read_fc`, `hessian_from_fc` — the `.FC` reader (§ 5.4) |
| `molbuilder/sidecars/spectra.py` | `write_spectra_payload` (the one writer of the file: a payload built phase by phase, as the PySCF script does) and `dump_spectra_json` (a results object, through it), `finite_or_none`, `parse_spectra_json` |
| `molbuilder/pyscf/vibration_deck.py` | the PySCF deck composer, the `VibrationConfigView`, the relaxation / gradient / thermo / IR-only blocks, `vibration_stages` in `pyscf/stages.py` |
| `molbuilder/pyscf/vibration_emitters.py` | the emitted blocks: constants, the molecule, the equilibrium SCF, the Hessian, the Raman sweep, the IR projection, the electronic-structure loop, the Methods fragment |
| `molbuilder/pyscf/scf_setup.py` | the generated SCF and DFT dressers (I9) |
| `molbuilder/siesta/vibration_deck.py` · `siesta/input.py::spec_for` · `siesta/layout.py::FC_SECTION` · `siesta/warm-files.toml` | the SIESTA deck (§ 5.3) |
| `molbuilder/config/pyscf.py` · `config/siesta.py` · `data/catalogue.template.toml` | the fields and the catalogue rows of § 3 |
| `molbuilder/transport/sort.py` · `molbuilder/atom_permutation.py` | `sort_by`, `SORT_KEYS`, `apply_order` — `sort_by` sets the copy's `atom_order` and refuses a remapped input; the record itself — `Permutation`, the type of the structure's `atom_order`, read from a dict by its one reader (`Permutation.from_dict`), `map_to_user_index` and `map_to_engine_index` — in `atom_permutation`, which travels with the finish (§ 5.2; `model/overview.md` § 2.2a) |
| `molbuilder/deck_record.py` | the molbuilder blocks' grammar and their one JSON reader, with `extract_engine_offset`, `extract_vibration_record`, and `extract_atom_metadata` with `order_record_of`, the ATOM-METADATA block's order record (`model/overview.md` § 2.2a, I5) — below `script_emit`, which writes the blocks, so the finish can read the three beside the job (§ 5.3) |
| `molbuilder/validation/spectra.py` · `validation/__init__.py` | the two kinds' checks and the dispatch (§ 3.3) |
| `molbuilder/jobset/prep.py` · `jobset/_cli.py` · `jobset/materialize.py` | the sort at prep, the phases' decks and the finish onto the stage's job; `molbuilder describe`'s kind gate; the bundle brought into each attempt |
| `molbuilder/runwrap.py` | the run wrapper — its banner names the held atoms from the deck's one spelling, and it runs each call's plan and record steps, the finish among them; `VIBRATION_BUNDLE` / `VIBRATION_COMPANIONS`, what travels for the finish (§ 5.5); `PYSCF_BUNDLE` / `PYSCF_COMPANIONS`, what the PySCF script imports (`engines/pyscf.md` § 3); `bundles_for`, the one list of what travels beside a deck |
| `molbuilder/web/blueprints/spectra.py` · `build.py` | the tab page, `/api/spectra/load` and `_loaded`, `/api/spectra/export`; the schema, preflight and hand-over doors |
| `molbuilder/web/static/lib/inspectors/fc-sweep.js` | the sweep record's presenter on the Results tab (§ 5.9) |
| `molbuilder/web/static/spectra/viewer.js` · `lib/spectra/core.js` · `lib/inspectors/spectra.js` · `lib/spectrumchart/` · `lib/vibrationview/` · `lib/task-handover.js` · `task-setup/viewer.js` | the tab, the shared engine, the presenter, the chart, the animation, the hand-over, the rung tab that prints the commands |

### 8.2 The data flow

```mermaid
flowchart TB
  subgraph describe["describe (browser or CLI)"]
    ST["structure + frozen_atoms region<br/>(.xyz + .molstruct.json)"]
    CAT["catalogue narrowed to (engine, vibration)<br/>→ the form · → &lt;label&gt;.template.toml"]
    TJ["task.json: calculation vibration,<br/>engine, the ladder (one stage, freq)"]
  end
  subgraph prep["prep task"]
    G["validate(struct, cfg, calculation)<br/>PySCF checks #124; SIESTA checks #124; refuse"]
    SORT["SIESTA: sort a copy held-first,<br/>its deck states the order record"]
    D["spec_for(…, calculation='vibration') → prepare_deck<br/>PySCF: the script · SIESTA: the .fdf"]
  end
  subgraph run["launch task"]
    PY["PySCF script: relax → SCF → Hessian(free) →<br/>modes → strengths → thermo → ES → .spectra.json"]
    SR["SIESTA's run in phases, one job (§ 5.11):<br/>its relax call (box unticked) → its fc call,<br/>the relaxed geometry written by the plan step →<br/>SCF + 6·n_free displaced SCFs → .FC"]
  end
  SUM["the record step after the fc call (SIESTA):<br/>mb_vibration.pyz — read .FC → H_AA → vibrational_modes →<br/>invert the permutation → .spectra.json"]
  LOAD["POST /api/spectra/load → SpectraResults.from_dict<br/>+ with_ir_route + motion_share_by_element<br/>+ the views asked: infrared_view, raman_view_of"]
  VIEW["Results tab: chart · table · animation ·<br/>electronic structure · thermochemistry"]
  ST --> G; CAT --> G; TJ --> G; G --> SORT --> D; G --> D
  D --> PY; D --> SR
  SR --> SUM; PY --> LOAD; SUM --> LOAD; LOAD --> VIEW
```

*(The sort writes no file: the copy carries its order record, each deck states it, and the finish reads it from its deck — `model/overview.md` § 2.2a.)*

### 8.3 The doors

| door | does |
|---|---|
| `GET /api/build/schema/common?calculation=vibration&surface=state` · `GET /api/build/schema/common?calculation=vibration` · `GET /api/build/schema/<engine>?calculation=vibration&surface=own` | the form schema from the catalogue narrowed to the kind: the charge and spin both engines share, their other shared settings, and each engine's own |
| `POST /api/build/preflight` | the live checks: `validate(struct, cfg, calculation="vibration")` |
| `POST /api/structure/analyze` | the charge and spin each form's vibration will carry — the one electronic state, for exactly what the form says, about the structure the viewer holds; the chemistry card shows it and fills nothing in |
| `POST /api/task-setup/handover` | render `<label>.template.toml` and `task.1st.json` for the kind; the browser writes them where the person chose |
| `POST /api/task-setup/save` · `/prep` | write `task.json` and `job-request.json`; run `prep` for the groups picked, each one job, on a named machine — the function the terminal's `prep task` runs |
| `POST /api/spectra/load` | parse an existing `.spectra.json` into display data, with the infrared and Raman views asked (§ 5.10 ②, § 5.11 § 10); typed errors |
| `GET /api/spectra/export` · `molbuilder jobset export` | a vibration calculation's data files and its Raman view at a laser, one zip or a folder (§ 5.11 § 10) |
| `molbuilder.postprocess` | a script's readers of the result, its data files and the views (§ 5.11 § 10) |
| `molbuilder describe / prep / launch` | the same road from a terminal (§ 2.1); the run ends with the result |
| `molbuilder jobset summarize task [--tolerance-cm1 X]` | a SIESTA displacement sweep's comparison, `<label>.fc-sweep.json` (§ 5.9) |

---

## 9. Validation status, and the tests

**Validated, with numbers:**

- the rank rule against PySCF's own analysis on free molecules (§ 4.5);
- water with its oxygen held through the whole road: three modes, three motions
  removed, every mode orthogonal to every removed pattern in the mass metric;
- the free-atom Hessian against compute-everything-and-slice, Hartree–Fock
  and DFT, with and without dispersion (§ 4.4); the wavenumber constant
  against PySCF's derivation to 1·10⁻⁹, $\sum m\lvert L\rvert^2$ at 1.00000000 on every
  path, the $d\mu/dR$ tensor layout (8.4·10⁻⁵ as emitted against 5.2·10⁻¹
  transposed), the two analytic-route corrections at 6.7·10⁻¹⁶ (§ 4.5, § 4.6);
  both rendered decks parse, import their bundle and run to exit 0 (I20);
- BDT free against BDT with both sulfurs held at one geometry (§ 11;
  [`science/normal-modes.md`](?doc=science/normal-modes.md) § 9);
- infrared at band level: water at B3LYP/def2-SVP inside the literature windows
  with bend > asymmetric > symmetric; CO₂ reproducing mutual exclusion (the
  numbers are § 11);
- the `.FC` units and the `.FCC` shape on H₂ (§ 5.4); CO₂ with its carbon
  held, listed last, on SIESTA through jobset — the held-first deck a 3-cycle
  of the input, the result back in the user order (§ 5.5).
- SIESTA's infrared from the charges (§ 5.10 ②, 2.36.0) on that CO₂: a
  field along z sees neither bend, along x neither stretch, each stretch's
  dipole change opposes the oxygens' net motion and the molecule moves no
  charge; the stretch whose oxygens move apart reads 0.5 % of the other at
  `MeshCutoff` 150 Ry and $6.5\times10^{-8}$ (Hirshfeld) at 500 Ry — the mesh
  tells the two oxygens apart, the relaxation tolerance does not
  ([`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md) § 7).
- **the convergence condition and the step, measured 2026-09-24** on H₂ with
  one atom held (GGA/DZP, the road end to end: an optimization calculation,
  its final frame exported from the Results tab, the vibration on the pair):

  | input geometry | largest force component on the **free** atom at the reference step (eV/Å) | $\delta$ (Å) | $\omega$ (cm⁻¹) |
  |---|---|---|---|
  | the experimental 0.741 Å, unrelaxed | 1.274 (1.274 on the held atom) | 0.0212 | 3358.0 |
  | relaxed to `MD.MaxForceTol` 0.02 eV/Å (0.7744 Å) | 0.0068 (0.011 on the held atom) | 0.0212 | 3024.4 |
  | relaxed to 0.001 eV/Å (0.7745 Å) | 0.00006 (0.004 on the held atom) | 0.0106 | 3012.1 |
  | the same | 0.00006 | 0.0212 (the default 0.04 Bohr) | 3022.3 |
  | the same | 0.00006 | 0.0423 | 3044.7 |

  The missing relaxation costs a tenth of the frequency; the tolerance, two
  wavenumbers (the held atom keeps its constraint force, which is not judged —
  R5). The one-sided constants differ by 2.4, 4.7 and 9.0 eV/Å², linearly in
  $\delta$: the bond's cubic term, which the central difference cancels (every odd
  order; its leading error is $O(\delta^2)$). The frequency drift — 10 cm⁻¹ between
  the default and its half, 22 more at its double — is **not** that $O(\delta^2)$
  signature: a Morse estimate of the bond's quartic term gives a few
  wavenumbers with the wrong power of $\delta$, so most of the drift is numerical, and
  the real-space grid (0.09 Å spacing against 0.01–0.04 Å nudges) is the usual
  suspect, not isolated here. So the plateau $\omega(\delta) \approx \omega(\delta/2)$ of
  [`science/normal-modes.md`](?doc=science/normal-modes.md) § 4b.6 C is not
  reached at the default, and a δ-only ladder cannot say why; the sweep takes
  a mesh rung beside the δ rung (V1.23, § 5.9), and the measured δ+mesh
  comparison on H₂ is V1.38, not yet run. The block's asymmetry is 10⁻¹² eV/Å² throughout:
  on this axial block the off-diagonals vanish by symmetry, so it shows nothing
  here.

**Not done:** a mode-by-mode cross-check of intensities against an external
code (Gaussian, ORCA, Turbomole) — absolute intensities carry that caveat;
the reduced Hessian with a GPU mean field; and four of the design's five held systems
through the whole road — acetylene with both carbons held (the collinear
trap), NH₃ with its three hydrogens held (nothing removed), an empty held
list reproducing the free path *exactly*, and the water dimer with one
molecule held (the over-removal guard) — which exist today as rank rows
only (V1.17); the δ-convergence comparison run with a mesh rung beside it
(V1.38 — the sweep takes one, § 5.9); mode matching across runs (V1.24).

**The tests**, by what each proves:

| file | proves |
|---|---|
| `tests/spectra/test_normal_modes.py` | every row of the science acceptance table — the rank rule alone, no engine |
| `tests/test_vibration_e2e.py` | the rank gate against PySCF; the water loop (relaxation, three modes, thermo, the viewer loads it; the file states every atom's mass, `chemistry.atomic_mass`'s, and every mode's derived values, which its reader checks — red with the writer's derivation removed, and the deck refusing to run on masses other than PySCF's own); IR alone in water's windows with the route recorded; the solvated chain; frequencies unmoved by asking for IR; water with O held; the free-atom block check; the probe on the modes an explicit list names and no other, after a list the reader cannot take is refused at prep; at prep alone, a setting that enters nothing said to (a pressure with atoms held, a window outside `all`), and a Hartree–Fock deck naming no functional, no dispersion and no grid, with the warnings that say so and none for `dispersion = "none"` |
| `tests/test_siesta_vibration_deck.py` · `tests/test_siesta_relax_run_e2e.py` · `tests/test_siesta_vibration_e2e.py` | the force-constant deck prep writes for the `fc` phase on the road's CO₂ (O O C, the carbon held — the held-first deck a 3-cycle of the input), read as the engine reads it: the free atoms the trailing range, nothing relaxing, the start state the kind's, both charge partitions at every step; the record table of § 2.2 on a relaxation run on the road (a matching record's info line, a looser record's warning, another geometry, another level of theory or engine, the unticked offer to skip, no record accepted with a hint); the whole SIESTA road through jobset on the road's CO₂ — one run in phases, its calls, index, result and notifications (rewritten for § 5.11 in plan § 1ae M2, 2.42.0) |
| `tests/test_siesta_vibration_e2e.py::test_the_results_tab_shows_a_siesta_vibration_as_its_modes` | the ticked SIESTA road on CO2 and then the Results tab, read against `web/spectra.md`: the mode positions under their heading — one line of one height, no curve, no height numbers — with no width or floor control and the sentence saying which case it is, the infrared controls with the IR and charge-moved columns of a result stating its charges and the door's answer once the atoms are named (`web/spectra.md` § 3.2), no Raman or orbital column, no Raman controls, and the summary saying *not requested* for the Raman — the box unticked ran no optical calls — and *not computed on this route* for the probe, no Raman or probe dot, the electronic-structure tab naming the planned projected density of states, the thermochemistry labelled vibrational-only with no pressure and its bars summing to $F_{\mathrm{vib}}$ — every hiding read from the computed style (red against the width control shown whatever the strengths, the `.spectra-phase[hidden]` guard removed, the notice's route branch removed, the RRHO labels forced) |
| `tests/spectra/test_raman.py` · `tests/test_siesta_vibration_e2e.py::test_the_raman_at_a_laser_holds_co2s_symmetry_and_exports_whole` · `::test_the_page_asks_the_raman_at_a_laser_and_exports_it` | the Raman at a laser (§ 5.11 § 10): Kramers–Kronig against a Lorentz oscillator's closed form and beside a narrow line, a run widened after it against the wider spread; on CO₂ at 633 nm, light along z, the stretch whose oxygens move together reads 0.5 % of the one whose oxygens move apart — 0.23 against 45.1 Å⁴/amu; the export, NetCDF and CSV, read back through `postprocess` and `xarray` equal to the load door's view, each source named by its sha256, nothing written into the calculation; the page asking the laser, its columns and summary, and its Export |
| `tests/test_engine_offset_reaches_every_deck.py` | a vibration's force-constant deck keeping the engine's frame when its input is stated relaxed, placed by the rule otherwise — read from the `fc` phase's deck as the engine reads it; the input's relaxation record left behind while its calculation record still sets the electronic state (V1.35) |
| `tests/test_vibration_e2e.py` · `tests/test_displacement_sweep.py` | matching by shape on free water made on the road (atoms of unequal mass): a rank swap found, a mixed pair's overlap the mass-weighted cosine; a PySCF vibration refused by `summarize task` on the road |
| `tests/test_vibration_render_gate.py` | the deck runs the science gate; an unknown engine class is refused |
| `tests/test_vibration_form_honesty.py` | every offered parameter changes the deck |
| `tests/spectra/test_types.py` · `test_parsers_json.py` · `test_atom_index_contract.py` | the artifact's gates and round trip; the free-atom invariant |
| `tests/spectra/test_activity.py` · `test_motion_share.py` · `test_selection.py` · `test_methods.py` · `test_config.py` · `test_blueprint.py` | the derived classes; the element shares; the selector's four rules -- the one the script imports; the prose, its citations and the provenance rows of § 6.4; the defaults; the page and the load door (the old generator's `test_engine.py` / `test_script.py` died at P3) |
| `tests/spectra/test_spectrumchart_*.py` · `tests/test_vibrationview_*_js.py` · `test_spectra_transition_js.py` · `test_spectra_phase_indicator_js.py` · `test_spectra_no_spectrum_sentence_js.py` · `test_task_setup_tab.py` · `test_spectrum_form_e2e.py` | the chart's maths, seal and box — the mode positions among them, several modes with an imaginary one, and the switch to heights when strengths land; the animation's maths and mount; the viewer's state; the phase indicator; the sentence where no strength is drawn; the send flow, and `dispersion = "none"` reaching the deck as no correction; the form's locks, at load and on each change, and every finding its live check returns drawn on the page |
| `tests/test_warmfiles.py` · `tests/validation/test_siesta.py` | the vibration warm-file section; the keyword table the deck is checked against |
| fixtures: `tests/fixtures/psml/` -- an input, the H, C and O pseudopotentials a live run needs; no run is saved (`process/testing.md` § 6, 2026-10-06) | |

---

## 10. Shipped and owed

Every open row below is registered under **V1** in
[`plans/plan.md`](?doc=plans/plan.md); this table says what stands, the plan
says what is next. *Working tree* would mean edited but not yet verified or committed;
nothing is in that state as of 2026-09-28.

| | status | note |
|---|---|---|
| the remapped state — the index carried by the structure and each deck, restored at every door that crosses back, `atom-permutation.json` retired (`model/overview.md` § 2.2a) | **built 2026-10-14**, 2.10.0 (V1.41 with R3; the finish reads its deck's order record with R5) — plan RMP, § 5ab (R4 dropped: within one calculation every deck is in one order) |  |
| the one harmonic path, the rank rule, its gate | **built 2026-09-23** | § 4.5 |
| one mass convention | **built 2026-09-21** | the 1823× defect |
| stationarity on the free atoms | **built 2026-09-23** | § 4.3 |
| the runs write the structure pair | **built 2026-09-22** | § 4.2 |
| the Hessian over the free atoms, the two corrections, the scope in the file | **built 2026-09-23** | § 4.4 |
| the SIESTA arm: deck, sorted copy, record, `.FC` reader, read-back (in the job since 2026-09-28, § 5.5), warm-file section, start state | **built 2026-09-23 / 24** | § 5 |
| the equilibrium block optional; the display form derived | **built 2026-09-23** | schema 6 |
| the wrapper names the held atoms | **built 2026-09-23** | it read a comment no deck wrote, then counted the `0` of `0-based` |
| the thermochemistry's self-defence filter: the `> 0` line deleted, the imaginary exclusion stated as `n_imag_excluded` | **built 2026-09-23** | § 4.7 — the free energy no longer depends on the sign of noise |
| the three counting sites read the one derivation; `_is_linear` and `_mode_count`'s `n_free < 2` arm deleted | **built 2026-09-23** | § 4.10, I1 |
| the pre-run notice and the tab's prose say the relaxation holds the set *when it runs*, not unconditionally | **built 2026-09-24** | § 4.2; under `already_relaxed` there is no relaxation and no constraints file |
| the emitted deck's comment on the canonical normalisation says amu, not "atomic units" | **built 2026-09-24** | the phrase the 1823× confusion lived in |
| the geometry in the file is the Hessian's | **built 2026-09-24** | § 4.2; the water loop asserts the relaxed geometry differs from the input |
| the thermochemistry headline and grid as one quantity, the headline T on the grid, no $kT$ in the held regime | **built 2026-09-24** | § 4.7; the free and the held water runs assert the headline equals its grid row |
| `raman_route`, `raman_fd_step_ang`; the Methods text states the Raman method one way | **built 2026-09-24** | § 4.6; the infrared-only and the solvated runs assert both |
| the reader's unknown-key gate; the partition check without a range the size of a lie | **built 2026-09-24** | § 6.7; two forward-compatibility tests that asserted the old rule are retired into the gate test |
| the Spectrum tab offers both engines — one strip over two catalogue-built forms, the structure choosing the default, the checks and the hand-over speaking the strip's engine; the hand-over gate admits SIESTA | **built 2026-09-24** | § 2.1, § 3.1; `web/spectra.md` § 5 |
| Task setup prints `--target this` for this machine whenever the CLI would refuse to guess; a SIESTA vibration's commands end with the launch, whose job writes the spectrum (the `summarize run <stage>` line it printed from 2026-09-24 went with the host read-back) | **built 2026-09-24; the last line retired 2026-09-28** | `web/task-setup.md` § 11 |
| the SIESTA job finishes its own calculation: the wrapper runs the finish after the force-constant run, the modules travel as `mb_vibration.pyz`, the deck carries its `vibration` block, and `summarize` derives nothing for the kind (plan W41) | **built 2026-09-28** | § 5.5, I22, I23 |
| the Results viewer draws a SIESTA file: a `null` energy as a dash, the Raman and infrared lines by route, the fingerprint on the two routes, the relaxation line carrying a disabled phase's warning, the thermochemistry headline naming its regime | **built 2026-09-24** | § 6.5 |
| the force keys say their unit: `max_force_eh_bohr`, `max_force_all_atoms_eh_bohr`, and the viewer prints Eh/Bohr | **built 2026-09-24** | § 4.3 |
| `removed_motions` and `hessian_scope` shown beside the result | **built 2026-09-24** | R7's second half |
| the phase flags' *not requested* state: a phase the description never asked for is `not requested` from the first write, on both writers; the viewer counts it finished and draws it | **built 2026-09-24** — V1.5 | § 4.9 |
| `top_n` and `threshold` retired from the config, the catalogue, `selection.py`, `methods.py`, `validation/spectra.py`, the emitter's ranking and the tab's lock map | **built 2026-09-28** — V1.6, decided 2026-09-23 | § 4.8 |
| `temperature_K` reachable on SIESTA; a vibrational-only result records no pressure (`pressure_atm` `null`), on both engines | **built 2026-09-28** — V1.7 | § 3.1, § 4.7, § 5.5 |
| a setting that enters nothing is not left silent: `pressure_atm` set away from its default with atoms held is warned about; the frequency window is locked outside `all` on the form and warned about at prep, where a locked field's value and a hand edit still arrive | **built 2026-09-28** — the user's yes to the recommendation; the prep warning the M2b⁗ review's | § 3.1, § 4.8 |
| `phase_ir`, the infrared intensities' own flag, on both writers; the viewer waits for it and says *still being computed* while it runs | **built 2026-09-28** — the review of M2b‴ | § 4.9, `web/spectra.md` § 2, § 7 |
| the thermochemistry near 0 K: `vibrational_thermo` returns the ZPE at $T \le 0$; `temperature_K` starts at 1 K | **built 2026-09-28** — the review of M2b‴ | § 4.7 |
| sisl in every SIESTA env (`envs/recipes.py`), installed where missing by `molbuilder envs repair <env>` | **built 2026-09-28** — W42 D5, M2b‴ | `ops/installation.md` |
| the result says what its route computed: the mode positions where no strength was computed — one line of one height at each mode, picked like a stick, under the heading *Mode positions* (user, 2026-09-28); the columns, the summary, the phase dots and the electronic-structure tab by role; the thermochemistry labelled by regime, its electronic reference the file's own | **built 2026-09-28** | [`web/spectra.md`](?doc=web/spectra.md) § 2, § 3, § 9b.3 |
| the Methods paragraph reads the effective level of theory — one answer, `PySCFConfig.is_dft`, for every reader — and a functional changed under Hartree–Fock is warned about; the dispersion correction applies on either method (HF-D3(BJ)), named with its own papers | **built 2026-09-28** — V1.8; the dispersion half the user's ruling that day (*"scientifically correct decision applied"*) | § 4.10 |
| the explicit mode list read by one reader, `PySCFConfig.explicit_modes` (the atom index list's grammar, `selection.parse_index_list`), for the deck, the Methods count, the reference selector and prep's refusal — the deck wrote the text's characters until 2026-09-28 and Phase 4 stopped on `int(',')` | **built 2026-09-28** — the M2b⁗ review | § 3.1, § 4.8 |
| `dispersion = "none"` stored as itself: the forms turned it into `None`, which a template writes valueless and `prep` fills with D3BJ | **built 2026-09-28** — the M2b⁗ review | [`engines/pyscf.md`](?doc=engines/pyscf.md) § 7a |
| the Methods rules owned by the spectra inspector's sheet (`lib/inspectors/spectra.css`), where the panel is mounted, in the token vocabulary — the text wraps on the Results tab | **built 2026-09-28** | [`web/spectra.md`](?doc=web/spectra.md) § 10 |
| the structure identity: two hashes (geometry, broad), minted at the three gates and carried, the job name out of it | **owed** — ruled 2026-09-22 | § 4.3; the run-written pair already hashes its bytes |
| the pair writer renders both halves before it writes either (`pair()` returns text) | **owed** — ruled 2026-09-23; **the deck's half built 2026-10-05**: the deck writes its pairs with the codec itself, imported from `mb_pyscf.pyz` (`engines/pyscf.md` § 3), so its own serialiser — the third — is gone | `plans/plan.md` V1.10; what remains is the half-written pair: a NaN in `info` raises at the sidecar write after the `.xyz` is on disk |
| the reduced Hessian with a GPU mean field | **untested** | § 4.4 |
| a structure sorted for two reasons | **never** — `sort_by` refuses a remapped structure, so two orders are never composed (`model/overview.md` § 2.2a, I4) | § 5.2 |
| `transport/compose.py` writes and reads its record, the junction carrying its order record in its sidecar | **built** — `sort_by(relaxed, "transport")`; the record read back from the junction's own sidecar (`load_compose_record`, 2.10.0) | § 5.2, I7 |
| the transport connection's level two (the coupling from `FC.Save.dHS`); Born-charge infrared on SIESTA | **not in scope** — recorded so the design does not foreclose them; level one is V1.25 below | § 5.6 |
| an external mode-by-mode intensity cross-check | **not done** | § 9 |
| four held systems through the whole road (acetylene, NH₃, the empty held list, the water dimer) | **owed** — V1.17 | § 9 |
| `_mode_count`'s results arm: wire it into the load path or delete it | **deleted 2026-09-28** — V1.18: nothing called it, and the count before the run is the run's by construction | § 4.10 |
| a release note: every held-atom spectrum and free energy computed before 2026-09-23 contains a non-vibration | **dropped 2026-09-28** — V1.16, by the user's ruling: obsolete runs are removed and run again | true and intended; the old runs disagree with the new ones |
| the presenter's category label names no engine (*Vibrational spectrum*); the Spectrum tab's engine sentence is the strip's | **built 2026-09-24** | |
| the Molbuilder tab's save prompt doubling a typed suffix (`x.xyz.xyz`); the `#`-label unconsumed warning (needs a ruling) | **owed** — UI walk 2026-09-23 | not this kind's, recorded where found |
| the vacuum notice on a gas-phase PySCF run | **built 2026-10-01** — plan § 5w K8: the box's advice is for an engine that computes in a cell ([`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) § 2.1) | |
| the SIESTA route judges stationarity: the forces at FC step 0 read into `relaxation.max_force_eh_bohr` over the free atoms, `converged` against the description's own `relax_force_tol`, the warning printed and shown | **built 2026-09-24** — V1.21 | § 5.5; R5 on both routes |
| the asymmetry diagnostic $\max\lvert H_{ij} - H_{ji}\rvert$ recorded as `engine_metadata.fc_asymmetry_max_ev_ang2` | **built 2026-09-24** — V1.22; no warning threshold yet; on an axial block the off-diagonals vanish by symmetry, so H₂ shows nothing | § 5.5 |
| `already_relaxed` on both engines as the person's explicit say: unticked the tool relaxes first (PySCF Phase 0; SIESTA the `relax` phase of its one run, the relaxed geometry written as the force-constant call's coordinates), ticked it measures and warns plainly; the relaxation's convergence items on the vibration form with the kind's recommended tight values | **built 2026-09-24** | § 2.2, § 3.1, § 5.2a, § 5.8 |
| the mass-calibrated displacement per mode — `zero_point_amplitude_amu12_ang`, `zero_point_displacement_ang`, derived at serialisation | **built 2026-09-24** — V1.29 | § 6.3, § 6.6 |
| the result states its masses and its derived values on both engines: every atom's mass stated at prep by `chemistry.atomic_mass` in the deck (the PySCF deck checking it against PySCF's own), weighted by and written as `equilibrium.masses_amu`; the activity classes and the zero-point values computed by `spectra/derived.py`, stated by the one writer at every write, checked on read; schema 7; the finish needs no ASE | **built 2026-10-10** — plan § 5z Q17-e2 step 1 | § 6.1, § 6.2, § 6.6, I3, I16 |
| one stationarity rule for both routes: the largest absolute force component over the free atoms against the template's own force tolerance, a plain warning above it, on PySCF and SIESTA alike | **built 2026-09-24** — V1.30, by the ruling of § 2.2 | § 2.2, § 4.3, § 5.5 |
| the structure carries its relaxation record — `info.relaxation` (engine, the run's tolerance, the largest remaining force, the held set, the geometry's fingerprint) beside `info.calculation` (the level of theory) — recorded by the Results tab's structure inspector from the run directory, read by both kinds' gates on the box's card: absent and ticked accepted with a hint, present checked against this calculation's tolerance, level and held set | **built 2026-09-24** — V1.28 | § 2.2; `model/parse.md` § 5b.1 |
| ~~the PySCF deck's own `_optimized.xyz` pair records `info.relaxation`~~ | **closed 2026-09-29** by the one-source rule (V1.31): the record is the one reader's (`parse.contract.relaxation_of`, over the run's own output), which the Results tab's export writes; a copy computed in the deck would be a second writer of it | § 2.2 |
| a δ-convergence report: two stages at $\delta$ and $\delta/2$ and a printed comparison of $\omega_\nu$ and $e_\nu$ | **built 2026-09-28** — V1.23: `summarize task` writes `<label>.fc-sweep.json` (modes matched by shape, the force constants compared from the raw `.FC`), the Results tab presents it; each stage relaxes in its own run, and two stages at different geometries the summary refuses — the take-over of the first stage's `relax` call is not built (§ 5.9) | § 5.9 |
| mode matching across runs by eigenvector overlap in the shared free subspace (Models A/B/C; PySCF against SIESTA) | **owed, needs a design** — V1.24 | `science/normal-modes.md` § 4b.6 F |
| a mode's frame set — one mode as one multi-frame pair a transport calculation cites; the density-difference maps $\Delta\rho_\nu(r)$; the projected density of states along a mode | V1.25 **built 2.35.0** (D6, D7: the frame set of § 5.10 ③, `mode_frames.frameset_from_mode`, `molbuilder frameset from-mode` and the spectrum viewer's *Write frame set…*); V1.26 **needs a decision** (M10) | § 5.6, § 5.10 |
| the PySCF probe: five points, the coupling per zero-point amplitude $g_\nu$, a molecule-projected window for a cluster | **decided, not built** — V1.27 (D8: the same nodes as the frame set) | § 4.8, § 5.10 |
| the `fc` phase's deck leaves the input's relaxation record behind; its calculation record stays | **built 2026-09-29** — V1.35 | § 5.2a |
| what a SIESTA vibration says about its geometry after its `relax` phase: that call's outcome and its remedy, judged at the finish, not the describe-time box and record advice | **built 2026-09-29** — V1.36, the user's word the same day | § 5.2a, § 5.8 |
| the response screen — which mode to take to transport: each mode's character, the charge it moves, the frame set at the person's spread, an `electronic` kind, the per-mode response record and its ranking table | the frame set **built 2.35.0** (③), the charges' infrared and the charge a mode moves **built 2.36.0** (②); the rest **decided 2026-09-28, not built** (D1–D5, D7, D8; D6 as recommended; E3 owed) — V1.34, W42 | § 5.10; `science/normal-modes.md` § 4c |
| one SIESTA vibration run in phases — relaxation, force constants, Raman's optical single points, each SIESTA call a point of the run | **built** — the run 2.42.0, its records 2.43.0, the Raman at a laser and the export 2.44.0; owed: a sweep's take-over of the first stage's `relax` call and a transport sweep's one start and finish (§ 5.11 heading) | § 5.11; `science/vibrational-optical-response.md` § 3 |

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
20.1 cal/mol/K of entropy, −6.0 kcal/mol in $-TS$.
After: `removed_motions.count = 3`, three modes (bend, symmetric and
asymmetric stretch), every one orthogonal to every removed pattern. It runs
in a second and half its raw output is not a vibration, which is why it is the
test and the teaching case.

**BDT free and with both sulfurs held — the real case.** One RHF/STO-3G
geometry, 14 atoms. Free: $3\cdot14 - 6 = 36$ modes. Held: one surviving turn
about the S···S line (the sulfurs sit *on* it), $3\cdot12 - 1 = 35$ modes. The
ring C–H stretches agree to 0.001 cm⁻¹ between the two runs; the S–H
stretches fall by $\sqrt{\mu_{\mathrm{free}}/\mu_{\mathrm{held}}} = 0.984641$ (measured 0.984673 and
0.984664); the 36th number of the old two-branch code was the turn, at
−0.93 cm⁻¹ at the minimum and **96.78 cm⁻¹ off it**, in the middle of the real
vibrations, which is why it is removed before the solve and never spotted
after ([`science/normal-modes.md`](?doc=science/normal-modes.md) §§ 1, 4.1, 9).

**CO₂ on SIESTA — the road.** § 1.4 and § 5.5: listed O O C with the carbon
held, the held-first deck a 3-cycle of the input, four modes — a degenerate
bend pair in the (x, y) plane, two stretches along z — two motions removed,
every atom reported by its own index in the user order.

**CO₂ — mutual exclusion.** A centrosymmetric molecule's infrared-active
modes are Raman-silent and vice versa. The run: 653.45 cm⁻¹ (×2) infrared
32.85 km/mol and Raman-silent; 1388.81 Raman 14.74 Å⁴/amu and infrared-silent;
2460.11 infrared 613.04 km/mol. The silent entries came back as 10⁻⁹ to 10⁻⁸
residue, which is what § 6.6's classifier exists for. With both oxygens held
the free carbon sits *on* the O···O line, the surviving turn moves nothing,
and $n_{\mathrm{rigid}} = 0$ — the trap a table of cases gets wrong.

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
$G = E_{\mathrm{elec}} + \mathrm{ZPE} + H_{\mathrm{thermal}}(\text{trans} + \text{rot} + \text{vib}) - T\,S(\text{trans} + \text{rot} + \text{vib})$, from
PySCF's `thermo.thermo`, and the same at every point of the curve. Water with
its oxygen held, `regime = vibrational-only`: $G = E_{\mathrm{elec}} + \mathrm{ZPE} + U_{\mathrm{vib}} - T\,S_{\mathrm{vib}}$
— no translation, no rotation, no $pV$; the note in the file says so, and the
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
| **phase** | one of the tracked steps of a run, each with its flag in the file (§ 4.9) — on PySCF the script's, on SIESTA the run's calls: `relax`, `fc`, `optical` (§ 5.11) |
| **the sorted copy** · **the order record** | the structure SIESTA is given, and the `atom_order` its deck states that undoes it |
| **artifact** | `<label>.spectra.json` |
| **route** | which way a strength was obtained (`ir_route`, `raman_route`); and, in the titles of §§ 4–5, an engine's whole way to the block |
| **rung** · **ladder** · **tier** | one stage of a description, the list of them, and the coarse/medium/tight grades an optimisation's stages take; a vibration's ladder is one rung on both engines — on SIESTA one run in phases (§ 5.11) — or several in a displacement sweep (§ 5.9) |
| **RRHO** | rigid-rotor harmonic-oscillator: the gas-phase thermochemistry of a free molecule — electronic, translational, rotational and vibrational sums |
| **`axis_kind`** | the structure's own statement, per axis, of `isolated`, `periodic` or `transport`; with the cell it decides which turns cost nothing |
| **Γ** | the centre of the Brillouin zone, $q = 0$: every repeated cell moving in phase — the only point this calculation computes at |
| **the spectra migration** | the plan of 2026-08-20 that made a spectrum an ordinary described job, in phases P0–P3 ([`archive/2026-08-20-spectra-migration-plan.md`](?doc=archive/2026-08-20-spectra-migration-plan.md)) |

---

## 13. References

Cited by key; entries and their verification records are in
[`science/references.bib`](?doc=science/references.bib), the science
document's § 11 annotates the ones the rules rest on. Used here:
[Head1997], [LiJensen2002], [Besley2008], [Ghysels2008], [Vester2024],
[Tao2021], [QChemPHVA], [ASE2017], [Ghysels2010], [Wilson1955],
[Komornicki1979] (dipole and polarizability derivatives), [Greenler1966]
(the surface selection rule, § 4.6), [Grimme2010]
(the D3 dispersion correction), [Grimme2011] (its Becke–Johnson damping),
[Caldeweyher2019] (D4), [Sun2018], [Sun2020] (PySCF),
[Galperin2007], [Frederiksen2007] (vibrations in molecular junctions and
inelastic transport, the transport connection of § 5.6), [Gillet2013] (the
Raman tensor at a laser, complex, § 5.11 § 10).
