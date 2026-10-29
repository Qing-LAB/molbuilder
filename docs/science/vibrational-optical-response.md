# The optical activity of a SIESTA vibration — infrared and Raman

**Role:** science reference — a method; the charges are written since
2.35.2, the infrared and the displacement reference built in 2.36.0, Raman's
optical calls in 2.42.0–2.43.0 and its strength at a laser in 2.44.0
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11)
**Domain:** science
**Companions:** [`engines/vibration.md`](?doc=engines/vibration.md) § 5.3 (the
force-constant input), § 5.10 ② (the charge each mode moves), ③ (a mode's
frames) and § 5.11 (one run in phases — Raman its optical phase); [`vibrational-averaging.md`](?doc=science/vibrational-averaging.md)
§ 5 and § 2a (the displacement); [`normal-modes.md`](?doc=science/normal-modes.md)
§ 4a.6, § 4b.7 (the infrared of the free atoms, and Raman on SIESTA at
a metal surface); `plans/plan.md` § 1ac.

*Written 2026-10-24 at the user's request: "we need a new method document for
this, science, procedure design, and quantitative explanation, all formula
should be latex and checked, with diagram of workflow and explanation of
dependency". Its scope, the user's: "it is simple: siesta calculate correctly
optical activity and evalaute the displacement"; "all i asked is a vibration
calculation, assumed an EM in the vacuum"; "here we focus for a unit EM
intensity with a specific direction how the molecule respond".*

---

## 0. In short — the question, and the workflow

A SIESTA [Soler2002, Garcia2020] vibration gives a junction's modes: the
molecule and the metal atoms left free beside it move, the electrodes are
held.

```text
   held electrode                 the molecule M                 held electrode
  ● ● ● ● ● ● ○ ─ S ─────────── C6H4 ─────────── S ─ ○ ● ● ● ● ● ●
              └──────────── the free set F: every atom that moves ───┘
                     ○ a free metal atom      ● a held one

        light in vacuum, its field along a direction ê the person chooses
        ──────────────────────────────────────────────►  (z, the gap's axis, for one)
```

**Which modes are infrared- or Raman-active, and how strong is each relative
to the others**, for light polarized along $\hat e$? And **how far should the
frames along a mode reach** — the motion they average over?

Everything comes from SIESTA's own electronic structure at the mode's
geometries. **No field is applied in any calculation.** In the infrared, the
dipole is the free atoms' — the system the person described by what they
freed — and the held atoms are not in it; for Raman, the polarizability is
the whole cell's (§ 3.1, § 3.4).

| | a mode shows in the spectrum when it changes … | read from |
|---|---|---|
| **infrared** | the free atoms' **dipole** — how their charge sits | the charges SIESTA prints at every step of the vibration itself |
| **Raman** | the **polarizability** — how easily light shifts the electrons | SIESTA's optical calculation at the mode's displaced geometries |

### 0.1 What the person gives

| input | for | default |
|---|---|---|
| the relaxed junction — its structure, labels and held atoms | everything | — |
| the free atoms — every atom not held, the person's choice when describing the vibration | the modes and the infrared dipole | the structure's held atoms (`frozen_atoms`) |
| the light's direction $\hat e$ | infrared and Raman | z on the spectrum viewer, which takes x, y, z or any vector *(the user, 2026-10-24)*; a script names it. None for a free molecule — nothing held — whose infrared is the orientation average (§ 2.3). The infrared takes any direction at no cost (§ 2.3); Raman takes one per optical run: z, the gap's axis, by default, another chosen before the run (`plans/plan.md` § 1ac, D-h) |
| the charge partition | infrared | Hirshfeld (§ 1a) |
| the modes to compute Raman for — fixed before the run, since a mode exists only after the force-constant run | Raman | every mode — *Raman for*, the setting both engines share: all modes, or the modes in a frequency range a chemist knows before the run ([`engines/vibration.md`](?doc=engines/vibration.md) § 3.1, § 5.11 § 9) |
| the laser's wavelength | Raman, after the run — the polarizability is taken at its frequency (§ 3.1) | — |
| a temperature — the junction's, or a pumped mode's effective one | the spread of the frames along a mode (`vibrational-averaging.md` § 2a, § 5) | the run's thermochemistry temperature, offered by the dialog |

### 0.2 The workflow

| step | what runs | molbuilder today |
|---|---|---|
| **1. The vibration** | SIESTA's force-constant run on the relaxed junction, as today: each free atom nudged ±0.0212 Å along x, y and z, a self-consistent solution at each — $1 + 6\,n_{\mathrm{free}}$ of them; SIESTA prints every atom's Hirshfeld and Voronoi charge after each | **exists** — the charges since 2.35.2 |
| **2. Infrared, every mode** | nothing more: the vibration's finish stores the charges of step 1; asked for a direction and a partition, one function derives each mode's dipole change — the free atoms' → its infrared strength $A_\nu(\hat e)$ and rank (§ 2; a worked example, § 2.4) | **built 2.36.0** — the spectrum viewer and `molbuilder.postprocess.infrared_view` |
| **3. Raman, the chosen modes** | the vibration run's optical phase, after its force-constant phase in the same run: the geometries at $0$ and $\pm\delta Q$ along each chosen mode (§ 3.2); one SIESTA single point at each with its optical calculation (the `OpticalCalculation` option: the light-absorption spectrum from its own states, § 3.1) → the polarizability at each, at the laser's frequency the person enters after the run (static beside it) → its change → the Raman strength $R_\nu(\hat e)$ and rank (§ 3), from the whole cell's change (§ 3.4) | **built** — the optical phase 2.42.0–2.43.0, the strength at a laser 2.44.0: the spectrum viewer and `molbuilder.postprocess.raman_view` ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 10); in a cell holding metal, the value at the laser is the one to read and the static one is not defined (§ 3.1) |
| **4. The displacement** | nothing: the frames' spread at the temperature the person states, and the outermost move — how far the outermost frames move the atom that moves most — from the vibration result (§ 4; an example, § 4.1) | **built 2.36.0** with a displacement as its input; the temperature as its input, the user's ruling of 2026-10-28, to be built ([`plans/plan.md`](?doc=plans/plan.md) § 1al F1) — the *Write frame set…* dialog and `molbuilder.postprocess.outermost_move` |

```mermaid
flowchart TD
    J["relaxed junction<br/>(its structure, labels, held atoms)"] --> FC["1. SIESTA force-constant run<br/>Hirshfeld and Voronoi charges at every step"]
    FC --> MODES["the modes: ω_ν, L^(ν)<br/>(the vibration's finish)"]
    FC --> QS["every atom's charge<br/>at every displaced step"]
    QS --> IR["2. infrared, every mode<br/>∂μ_F/∂Q → A_ν(ê) and its rank"]
    MODES --> IR
    MODES --> DISP["4. the displacement<br/>the frames' spread at a temperature,<br/>the outermost move d_out"]
    MODES --> FR["3. Raman, each chosen mode<br/>frames 0, ±δQ: the run's optical phase"]
    FR --> EL["one SIESTA single point per frame<br/>+ OpticalCalculation along ê"]
    EL --> AL["ε₂(ω) → α_êê(ω_L) at each frame<br/>at the laser the person enters, after the run"]
    AL --> RA["∂α_êê/∂Q → R_ν(ê) and its rank"]
```

**What it costs** (§ 5.1, measured on AuBDTAu): infrared and the
displacement, nothing beyond the vibration already run; keeping the held gold
in Raman, no SIESTA run either; Raman, $1 + 2\times$(the number of modes) SIESTA runs — frame 0 once,
$\pm\delta Q$ for each mode — each one junction SCF and SIESTA's optical
calculation: for five modes about 20–40 min of SCF on 64 ranks, against the
vibration's 3.3 h; the optical calculation's own cost not yet measured.

**Which atoms.** The modes are defined on the **free set** $F$, every atom the
vibration moves, and **the infrared is $F$'s**: its dipole made of the free
atoms' charges. $F$ is the system the person described — they chose what to
free when they described the vibration — and the held atoms are not part of
it *(the user, 2026-10-28: "whatever is free is the whole thing that we're
trying to calculate … whatever is fixed … is not taking into consideration";
"I see no point at all to select any atoms")*. No other set is offered: a
set other than the one the vibration was described for is not the
calculation's intention. How much another set would change the numbers is
measured in § 2.5, as information.

### 0.3 Where each number comes from

| number | file | written by |
|---|---|---|
| the modes $\omega_\nu$, $\mathbf L^{(\nu)}$, the masses | `<label>.spectra.json` | the vibration's finish (`engines/vibration.md` § 5.5) |
| every atom's charge at every force-constant step | `<label>.spectra.json` (`partial_charges`), read from the run's output's *Hirshfeld* and *Voronoi Atomic Populations* blocks | SIESTA (asked by the input since 2.35.2), kept by the vibration's finish since 2.36.0 |
| $\varepsilon_{2,\hat e\hat e}(\omega)$ at a frame | `<label>.optical.nc` (`eps2`), read from each call's `<label>.EPSIMG` | SIESTA's optical calculation, at each frame of the vibration run's optical phase, kept by the run's record step ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 8) |
| the zero-point amplitude $Q_{\mathrm{zp}}$ | `<label>.spectra.json` (`zero_point_amplitude_amu12_ang`) | the vibration's finish |

---

## 1. The quantities

| symbol | what it is | unit |
|---|---|---|
| $F$ | the free set: every atom the vibration moves | — |
| $m_k$ | atom $k$'s mass, as the vibration result states it | amu |
| $\omega_\nu$ | mode $\nu$'s frequency, as the result states it | cm$^{-1}$ |
| $\mathbf L^{(\nu)}$ | mode $\nu$'s displacement pattern over $F$, normalised with the masses, $\sum_{k\in F} m_k\,\lvert\mathbf L_k^{(\nu)}\rvert^2 = 1$ (`engines/vibration.md` § 6.3); zero on every held atom | amu$^{-1/2}$ |
| $Q_\nu$ | how far the structure is moved along mode $\nu$: atom $k$ moves by $Q_\nu\,\mathbf L_k^{(\nu)}$ | amu$^{1/2}$ Å |
| $q_a$ | atom $a$'s partial charge: its valence charge less the electrons a partition gives it — positive where electrons left it; by Hirshfeld [Hirshfeld1977] or Voronoi [FonsecaGuerra2004], both computed (§ 1a) | $e$ |
| $\bar{\mathbf r}_F^{\,0}$ | the free atoms' centre of mass at equilibrium, $\sum_{a\in F} m_a\mathbf r_a^0 \big/ \sum_{a\in F} m_a$, the masses the result states — a fixed point *(the user, 2026-10-24: "center should be mass center")* | Å |
| $\hat e$ | the direction of the light's field, the person's choice | — |
| $\varepsilon_{\hat e\hat e}(\omega)$ | the cell's dielectric function for a field along $\hat e$, $\varepsilon_1 + i\varepsilon_2$ | — |
| $V_{\mathrm{cell}}$ | the cell's volume | Å³ |

### 1a. The two charge partitions — Hirshfeld and Voronoi

*(Decided by the user, 2026-10-24: "both on"; "default to hirshfeld".)*
**Both are computed, every number is stated with the partition it came from,
and Hirshfeld is the one shown unless the person picks Voronoi.**

**What they share.** Both give each atom a charge, and both start from the
same thing — how the electrons moved when the atoms bonded: the real electron
density $\rho$ less the density the same atoms would have as neutral, isolated
atoms,

$$
\Delta\rho(\mathbf r) = \rho(\mathbf r) - \sum_{b} \rho_b^{\mathrm{free}}(\mathbf r) .
$$

(SIESTA works with the valence electrons; $\rho_b^{\mathrm{free}}$ is atom
$b$'s neutral valence density on its own basis.) A charge of $+0.1\,e$ means
the atom ended with 0.1 electron fewer than it had alone. **The two differ
only in how they share $\Delta\rho$ between neighbouring atoms**:

$$
q_a^{\mathrm{Hirshfeld}} = -\int w_a(\mathbf r)\,\Delta\rho(\mathbf r)\,d\mathbf r,
\quad w_a(\mathbf r) = \frac{\rho_a^{\mathrm{free}}(\mathbf r)}{\sum_b \rho_b^{\mathrm{free}}(\mathbf r)} ;
\qquad
q_a^{\mathrm{Voronoi}} = -\int_{V_a} \Delta\rho(\mathbf r)\,d\mathbf r ,
$$

$V_a$ the points nearer to atom $a$'s nucleus than to any other.

- **Voronoi draws a hard wall.** Every point belongs to the nearest nucleus —
  by distance alone, whatever the atoms' sizes — so on the S–Au bond the wall
  sits exactly halfway between the two nuclei, and each atom takes all the
  moved charge on its side (a point exactly on the wall is shared equally).
- **Hirshfeld shares softly.** At each point it asks how much density a lone
  neutral S and a lone neutral Au would each have there, and splits the moved
  charge in that ratio: near S, S takes nearly all of it; at the bond's middle
  it is shared; near Au, Au takes it — no wall, a split that changes smoothly
  through the bond.

**The two shares along a bond, drawn.** The share of the moved charge that
goes to S, at points along an S–Au bond 2.4 Å long. *An illustration with
model densities* — one-dimensional Gaussians for the two neutral atoms (S: 6
valence electrons, width 0.8 Å; Au: 11, width 1.3 Å), not SIESTA's:

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1971c2, #d9480f"}}}}%%
xychart-beta
    title "share of the moved charge given to S, along S (0 Å) – Au (2.4 Å)"
    x-axis "position along the bond (Å)" [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
    y-axis "share to S" 0 --> 1
    line [0.830, 0.753, 0.636, 0.478, 0.306, 0.163, 0.073, 0.028, 0.010]
    line [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
```

*(Blue: Hirshfeld — a smooth hand-over through the bond, set by the two atoms'
own densities. Orange: Voronoi — everything on S's side of the midpoint, a
step.)*

Both are read in SIESTA 5.4.2's source (`partial_charges.F90`: Voronoi
subtracts the neutral-atom density at each grid point and gives the rest to
the nearest atom; Hirshfeld weights the density by $w_a$); the Hirshfeld form
above follows because $w_a\sum_b\rho_b^{\mathrm{free}} = \rho_a^{\mathrm{free}}$
integrates to the atom's valence charge.

**What it means for this method.**

1. **Every number changes with the partition.** The free atoms' charge
   depends on where the boundary between them and the held atoms is drawn,
   and that boundary is a convention — no partition is the true one. Where
   the ranking of the modes changes with it, the ranking is set by how the
   bond is split. (In the worked example of § 2.4 the one bright mode reads
   0.13 km/mol by Hirshfeld and 47.2 by Voronoi; on CO₂, § 7, the two
   partitions give the same ranking, the strengths about twice apart.)
2. **Smoothness as atoms move.** The derivatives compare charges at
   geometries $2\delta$ = 0.042 Å apart. Under Voronoi the wall moves with the
   atoms and points of SIESTA's grid jump from one atom to the other; under
   Hirshfeld the shares change gradually, so its derivatives are expected to
   be less noisy — the reason it is shown first. This is expected, not yet
   measured; a displacement sweep (§ 6) shows it, where one is run.
3. **The cost is the same and small**: each is a sum over SIESTA's grid after
   every step, with no extra self-consistent solution, and both are asked for
   in the same run.

---

## 2. Infrared — the dipole change along a mode (workflow step 2)

**The idea.** Light whose field points along $\hat e$ pushes positive and
negative charge apart along $\hat e$. A mode absorbs it when moving along the
mode changes the dipole along $\hat e$ — the charges move, or the
charged atoms move, or both. How strongly is the square of that change.

```text
   equilibrium              the mode at +Q             what the light sees
   δ−   δ+                   δ−      δ+               the dipole along ê
   ●────●            →       ●───────●                changes:  Δμ ≠ 0  → infrared-active
                         (bond longer, charge         stays:    Δμ = 0  → infrared-silent
                          shifted)
```

### 2.1 The free atoms' charge and dipole

**Why partial charges.** The cell's own dipole is not defined for a periodic
metal, and the Born effective charges SIESTA can compute come from the
Berry-phase polarization, which is defined for an insulator [KingSmith1993]
(SIESTA 5.4.2 manual, `BornCharge` and `PolarizationGrids`: the system "must be
an insulator"; `normal-modes.md` § 4b.7); the charges of the atoms are
defined in any cell. From the free atoms' charges, at any geometry $R$:

$$
q_F(R) = \sum_{a\in F} q_a(R),
\qquad
\boldsymbol\mu_F(R) = \sum_{a\in F} q_a(R)\,\bigl(\mathbf r_a(R) - \bar{\mathbf r}_F^{\,0}\bigr).
$$

As the free atoms move, charge moves between them and the held atoms, so
$q_F$ changes along a mode. Two numbers are given for each mode by the one
derivation — the dipole change as the full vector; the
spectrum viewer shows the strength it gives along $\hat e$ (§ 2.3) and the
charge moved:

- **the dipole change** $\partial\boldsymbol\mu_F/\partial Q_\nu$, taken about
  the free atoms' own centre of mass at equilibrium, $\bar{\mathbf r}_F^{\,0}$, a fixed point — no point outside
  them enters it. Where the mode moves charge, the number depends on
  that centre: moving it by $\Delta\mathbf r$ changes
  $\partial\boldsymbol\mu_F/\partial Q_\nu$ by
  $-(\partial q_F/\partial Q_\nu)\,\Delta\mathbf r$, which is why the
  charge moved is reported beside it;
- **the charge the mode moves** $\partial q_F/\partial Q_\nu$ into or out
  of the free atoms — beside the dipole change, not folded in.

### 2.2 Read from the force-constant run, with no extra calculation

SIESTA's force-constant run moves each free atom $k$ by $\pm\delta$ along x,
y and z in turn (`FC.Displacement`, 0.04 bohr = 0.0212 Å by default) and
solves the electrons self-consistently at each step. With
`Charge.Hirshfeld geometry` and `Charge.Voronoi geometry` in its input —
written there when *Compute IR* is on (`engines/vibration.md` § 3.1, § 5.3)
— SIESTA prints
every atom's partial charge, by both partitions, "at the end of every geometry
step" (SIESTA 5.4.2 manual, `Charge.Hirshfeld`; `grdsam.F`), to $10^{-6}\,e$
(`partial_charges.F90`), each column summing to zero for a neutral cell.

The steps come in a fixed order — displaced atom, then direction, then side,
$-$ before $+$ (`engines/vibration.md` § 5.4):

```text
 step 0          the relaxed geometry
 steps 1–6       free atom 1:  −x, +x, −y, +y, −z, +z
 steps 7–12      free atom 2:  −x, +x, −y, +y, −z, +z
 …               every free atom in turn: 1 + 6·n_free steps
```

By central differences, each displaced geometry's dipole taken at that
geometry's own positions — so both the charges changing and the charged
atoms moving are counted:

$$
\frac{\partial\boldsymbol\mu_F}{\partial R_{k\beta}}
\approx \frac{\boldsymbol\mu_F(R^{+}_{k\beta}) - \boldsymbol\mu_F(R^{-}_{k\beta})}{2\delta},
\qquad
\frac{\partial q_F}{\partial R_{k\beta}}
\approx \frac{q_F(R^{+}_{k\beta}) - q_F(R^{-}_{k\beta})}{2\delta},
$$

where $R^{\pm}_{k\beta}$ is the geometry with free atom $k$ moved by
$\pm\delta$ along $\beta \in \{x, y, z\}$. Every free atom counts, by its own
charge and by its motion. Projected onto each mode,

$$
\frac{\partial\boldsymbol\mu_F}{\partial Q_\nu}
= \sum_{k\in F}\sum_{\beta} \frac{\partial\boldsymbol\mu_F}{\partial R_{k\beta}}\;L^{(\nu)}_{k\beta},
\qquad
\frac{\partial q_F}{\partial Q_\nu}
= \sum_{k\in F}\sum_{\beta} \frac{\partial q_F}{\partial R_{k\beta}}\;L^{(\nu)}_{k\beta}.
$$

Every mode's values come out of the one run that gave the modes, once for
each partition.

```mermaid
flowchart LR
    S["the steps' charges<br/>q_a at each ±δ"] --> MU["the free atoms' dipole μ_F<br/>and charge q_F at each step"]
    MU --> D["∂μ_F/∂R and ∂q_F/∂R<br/>for each free atom and direction"]
    L["each mode's pattern L^(ν)<br/>(the result)"] --> P
    D --> P["projected on the mode:<br/>∂μ_F/∂Q_ν, ∂q_F/∂Q_ν"]
    P --> A["A_ν(ê) and its rank"]
```

### 2.3 The infrared strength for a unit field along $\hat e$

For a molecule held in one orientation, in light whose field points along
$\hat e$, mode $\nu$'s integrated absorption is

$$
A_\nu(\hat e) = \frac{N_A}{4\varepsilon_0 c^2}\,
\Bigl(\hat e\cdot\frac{\partial\boldsymbol\mu_F}{\partial Q_\nu}\Bigr)^{2}
= 126.768\ \tfrac{\mathrm{km}}{\mathrm{mol}}
\times \Bigl(\hat e\cdot\frac{\partial\boldsymbol\mu_F}{\partial Q_\nu}\Bigr)^{2} ,
$$

with the derivative in D Å$^{-1}$ amu$^{-1/2}$ (a charge of $e$ moved 1 Å is
4.8032 D, so a derivative in $e$ amu$^{-1/2}$ is multiplied by 4.8032). A free
molecule tumbles in the light, and its average over orientations is the
familiar $\frac{N_A}{12\varepsilon_0 c^2}\lvert\partial\boldsymbol\mu/\partial Q\rvert^2 = 42.256\ \mathrm{km/mol}$
per $(\mathrm D/\text{Å})^2/\mathrm{amu}$
[Porezag1996] — a third of the value above, since $\cos^2\theta$ between a
dipole and the field averages to $1/3$ over orientations; held in the
junction, it does not tumble. (Both prefactors computed from the CODATA
constants.)

**Held or free — the result decides** *(the user, 2026-10-24: "if no fix atoms
are listed, that means gas phase, and you can keep conventional average, but
when we have fixed atoms that means we have rigidity and surface effects, so
it is correct to see specific peak respond better in certain EM and the result
is real")*. Where the vibration held atoms, the molecule is rigid in a
junction or on a surface and meets the light in one orientation: its strength
is $A_\nu(\hat e)$ above, along the light's direction, and a peak that
answers one polarization more than another is a real result. Where nothing
was held, the molecule is free and tumbles: its strength is the conventional
orientation average, $42.256\ \mathrm{km/mol}\times\lvert\partial\boldsymbol\mu/\partial Q_\nu\rvert^2$,
with no direction — the mean of the three axes' $A_\nu$, the number a free
molecule's infrared reports (`spectra.infrared.orientation_of`). Each mode's $A_\nu(\hat e)$ is also given relative to the
strongest real mode's — an imaginary mode is never the reference.

**The direction is applied after the run.** The light's field enters only
through the projection $\hat e\cdot\partial\boldsymbol\mu_F/\partial Q_\nu$:
the coupling is first order in the field, so the run needs no field, and the
force-constant charges give the whole vector $\partial\boldsymbol\mu_F/\partial Q_\nu$
— one run answers every direction, each a dot product (the viewer's *Light
along*, `web/spectra.md` § 3.2).

**Given** by the one derivation (`spectra.infrared.infrared_view`), and drawn
by the spectrum viewer as its column and its lane: $A_\nu(\hat e)$ for every mode, and
$A_\nu(\hat e)/\max_{\nu'}A_{\nu'}(\hat e)$; the charge the mode moves beside
it.

### 2.4 A worked example — the toy junction, from its real run

The charges are those the force-constant run of § 7 printed: the toy junction
`claude-transport-walk` — a straight chain of 22 hydrogen atoms, ten held on
each side and the middle two (user indices 10, 11) free, the set $F$;
$\hat e = \hat z$, along the chain. The run predates the result keeping its
charges (2.36.0), so they were read from its output by the record step's own
reading (`siesta_vibration.partial_charges_of`) and the built function
derived the numbers.

**The modes** (the run's result): two free hydrogen atoms give six modes. The
four that move them off the chain's axis (along x or y) are imaginary (−666.35 and
−481.53 cm⁻¹, each twice) — a straight hydrogen chain is not a minimum
sideways; their dipole change along z is zero. The two along the chain:

| mode | frequency | the two free atoms | $\partial\mu_{F,z}/\partial Q$ ($e$ amu$^{-1/2}$), Hirshfeld / Voronoi | $A_\nu(\hat z)$ (km/mol), Hirshfeld / Voronoi | $\partial q_F/\partial Q$ ($e$ amu$^{-1/2}$ Å$^{-1}$), Hirshfeld / Voronoi |
|---|---|---|---|---|---|
| 5 | 2023.77 cm⁻¹ | move together, both along −z | −0.0067 / −0.1270 | **0.1 / 47.2** | 0 / 0 |
| 6 | 2548.37 cm⁻¹ | move apart, mirror images | 0 / 0 | **0 / 0** | −0.0128 / −0.0206 |

**Why — the symmetry, and what it checks.** The chain is the same on both
sides of a mirror plane through the free pair's centre ($z$ = 11 Å).

- **Mode 6** is its own mirror image: the two atoms move apart as each
  other's reflection. The dipole along z changes sign under that mirror, so
  it cannot change in this mode — **infrared-silent**, under either
  partition. The pair's total charge does not change sign under the
  mirror, so it *can* change — and does: the mode moves charge between the
  free pair and the held atoms. Its dipole change reads zero about the pair's
  centre, which lies on the mirror plane; about a centre $\Delta z$ away it
  would read $-(\partial q_F/\partial Q)\,\Delta z$ (§ 2.1).
- **Mode 5** turns into its own reverse under the mirror (both atoms shift the
  same way). Now the dipole along z can change — **infrared-active** — and
  the total charge cannot change at first order.

Both zeros come out of the numbers, not out of an assumption: the method
reads symmetry correctly. The two partitions agree on which mode is dark and
put mode 5's strength far apart (0.1 against 47.2 km/mol): for two hydrogen
atoms in a chain, how the electrons between them are shared is most of the
answer — § 1a's point, in numbers.

### 2.5 What another set of atoms would give — measured, for information

The set is the free atoms because they are the system the vibration was
described for (§ 0, *Which atoms*). The numbers below say how much the
strengths move when the same arithmetic sums the dipole over another set —
the AuBDTAu vibration on Sol (2026-10-28; 20 free atoms, the 12 of BDT and 8
Au at its two contacts; 292 held; light along z; Hirshfeld), every number
from the run's stored charges by the arithmetic of § 2.1–2.3 summed over
each set, the free atoms' row the built function's own:

| the set the dipole is summed over | atoms | 498 cm⁻¹ | 1040 cm⁻¹ | 1073 cm⁻¹ | 125 cm⁻¹ | 3071 cm⁻¹ | every mode |
|---|---|---|---|---|---|---|---|
| BDT | 12 | 26.8 | 29.1 | 6.1 | 2.0 | 0.15 | 74.5 |
| **the free atoms** | 20 | 3.1 | 4.0 | 0.03 | 29.3 | 28.2 | 103.2 |
| BDT and the Au within 5 Å of it | 36 | 0.2 | 1.1 | 2.9 | 5.9 | 56.9 | 122.2 |
| BDT and the Au within 11 Å of it | 188 | 9.0 | 2.3 | 12.5 | 1.9 | 68.6 | 149.4 |
| every atom of the cell | 312 | 9.5 | 4.2 | 13.9 | 1.6 | 54.8 | 144.2 |
| the held atoms alone | 292 | 23.4 | 0.0 | 15.1 | 44.7 | 4.4 | 180.8 |

(km/mol.) The strengths move with the set's edge — taking in more of the
gold changes them, up to the whole cell — and atoms that never move still
give a strength, from the charge induced on them (the held atoms' row).

**What the literature says** — references, not a verdict on any
calculation:

- For a molecule on one metal surface with vacuum above it, a measured
  infrared band follows the dipole change perpendicular to the surface, the
  metal's image response included [Greenler1966, Hoffmann1983];
  first-principles studies compute the infrared of adsorbates on metal slabs
  [Zaitsev2018, Zhou2021], a slab's dipole taken in a cell with a vacuum gap
  and a dipole correction [Bengtsson1999].
- The polarization of a cell periodic in every direction is defined for an
  insulator, through the geometric phase of its states [Resta1994,
  RestaVanderbilt2007]; a charge moving near conductors induces a current in
  them [Shockley1938, Ramo1939].
- A molecular junction's vibrations have been measured by Raman scattering
  together with its conduction [Ward2008]; which modes inelastic electron
  tunnelling shows is described by propensity rules [TroisiRatner2006,
  Paulsson2008].
- A dipole derivative splits into a charge term, a charge-flux term and a
  dipole-flux term; atomic charges alone carry the first two
  [HaidukeBruns2005].
- With atoms held, the modes are a partial Hessian's, an approximation
  assessed against the full Hessian [VesterOlsen2024].

---

## 3. Raman — the polarizability change along a mode (workflow step 3)

**The idea.** Light's field shifts the electrons; the polarizability says how
easily. A mode is Raman-active when moving along it makes the electrons
easier or harder to shift — when the polarizability changes along the mode.
How strongly is the square of that change.

### 3.1 The static polarizability at one geometry, from SIESTA's own states

The polarizability is a property of the electrons at a geometry: how far
light of a given polarization would shift them. **It is computed from the
states SIESTA has already solved — no field is applied.**

**What "the optical calculation" is.** After SIESTA has solved the electrons
at a geometry it holds every occupied and every empty state. Its
`OpticalCalculation` option then computes, for each pair of an occupied and
an empty state, how strongly light polarized along $\hat e$ couples the two
(the dipole transition between them), and from those couplings the
structure's light-absorption spectrum, $\varepsilon_2(\omega)$. The static
polarizability is the same couplings, each weighted by one over the energy
between its two states — one integral over that spectrum (below).

```text
   empty states   ─────────  ─────────  ─────────
                      ▲          ▲           ▲         each arrow: one pair, its coupling to
                      │          │           │         light along ê, at the energy of the gap
   occupied       ─────────  ─────────  ─────────
                  → ε₂(ω): every pair's coupling, placed at its gap ħω (the absorption spectrum)
                  → the static polarizability: the couplings weighted by 1/gap, summed
```

In the input it is:

```text
OpticalCalculation        true
Optical.PolarizationType  polarized
%block Optical.Vector
  0.0  0.0  1.0           # e, here along z
%endblock Optical.Vector
```

with `Optical.Mesh` (its k-points), `Optical.NumberOfBands` (all bands by
default) and `Optical.Energy.Maximum` (SIESTA's default 10 Ry; molbuilder
writes 3 Ry — *The cutoff*, below). In SIESTA's words:

- `OpticalCalculation true`, with `Optical.PolarizationType polarized` and
  `Optical.Vector` $\hat e$: the imaginary part of the dielectric function for
  a field along $\hat e$, $\varepsilon_{2,\hat e\hat e}(\omega)$, "using first
  order perturbation theory" from "the dipolar transition matrix elements
  between different eigenfunctions of the self-consistent Hamiltonian" —
  with the position operator only for a cell SIESTA judges an atom or a
  molecule and samples at Γ alone, and with the momentum operator
  otherwise: a junction, its atoms joined across the cell's boundary, takes
  the momentum formulation and needs `Optical.Mesh` (SIESTA 5.4.2 manual,
  `OpticalCalculation`; `Src/optical.F`). It covers the transitions between
  different states (interband) and is written to `<label>.EPSIMG`.
- The real part at any frequency is one integral of it — the
  Kramers–Kronig relation, its principal value taken:

  $$
  \varepsilon_{1,\hat e\hat e}(\omega) = 1 + \frac{2}{\pi}\,
  \mathrm{P}\!\int_0^{\infty}
  \frac{\omega'\,\varepsilon_{2,\hat e\hat e}(\omega')}{\omega'^2-\omega^2}\,d\omega' ,
  $$

  which at $\omega = 0$ is the static one,
  $1 + \frac{2}{\pi}\int_0^\infty \varepsilon_2(\omega')/\omega'\,d\omega'$.

**At the laser's frequency, the static value beside it** *(decided
2026-10-24 — the user: "yes", to taking $\alpha$ at the laser the person enters
after the run)*. In the pair sum of the optical calculation each pair of
states a gap $\Delta\varepsilon$ apart adds to $\alpha$ with the weight
$2/\Delta\varepsilon$ at $\omega = 0$ and
$2\Delta\varepsilon/(\Delta\varepsilon^2-\omega_L^2)$ at the laser's
$\omega_L$. For a 633 nm laser, $\hbar\omega_L$ = 1.96 eV:

| a pair | static weight | at the laser | what it means |
|---|---|---|---|
| the molecule's, a 3 eV gap | 0.67 | 1.16 | 1.7 times larger — the laser near enough to the gap to count (pre-resonance) |
| the gold's at the Fermi level, a 0.05 eV gap | 40 | −0.026 | from dominant to negligible |
| a gap equal to the laser's energy | — | very large | resonance Raman, a real effect the static number cannot show |

So the polarizability at the laser is what an experiment measures: the
near-Fermi gold pairs shrink to almost nothing, the molecule's pairs near the
laser are enhanced, and the result moves with the laser as a measured
spectrum does. **The static value** — the far-from-resonance limit in which
the textbook Raman activity is defined [Porezag1996], independent of the
laser — is shown beside it, always; for a molecule with a gap it is
defined, and beside a metal it is not (below), which the reader judges with
the cell in view. (molbuilder's PySCF Raman
takes its static polarizability from coupled-perturbed theory,
`engines/vibration.md` § 4.6, this one from independent states, § 3.3, so the
two differ by construction.)

**The laser costs no calculation.** Each frame's optical calculation writes
the whole absorption spectrum $\varepsilon_2(\omega)$, up to
`Optical.Energy.Maximum`, in one run; the polarizability at zero or at any
laser is one integral over that saved spectrum, done after the run — so the
wavelength is entered afterwards, like the infrared's light direction, and
changed or compared on the same runs. It asks only a broadening above zero
(`Optical.Broaden`), which near a resonance is a setting to check, not a
constant. (The measured Raman intensity carries, besides the polarizability,
a laser-dependent factor and a temperature factor; molbuilder reports the
strength, on both engines, and applies neither — decided 2026-10-26,
[`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 13, M4-d.)

**The cutoff — which pairs are computed, and what it changes** *(measured
2026-10-28; the user: "go ahead with 3 Ry")*. `Optical.Energy.Maximum` is
the top of the photon-energy axis the calculation runs on — not a field, and
not an energy given to the molecule: no field is applied (above). A pair of
states whose gap lies above it is skipped — SIESTA computes no dipole element
for it (`Src/transition_rate.F:250`, `Src/optical.F:566`) — and the spectrum
is written up to it. Each pair adds to $\alpha(\omega)$ its oscillator
strength over $\Delta\varepsilon^2-\omega^2$ (atomic units), so at a
visible laser a pair weighs about one over its gap squared: a pair at 40 eV
counts about a four-hundredth of one at 2 eV for the same strength. And the
pairs far up are not a molecule's: a localized basis has no continuum, a DZP
basis describes optical properties reasonably below about 10 eV and a more
complete one to about 20 eV [SiestaOptical2021], and its empty states above
are artefacts of the confined orbitals — CO₂'s DZP basis, the highest 116 eV
above $E_F$.

Measured on our runs, read through molbuilder's own reader and its
Kramers–Kronig integral (`spectra.raman.eps1_minus_one`), the spectrum cut at
each energy against the full 10 Ry:

| | where the spectrum ends | cut at 2 Ry | 3 Ry (41 eV) | 5 Ry (68 eV) |
|---|---|---|---|---|
| CO₂, $\alpha$ at 633 nm | 126 eV | −6.4 % | −1.1 % | −0.12 % |
| CO₂, $\partial\alpha/\partial Q$, 1250 cm⁻¹, at 633 nm | — | +0.05 % | −0.34 % | −0.06 % |
| a toy junction (H, periodic), $\partial\alpha/\partial Q$, 2548 cm⁻¹ | 34 eV | +0.17 % | 0.00 % | 0.00 % |
| H₂, $\alpha$ at 633 nm | 65 eV | −0.05 % | −0.05 % | 0.00 % |

The static values move the same way. **molbuilder writes 3 Ry**: the Raman
derivative within 0.4 % of 10 Ry on every run measured — far inside what the
method's own approximations (independent states, no local field, a DZP
basis) leave — while the cost falls: in CO₂, half the pairs SIESTA computes
at 10 Ry lie between 41 and 136 eV (from its eigenvalues, `<label>.EIG`), the
optical step was 58 % of the call (9.7 of 16.7 s), and each spectrum file is
30 % of the size. A person raises it for a basis with many more empty states,
or to check a result against it. **The f-sum in the `.EPSIMG` header reads
lower under a cutoff, by design**: it weighs each pair by its energy, so the
pairs the cutoff drops are the ones it counts most — CO₂'s, 0.92 at 10 Ry, is
76 % of that within 3 Ry — and it is not a check of $\alpha$; the check of
the range is how far the Raman derivative moves when the cutoff is raised.

The cell's polarizability along $\hat e$ at the frequency $\omega$, as a
polarizability volume:

$$
\alpha_{\hat e\hat e}(\omega) = \frac{V_{\mathrm{cell}}}{4\pi}\,
\bigl[\varepsilon_{1,\hat e\hat e}(\omega) - 1\bigr]
$$

(from $\mathbf P = \varepsilon_0(\varepsilon_r - 1)\mathbf E$, a dipole
$\mathbf P V_{\mathrm{cell}}$, divided by $4\pi\varepsilon_0$).

**This is the whole cell's polarizability**, the held gold's electrons in it,
and it is kept: what is reported from it, and why, is § 3.4.

**Beside a metal — what the published work says, and what one test showed**
*(2026-10-27)*. SIESTA's $\varepsilon_2$ keeps every pair of states more than
$10^{-3}$ Ry apart, each weighted by $f_i(1-f_j)$ — the lower state's
occupation times the upper's emptiness (`Src/optical.F:574`), no emission
term — and its momentum element divided by the gap (`Src/transition_rate.F`,
`Src/optical.F`). A metal's response within one band, the free-electron
(Drude) part, is not in it: SIESTA reports its plasma frequency apart, in
the `.EPSIMG` header, counted over the states within the broadening of the
Fermi level and printed to four decimals in Ha² — so a coarse k-mesh counts
none, and reads 0. `Optical.Broaden` 0, SIESTA's default, makes its
frequency step zero (`dw = smooth/10`, `Src/optical.F`).

- **The static polarizability of a cell holding a periodic metal does not
  exist**: the free-electron part diverges as $\omega\to0$ [Prandini2019].
  What SIESTA's spectrum gives at $\omega = 0$ is the remainder, set by the
  near-Fermi pairs the k-mesh and the broadening happen to keep; no published
  method uses it.
- **The derivative at the laser's frequency, with a finite broadening, is
  the published object** — for molecules with a lifetime broadening
  [Jensen2005a, Jensen2005b], for solids [Gillet2013, Taghizadeh2020]: about
  0.1–0.2 eV, at least the spacing between the discrete transitions
  [Gillet2013]; near a resonance the value scales strongly with it
  [Saikin2009]. A metal's near-Fermi pairs weigh almost nothing at a laser
  (the table above).
- **A molecule on or between metal, as others compute it**: a metal cluster
  and time-dependent theory, static or at the laser [Zhao2006, Aikens2006,
  Saikin2009]; a periodic slab with a static field along the surface normal,
  the modes those of the molecule and its S-bonded gold, the other gold
  infinitely heavy — the held-set choice molbuilder's vibration makes — and
  the component along the normal alone [Zayak2011]; junction theory that
  projects the polarizability on the molecule [Galperin2008]. **None
  differentiates the independent-state dielectric function of a periodic
  metal–molecule–metal cell**, molbuilder's route: it is untested in the
  literature.
- **What is compared with experiment** is the ratio between the molecule's
  modes: a junction's or a surface's measured Raman is dominated by the
  field enhancement of the metal's gap, which no small cell holds
  [Zhao2006, Ward2007], and the absolute values of the cell's derivative
  depend on the binding site [Zayak2011].
- **k-points**: a Raman derivative converges far more slowly with the k-mesh
  than $\varepsilon$ itself [Gillet2013], and a metal's optics needs dense
  meshes [Prandini2019]; no published test of a molecule-on-metal
  derivative's convergence was found.
- **A free atom that is part of the metal** — measured 2026-10-27 on a
  hydrogen chain whose two free atoms are links of the chain (§ 7): at the
  equilibrium the chain's one band gives no interband pair, $\alpha\approx0$;
  either displacement opens a gap at the Fermi level, so $\alpha$ jumps to a
  finite value at $+\delta Q$ and $-\delta Q$ alike — not a smooth function
  of the displacement, its central difference a remainder — as a half-filled
  chain's dimerization does [Heeger1988]. Widening the broadening from 0.027
  to 0.1 eV moved the static $\alpha$ by 48 % on a 1×1×8 optical mesh and the
  value at 633 nm by 3 %. In a junction the free gold beside the anchors is
  where this can enter, the static value first.

### 3.2 Its change along a mode, and the Raman strength

**What the Raman calculation asks, and what it takes itself** *(decided
2026-10-24 — the user: "go ahead with your recommendation")*:

- **The step is the vibration's own, taken, never typed.** Every free atom is
  moved by $\pm\delta Q\,\mathbf L_k^{(\nu)}$,
  $\delta Q = \delta/\max_k\lvert\mathbf L_k^{(\nu)}\rvert$ — the atom that moves
  most by $\delta$, the force-constant run's own step read from its result
  (`partial_charges.displacement_ang`, `engine_metadata.fc_displacement_ang`),
  in the same linear range — and the step is stated beside the strengths. It
  is not the frame generator's spread, which is the motion to average over
  (§ 4), and its averaging weights do not enter.
- **The modes are the person's; the equilibrium is shared.** The optical
  phase takes the modes *Raman for* picks — fixed before the run, since a mode
  exists only after the force-constant run: every mode, or those in a
  frequency range ([`engines/vibration.md`](?doc=engines/vibration.md) § 3.1) —
  and runs the undisplaced geometry once and the two displaced ones for each
  mode: $1 + 2N$ runs (§ 5.1).
- **The light's direction: z by default, another chosen before the run.** One
  optical run gives one direction's entry of the polarizability's change; any
  direction after the run would need the whole table — six optical runs a
  geometry — decided once an optical run's cost on a junction frame is
  measured (§ 5.1). The infrared, by contrast, takes any
  direction after the run at no cost (§ 2.3).
- **The settings the runs read are declared** — the optical keywords and
  *Compute IR*'s charges alike shown on Task setup and in each deck's
  provenance, as every input is (`plans/plan.md` § 1ac, L4).

Each geometry is one SIESTA single point with the optical calculation in
its input — a call of the vibration run's optical phase (§ 5).

```text
   frame 2: −δQ                 frame 0: equilibrium              frame 1: +δQ
   every free atom moved        as relaxed                        every free atom moved
   by −δQ·L  (largest: −δ)                                        by +δQ·L  (largest: +δ)
        │                             │                                 │
   one SIESTA run               one SIESTA run                    one SIESTA run
   α(−δQ)                       α(0)                              α(+δQ)
        └──────────── ∂α/∂Q ≈ [ α(+δQ) − α(−δQ) ] / 2δQ ───────────────┘
```

$$
\frac{\partial\alpha_{\hat e\hat e}}{\partial Q_\nu}
\approx \frac{\alpha_{\hat e\hat e}(+\delta Q) - \alpha_{\hat e\hat e}(-\delta Q)}{2\,\delta Q},
\qquad
R_\nu(\hat e) = \Bigl\lvert\frac{\partial\alpha_{\hat e\hat e}}{\partial Q_\nu}\Bigr\rvert^{2}
\quad[\text{Å}^4/\mathrm{amu}] ,
$$

the squared modulus of a complex change at a laser (§ 3.2a, step 4).

$R_\nu(\hat e)$ is the Raman strength for light polarized along $\hat e$,
incoming and scattered. The textbook Raman activity $45a'^2 + 7\gamma'^2$
[Porezag1996] — $a'$ the mean of the polarizability's derivative, $\gamma'$
its anisotropy — is a tumbling molecule's, averaged over orientations and
summed over both scattered polarizations. The same $\hat e\hat e$ component,
averaged over orientations, is
$\langle\alpha'^{\,2}_{\hat e\hat e}\rangle = (45a'^2 + 4\gamma'^2)/45$
(checked by sampling random rotations of a random tensor): related, not the
same number.

**Reported**: for each chosen mode, $R_\nu(\hat e)$ from the whole cell's
change (§ 3.4) and $R_\nu(\hat e)/\max_{\nu'}R_{\nu'}(\hat e)$. Frame 0's $\alpha$ is the
equilibrium polarizability, the same for every mode, reported once.

**On the toy junction of § 2.4**, the step would be
$\delta Q$ = 0.0301 amu$^{1/2}$ Å for both real modes (each free atom
carries $\lvert\mathbf L\rvert$ = 0.7043 amu$^{-1/2}$); no Raman number is given
here: the hydrogen chain's free atoms are links of its metal, which the
derivative cannot follow (§ 3.1, § 7).

### 3.2a From a laser wavelength to a Raman strength — the steps *(M4, 2026-10-26)*

**A new wavelength costs no calculation**: every step reads the absorption
spectra the run already stored (`<label>.optical.nc`, one per geometry), so
a laser changed on the page is a new number in a second
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 10).

1. **The wavelength as a photon energy**:

   $$
   \hbar\omega_L\,[\mathrm{eV}] = \frac{1239.842}{\lambda\,[\mathrm{nm}]}
   $$

   | laser | $\hbar\omega_L$ |
   |---|---|
   | 532 nm | 2.331 eV |
   | 633 nm | 1.959 eV |
   | 785 nm | 1.579 eV |

2. **Each geometry's spectrum smoothed to the width asked** — the run spread
   every transition as a Gaussian of its own width $\sigma_{\mathrm{run}}$
   (`Optical.Broaden`); a wider $\sigma$ is one more Gaussian of
   $\sqrt{\sigma^2-\sigma_{\mathrm{run}}^2}$ — exact for a transition a few
   widths above zero, not below: SIESTA keeps only the part of each line
   above zero (`Src/optical.F`), so a line within a few widths of zero — a
   metal's near-Fermi pairs — keeps less weight widened after the run than a
   run at the wider width would. The laser does not enter here.
3. **The real part at the laser's energy**, by Kramers–Kronig (§ 3.1):

   $$
   \varepsilon_1(\omega_L) - 1 = \frac{2}{\pi}\,
   \mathrm{P}\!\int_0^{E_{\max}}
   \frac{\omega'\,\varepsilon_2(\omega')}{\omega'^2-\omega_L^2}\,d\omega' .
   $$

   This is where the laser acts: a transition whose energy is near
   $\hbar\omega_L$ weighs heavily (pre-resonance), one far below it almost
   nothing; $\omega_L = 0$ gives the static value shown beside.
4. **The polarizability at the laser**, complex:
   $\alpha_{\hat e\hat e}(\omega_L) = \frac{V_{\mathrm{cell}}}{4\pi}\,[\varepsilon(\omega_L)-1]$
   in Å³ — its real part $\frac{V_{\mathrm{cell}}}{4\pi}[\varepsilon_1(\omega_L)-1]$, its
   imaginary part $\frac{V_{\mathrm{cell}}}{4\pi}\,\varepsilon_2(\omega_L)$, the absorption at
   the laser itself. Off any transition the imaginary part is zero; near
   one, and in a cell holding metal, it is not, and the Raman tensor is the
   complex susceptibility's change [Gillet2013].
5. **Its change along each mode**, both sides at the same $\omega_L$:

   $$
   \frac{\partial\alpha_{\hat e\hat e}}{\partial Q_\nu}(\omega_L) =
   \frac{\alpha_{\hat e\hat e}(+\delta Q;\,\omega_L) - \alpha_{\hat e\hat e}(-\delta Q;\,\omega_L)}{2\,\delta Q}
   $$

6. **The strength** $R_\nu(\hat e;\omega_L) = \bigl\lvert\partial\alpha_{\hat e\hat e}/\partial Q_\nu\bigr\rvert^2$
   in Å⁴/amu — the real part's change squared plus the imaginary part's —
   ranked against the strongest real mode.

**On H₂** (the proof of concept, [`engines/vibration.md`](?doc=engines/vibration.md)
§ 5.11 § 12): $\alpha$ = 1.32 Å³ static and 1.36 Å³ at 633 nm; $R$ = 23.4
and 25.8 Å⁴/amu — its gap is far from the laser, so the laser moves it
little. In a junction the gold's near-Fermi transitions fall to almost
nothing at 633 nm while the molecule's near 2 eV grow (the table of § 3.1).

**What the laser does not change**: the polarizability is taken at the
laser's frequency for the light going in and coming out — the usual
approximation while the vibration's energy is small beside the laser's
distance from the nearest electronic transition, its width included
[Gillet2013] — and
the measured intensity's factors, the scattered light's
$(\tilde\nu_L-\tilde\nu_\nu)^4$ and the temperature's
$1/\bigl(1-e^{-hc\tilde\nu_\nu/k_BT}\bigr)$ over $\tilde\nu_\nu$, are not
applied: the strength is reported, on both engines — the factors an option
later ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 13,
M4-d).

### 3.3 What the optical calculation is, and is not

- **Independent states.** SIESTA's "simplest approach": transitions between
  the self-consistent states, with no correction for the field the response
  itself creates (no local-field effects). For aromatic molecules such
  uncoupled polarizabilities run about twice the coupled ones, an error
  largely the same for every mode [Apra2020]; coupled density-functional
  Raman activities of small molecules match experiment well [Halls1999].
  Ratios between modes therefore carry more than the absolute values.
- **A difference of large numbers.** The metal's transitions are large; the
  contact's change is kept, the far gold's cancels only where it is computed
  alike at both displacements (§ 3.1), so the derivative is a small difference
  of large numbers; how far it stands above the k-point mesh, the broadening
  and the band count is the person's judgement, with the settings stated
  beside the value.
- **The integral needs the spectrum that weighs at the laser**: every pair
  of states up to the cutoff, `Optical.Energy.Maximum`, over enough bands
  (`Optical.NumberOfBands`) — a pair weighs about one over its energy
  squared, so 3 Ry keeps all the basis describes (§ 3.1, *The cutoff*).

### 3.4 What is reported — the whole cell's response, the held gold's kept *(decided 2026-10-24)*

**The question.** Most of a junction's gold is held. In the AuBDTAu vibration
on Sol (2026-10-24) 20 atoms are free — the 12 of BDT and 8 Au, four at each
contact, the label `VIB_CALC` — and 292 are held. **Held means the nucleus
stays where it is**: no held atom is nudged and no force constant is taken
for it. **It does not mean its electrons are left out**: every step of the
force-constant run solves the electrons of all 312 atoms — the run's output
reads 312 atoms, 5,824 orbitals and 5,740 electrons — which is how the held
gold shapes the free atoms' forces, and so the modes. SIESTA's optical
calculation reads that same whole-cell solution and cannot leave atoms out,
so the polarizability of § 3.1 is the whole cell's, the held gold's electrons
in it.

**A held atom's electrons answer its neighbours — measured.** In the CO₂ run
of § 7 the carbon is held and never moves, yet its charge changes as the
oxygens do: over the run's 13 steps its Hirshfeld charge ranges from 0.2215 to
0.2241 $e$. Along each mode:

| mode | what the oxygens do | the held carbon's charge change ($e$ amu$^{-1/2}$ Å$^{-1}$), Hirshfeld / Voronoi |
|---|---|---|
| 1, 2 — the bends | move off the axis | 0 / 0 |
| 3 — one bond shorter, the other longer | move together along z | 0 / 0.003 |
| 4 — both bonds longer | move apart along z | +0.022 / −0.037 |

Where symmetry forbids a change it reads zero (Voronoi's 0.003 is the mesh's
residue, § 6); where it allows one, the carbon's electrons rearrange though
the carbon stays put. (The two partitions disagree on its sign: how electrons
are assigned to atoms is a convention, § 1a — that they move is not.) **In the
junction the held gold next to the contact answers the anchors the same
way**: charge moves across the Au–S contact and the states there shift. That
is what gives a molecule on a metal surface its own Raman strengths and even
its own peak positions *(the user, 2026-10-24: "molecule next to a metal
surface will have different peak strength or even peak position precisely
because there will be some charge transfer and electronic state changes. most
of the fixed atoms will not feel much but the ones closer to the molecule
will contribute some, and molecule will feel it")*. Gold farther from the
contact barely changes.

**Decided: the whole cell's number** *(the user, 2026-10-24: "i am not
convinced that completely remove fixed Au is the correct solution")* — the
whole cell's $\partial\alpha_{\hat e\hat e}/\partial Q_\nu$, and $R_\nu(\hat e)$ from it:
every electron's response, the molecule's, the free gold's and the held
gold's — the contact's charge transfer and shifted states are the physics,
not background. No share of it on a set of atoms is reported *(the user,
2026-10-28: "I see no point at all to select any atoms")*.

**Rejected: leaving the held gold out of the electrons.** It would drop the
contact's physics above, and where it draws the line between the molecule and
the metal would be arbitrary.

**The infrared** has no whole-cell number — a periodic metal's dipole is
not defined (§ 2.1, § 2.5) — so it reads the charges of the free atoms, the
system the vibration was described for; the charge each mode moves into or
out of them is reported beside the strength (§ 2.1).

**What it does not settle** (§ 3.1): whether the far gold's part cancels in
the whole cell's difference at the settings a person can afford — the
person's check, refining the optical k-mesh and seeing the mode ratios hold.
Published work holds the far metal fixed as this does [Zayak2011,
Frederiksen2007]; it finds the substrate's own response changing with the
mode part of the signal, not noise [Zayak2011, Oren2012], and splits no
periodic cell's derivative into a molecule's share — junction theory
projects the polarizability on the molecule instead [Galperin2008]. Whether $\alpha$ is
taken static or at the laser's frequency is decided: both, the laser's the
one the strength is ranked by (§ 3.1, D-f) — beside a metal, the laser's the
only one defined.

---

## 4. The displacement (workflow step 4)

The frames along a mode sample its motion at a **temperature the person
states** *(the user, 2026-10-28)*: the junction's, or for a mode pumped out
of equilibrium its effective temperature — the one whose thermal state holds
the same mean number of quanta, measured from the line's anti-Stokes to
Stokes ratio (`vibrational-averaging.md` § 2a.1). From it, and from the
vibration result alone, come the mode's mean number of quanta $n$ and its
spread

$$
n = \frac{1}{e^{\hbar\omega/k_BT}-1}, \qquad
\sigma = Q_{\mathrm{zp}}\sqrt{2n+1} = Q_{\mathrm{zp}}\sqrt{\coth\!\left(\frac{\hbar\omega}{2k_BT}\right)},
$$

and the **outermost move** — how far the outermost frames move the atom that
moves most —

$$
d_{\mathrm{out}} = \sqrt2\,x_{\mathrm{out}}\;\sigma\;\max_k\lvert\mathbf L_k\rvert
\qquad (\sqrt3\,\sigma\max_k\lvert\mathbf L_k\rvert \text{ for three frames}),
$$

$x_{\mathrm{out}}$ the rule's outermost node ($\sqrt{3/2}$ for three frames,
2.0202 for five, 2.6520 for seven) and $Q_{\mathrm{zp}}$ the mode's
zero-point amplitude, which every result states (`vibrational-averaging.md`
§ 5). A mode pushed in step with the light swings, and its positions are
not a bell curve; its frames are another rule (`vibrational-averaging.md`
§ 2a.2, § 2a.3).
Raman's $\pm\delta Q$ (§ 3.2) is a different thing: a step for a derivative,
not a motion to average over.

### 4.1 An example — the toy junction's two real modes

From the result of § 2.4's run, computed with the vibration code's own
functions (`spectra.derived`):

| mode | frequency | $Q_{\mathrm{zp}}$ (amu$^{1/2}$ Å) | largest atom's zero-point move | $d_{\mathrm{out}}$, three frames, 4 K / 300 K | $d_{\mathrm{out}}$, five frames, 300 K |
|---|---|---|---|---|---|
| 5 | 2023.77 cm⁻¹ | 0.0913 | 0.0643 Å | 0.111 / 0.111 Å | 0.184 Å |
| 6 | 2548.37 cm⁻¹ | 0.0813 | 0.0573 Å | 0.099 / 0.099 Å | 0.164 Å |

Both modes are stiff — $\hbar\omega$ is about ten times $k_BT$ at room
temperature (0.251 and 0.316 eV against 0.026 eV) —
so their thermal spread is their zero-point spread at 4 K and at 300 K alike
($\sigma_T/Q_{\mathrm{zp}}$ = 1.000 at both). A soft mode, tens of cm⁻¹ — a
contact or a whole-molecule rocking — widens several-fold by room
temperature (`vibrational-averaging.md` § 2).

---

## 5. Why Raman is a phase of the vibration run

molbuilder runs SIESTA in three kinds of calculation today — an
optimization, a vibration and a transport calculation. Raman's step 3 needs
something none of them does: **SIESTA run once at a fixed geometry — a single
point — at each frame of a mode, with the optical calculation in its input.**

| kind | what SIESTA does | the geometry | the optical response it could give |
|---|---|---|---|
| optimization | relaxes the structure | moves, step by step | the final geometry's only |
| vibration | the force-constant run: $1 + 6\,n_{\mathrm{free}}$ nudged geometries | nudged and restored | the last displacement's only |
| transport | electrodes and the device, for a current | fixed | not this question |
| `electronic` (decided, not built) | one self-consistent solution per frame of a frame set | each frame, as given | each frame's |
| **the vibration run's optical phase** (built 2.42.0) | **one single point per frame** | **$R_0$ and $R_0 \pm \delta Q\,\mathbf L$, from the force-constant run's modes** | **each frame's — what Raman needs** |

- **The vibration cannot give it.** Its force-constant run does visit
  displaced geometries, but SIESTA computes the optical response once, after a
  run's last step, for its final geometry — `siesta.F` calls
  `siesta_analysis`, where `optical` is called, once, after the run. One
  force-constant run would give one geometry's response, its last
  displacement's, and not the mode's.

**Decided 2026-10-25** *(the user: "there is no intrinsic difference"; "i
choose B")*: **one vibration run in phases**, PySCF's shape — the
relaxation, the force constants, then the optical phase: one SIESTA single
point per frame, the equilibrium once and $\pm\delta Q$ along each selected
mode, each SIESTA call in its own folder of the run. Prep writes every deck
with every setting, the optical keywords among them (`OpticalCalculation`,
`Optical.PolarizationType polarized`, `Optical.Vector`, `Optical.Mesh`,
`Optical.Broaden`, `Optical.NumberOfBands`, `Optical.EnergyMaximum`, each
checked against SIESTA's manual before it is written); the run writes only
each call's coordinates, which SIESTA reads from a file beside the deck. How
the phases are joined and where every record lies: [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11. The
`electronic` kind (§ 5.10 ④) stays the design for the electronic response
along a mode; Raman does not wait on it.

### 5.1 What it costs — measured on AuBDTAu *(2026-10-24)*

**The reference** is the AuBDTAu vibration run on Sol
(`spectrum/AuBDTAu_relaxed_vib`, 2026-10-24): 312 atoms, 20 free, 64 MPI
ranks, at the vibration's settings — a 1×1×1 k-point grid and a 300 Ry mesh.
Read from the run's own SCF timer log (`…-run0.scf-timing.log`), each step's
time its iterations times the run's average iteration:

| measured | value |
|---|---|
| one SCF iteration, on average | 5.7 s — the run's 2,072 iterations over 3.30 h |
| the first step, from scratch | 37 iterations, about 3.5 min |
| every later step, starting from the step before | 17 iterations on average (11 to 21), about 1.6 min |
| the whole force-constant run | 121 steps, 3.3 h |

**What each part adds:**

| part | SIESTA runs it adds | its cost |
|---|---|---|
| the infrared, every mode | none | read from the vibration's own steps (§ 2.2) |
| keeping the held gold in Raman (§ 3.4) | none | it is in every run already: all 312 atoms' electrons are solved at every step |
| Raman for $N$ chosen modes | $1 + 2N$ — frame 0 once, $\pm\delta Q$ for each mode | each run one junction SCF and SIESTA's optical calculation |

**The calculation for five modes**: $1 + 2\times5 = 11$ runs, each started
from the force-constant run's reference density
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 3): frame 0,
at that same geometry, converges almost at once (one SCF iteration on H₂,
§ 5.11 § 12 there); each $\pm\delta Q$ run, a small displacement from it,
about 1.6 min, as a force-constant step is — $10\times1.6 \approx 16$ min —
or about $11\times3.5 \approx 40$ min if every run started from scratch. So
**about 16–40 min of SCF on 64 ranks, a twelfth to a fifth of the
vibration's 3.3 h**,
at the vibration's settings; a denser k-point grid for the SCF multiplies it.

**Not measured: the optical calculation itself.** After each SCF it
diagonalizes the 5,824-orbital Hamiltonian at every k-point of its own mesh
(`Optical.Mesh`) and couples every occupied state with every empty one; a
metal wants more k-points there than the vibration's single one. Its cost is
measured on one junction frame, on the machine that runs it, before any
batch.


---

## 6. Numerical choices, and their checks

Every choice is stated beside the values it gave; the checks below cost
nothing beyond reading what the runs wrote.

| choice | proposed | check |
|---|---|---|
| $\delta$ (`FC.Displacement`) | the run's own, 0.0212 Å by default | where the person runs a displacement sweep (`engines/vibration.md` § 5.9), the charge derivatives are read at each of its steps, as the force constants are |
| the charges' grain | Hirshfeld or Voronoi charges are integrated on SIESTA's real-space mesh, so an atom moved by $\delta$ against the mesh changes its charge a little even with no physics behind it | the same sweep, where it is run |
| `MeshCutoff` | the vibration's own: its force constants and its charges are both taken on the mesh | a coarse mesh tells two equivalent atoms apart in the force constants, so their modes mix and a symmetry-forbidden mode borrows strength — measured on CO₂, the forbidden stretch at 0.5 % of the other at 150 Ry and $6.5\times10^{-8}$ at 500 Ry (§ 7) |
| $\delta Q$ for Raman | $\delta/\max_k\lvert\mathbf L_k^{(\nu)}\rvert$ — the largest atom moved by $\delta$, the force constants' own linear range | — |
| `Optical.Mesh`, `Optical.Broaden`, `Optical.NumberOfBands`, `Optical.Energy.Maximum` | the person's, stated with every value; a broadening above zero | the mesh: the ratios between the modes hold when it is refined [Gillet2013]; the broadening: stated beside every value, 0.1–0.2 eV in published work and at least the spacing between discrete transitions [Gillet2013]; the bands: the f-sum in the `.EPSIMG` header near 1 for a molecule when the range reaches the basis's top; the range: how far the Raman derivative moves when the cutoff is raised — 0.34 % or less from 3 to 10 Ry on our runs, the f-sum reading lower under a cutoff by design (§ 3.1, *The cutoff*) |
| the laser's wavelength | the person's, entered after the run | none needed: a parameter of the integral, not of the runs |
| the charge sum | — | at every step the charge column over all atoms sums to zero (measured, § 7) |
| symmetry | — | a mode the structure's symmetry forbids reads zero, the dipole taken about a centre on the symmetry element (as § 2.4's mode 6 does) |

---

## 7. What is measured, and what is not yet

**Measured** (2026-10-24, the toy junction `claude-transport-walk`: a chain
of 22 hydrogen atoms, its middle two free) — **the charges at every
force-constant step**: the vibration `frequency/h-vib-q`, through jobset, its
input asking for both partitions (2.35.2): 13 steps, 13 Hirshfeld and 13
Voronoi blocks, every atom in each, each summing to zero within
1.2 × 10⁻⁵ e. A free atom moved off the chain's axis (along x or y) leaves
the charges unchanged both ways; moved along it (along z), it changes them
with opposite signs — as the
chain's symmetry asks. Molbuilder's reader reads the output without a
warning, and the modes are the earlier run's to the last digit. **From the same
run, the infrared strengths of § 2.4**: the symmetry-forbidden mode reads
zero under both partitions, the allowed one does not — 0.1 and 47.2 km/mol
for the free pair, the built function's numbers.

**CO₂ with its carbon held, through jobset** (the SIESTA vibration e2e,
`tests/test_siesta_vibration_e2e.py`, 2.36.0): on the relaxed molecule and on
the unrelaxed one alike, under both partitions, a field along z sees neither
bend, a field along x neither stretch; each stretch's dipole change along z
opposes the oxygens' net motion — their charge is negative under both
partitions — and no bend moves charge into or out of the free oxygens
(below $10^{-4}\,e$ amu$^{-1/2}$), while the stretch whose oxygens move
apart moves it between them and the held carbon: −0.0218 (Hirshfeld) and
+0.0334 (Voronoi) $e$ amu$^{-1/2}$ at 1312 cm⁻¹ on the relaxed molecule —
the two partitions disagreeing even in sign. **The stretch whose oxygens move apart
reads 0.5 % of the other** on the relaxed molecule (1.13 against
215.3 km/mol, Hirshfeld), not zero: the computed modes themselves are not
quite the symmetric pair — the oxygens' z components (−0.164, 0.189) give an
in-phase share whose square, 0.0050, is the measured ratio. With the carbon
held the two stretches lie close (1250 and 1312 cm⁻¹), so the slightest
difference between the two oxygens mixes them. **The mesh is what made the
difference**: relaxing to 0.001 eV/Å instead of 0.01 changes nothing (the
relaxation had reached 0.0009 eV/Å either way), while raising `MeshCutoff`
from 150 to 500 Ry (`frequency/co2-tight-500`, the same tolerance) gives the
symmetric pair — oxygen z components 0.17675 and 0.17682 — and the stretch
whose oxygens move apart then reads $6.5\times10^{-8}$ of the other under
Hirshfeld (216.5 km/mol against $1.4\times10^{-5}$) and $1.2\times10^{-5}$
under Voronoi. The same cutoff moves the bend from 281 to 325 cm⁻¹ and the
stretches by 6 cm⁻¹. A residue of this kind is the calculation's mesh,
read beside the values, never corrected (§ 6, the charges' grain).

**CO₂'s Raman at 633 nm, through jobset** (the same e2e, 2.44.0, at its
150 Ry mesh): light along z, the stretch whose oxygens move apart reads
$R(\hat z)$ = 45.1 Å⁴/amu and the one whose oxygens move together 0.23 —
0.5 % of it, the in-phase share of the mixed pair above again: a centre of
symmetry's mutual exclusion, the infrared's mirror, up to the same mesh
residue.

**Measured 2026-10-25 — the route of § 3, end to end on H₂** (a 10 Å
box, SIESTA 5.4.2, by hand outside jobset: the proof of concept of
[`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 12). The relaxation, the force-constant run
and three optical single points — the equilibrium and $\pm\delta Q$ along
the one mode, 4271 cm⁻¹ — in 46 s on two ranks; $\alpha_{zz}(0)$ = 1.32 Å³
and 1.36 Å³ at 633 nm (the molecule's measured 1.0 Å³ along the bond: the
independent-state picture of § 3.3, no local field);
$\partial\alpha_{zz}/\partial Q$ = 4.84 Å² amu$^{-1/2}$ static, 5.08 at
633 nm, so $R(\hat z)$ = 23.4 and 25.8 Å⁴/amu — non-zero for the one mode of
H₂, Raman-active and infrared-silent. The broadening composes exactly after
the run (§ 3.2a, step 2; H₂'s gap is far above zero): the 0.001 Ry spectrum convolved to 0.01 and 0.1 Ry gives the
direct runs' $\alpha$ to four decimals, and the width moves $\alpha$ by
0.6 % between 0.0136 and 1.36 eV.

**Measured 2026-10-27 — a metal's own atoms moving** (the hydrogen chain of
§ 2.4, its two free atoms links of the chain, SIESTA SZ, 150 Ry, the SCF at
Γ, through jobset: `describe` from the Spectrum page, Task setup, `prep`,
`launch`, read through `molbuilder.postprocess.raman_view`):

| | optical mesh Γ | 1×1×8 |
|---|---|---|
| frame 0: largest $\varepsilon_2$ · f-sum · $\alpha$ static | $4.5\times10^{-9}$ · 0.0 · $2\times10^{-9}$ Å³ | $9.5\times10^{-10}$ · 0.0 · $4\times10^{-9}$ Å³ |
| the 2548 cm⁻¹ stretch: $\alpha(+\delta Q)$ / $\alpha(-\delta Q)$, static | 0.366 / 0.363 Å³ | 75,600 / 75,900 Å³ |
| the same at 633 nm | 0.535 / 0.531 Å³ | −214 / −214 Å³ |
| static $\alpha$, the width 0.027 → 0.1 eV | +0.04 % | +48 % |
| $\alpha$ at 633 nm, the same | +0.2 % | +3 % |

One orbital per atom makes one band: the straight chain has no interband
pair, and its plasma frequency counted no state within the broadening of the
Fermi level (0.0). A displacement either way opens a gap at the Fermi level
(§ 3.1) — the denser mesh samples it, from 0.003 eV — so $\alpha$ jumps from
nothing to the same value at both sides, and the four bends are imaginary
(−666, −481 cm⁻¹): the straight chain is unstable sideways. What it shows is
the route's limit, not a junction's number: a free atom that is part of the
metal, and the static value beside a metal (§ 3.1).

**Not yet shown — the comparison owed** (the infrared built in 2.36.0 and
Raman in 2.44.0 without it), the person's to run on their machine: on a
molecule whose
answer is known independently, SIESTA's infrared (§ 2) and Raman (§ 3)
against PySCF's, which molbuilder computes already (`engines/vibration.md`
§ 4): water in a box, the same modes, compared mode by mode and in rank. The
infrared half needs only the two vibrations that exist; PySCF's
`ir_intensity_km_mol` is the tumbling molecule's, so it is compared with the
mean of SIESTA's three axes, $\tfrac13[A_\nu(\hat x)+A_\nu(\hat y)+A_\nu(\hat z)]$
(§ 2.3), the whole molecule being its free atoms. The Raman half needs the Raman runs (§ 5) and a comparison that is like for
like: PySCF stores the
orientation-averaged activity, $45a'^2+7\gamma'^2$, from coupled-perturbed
theory, while $R_\nu(\hat e)$ is one component from independent states —
what it is compared with is decided with the Raman route (§ 3.1). **What
water cannot prove**: SIESTA takes the position operator only for a cell it
judges a molecule, sampled at Γ alone, and the momentum formulation for a
junction (§ 3.1), so the comparison checks the chain from the response to
$R_\nu$, not the junction's formulation.

---

## 8. What the method assumes

1. **Point charges** (infrared): the free atoms' dipole is their partial
   charges at their positions; each atom's own dipole is not in it, and two
   partitions give two answers.
2. **Independent states** (Raman): SIESTA's optical calculation, without
   local-field effects (§ 3.3).
3. **The metal** (Raman): no free-electron (Drude) term is added, so the
   static value beside a metal is not defined and the laser's is the one to
   read; the contact gold's response is kept in the whole cell's number; the far gold cancels between the frames
   only where it is computed alike at both; a free atom that is part of the
   metal can make $\alpha$ jump with the displacement (§ 3.1).
4. **The polarizability at the laser's frequency** (Raman), from
   independent states — pre-resonance included, the laser's own; the static
   value beside it, the textbook activity, away from every resonance.
5. **Harmonic modes**, and derivatives taken in their linear range.
6. **Light in vacuum**: the field at the molecule is the incident light's,
   along ê; a metal's enhancement or screening of it is not in the number
   (§ 0).

---

## 9. Words used here

| word | what it means here |
|---|---|
| **mode** | one pattern in which the free atoms move together, with one frequency (`engines/vibration.md` § 6.3) |
| **force-constant run** | SIESTA nudging each free atom both ways along x, y and z and recording the forces — the vibration's own SIESTA run |
| **partial charge** | the charge a partition gives one atom: its valence charge less the electrons assigned to it (§ 1a) |
| **dipole change** | how the molecule's charge distribution shifts as it moves along a mode — what infrared sees (§ 2) |
| **polarizability** | how easily light's field shifts the electrons — what Raman sees through its change (§ 3) |
| **dielectric function** $\varepsilon$ | the material's response to light; its imaginary part $\varepsilon_2$ is the absorption, its real part $\varepsilon_1$ at zero frequency gives the static polarizability |
| **the optical calculation** | SIESTA's `OpticalCalculation`: $\varepsilon_2$ from the couplings between its occupied and empty states (§ 3.1) |
| **static** | at zero frequency — away from the molecule's electronic resonances |
| **frame set** | one mode written as a few geometries along it, each its own frame (`engines/vibration.md` § 5.10 ③) |
| **single point** | one SIESTA solution at one fixed geometry, nothing moved |
| **the `electronic` kind** | the calculation kind that runs a single point at each frame of a frame set — decided, not built; not Raman's route (§ 5) |
| **the optical phase** | the SIESTA vibration run's phase after its force constants: one single point with the optical calculation per frame — the equilibrium and $\pm\delta Q$ along each selected mode — each a SIESTA call in its own folder of the run ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11) |

---

## References

Each checked against Crossref, 2026-10-24, and in `references.bib`.

- [Soler2002] J. M. Soler, E. Artacho, J. D. Gale, A. García, J. Junquera,
  P. Ordejón, D. Sánchez-Portal, *J. Phys.: Condens. Matter* **14**,
  2745–2779 (2002), doi:10.1088/0953-8984/14/11/302 — the SIESTA method.
- [Garcia2020] A. García, N. Papior, A. Akhtar, E. Artacho, V. Blum,
  E. Bosoni, P. Brandimarte, M. Brandbyge, J. I. Cerdá, *et al.*, *J. Chem.
  Phys.* **152**, 204108 (2020), doi:10.1063/5.0005077 — SIESTA's recent
  developments and applications.
- [Hirshfeld1977] F. L. Hirshfeld, *Theor. Chim. Acta* **44**, 129–138
  (1977), doi:10.1007/BF00549096 — bonded-atom (Hirshfeld) partial charges.
- [FonsecaGuerra2004] C. Fonseca Guerra, J.-W. Handgraaf, E. J. Baerends,
  F. M. Bickelhaupt, *J. Comput. Chem.* **25**, 189–210 (2004),
  doi:10.1002/jcc.10351 — Voronoi deformation density charges, assessed
  against Mulliken, Bader, Hirshfeld and Weinhold.
- [KingSmith1993] R. D. King-Smith, D. Vanderbilt, *Phys. Rev. B* **47**,
  1651–1654 (1993), doi:10.1103/PhysRevB.47.1651 — the Berry-phase
  polarization of an insulator.
- [Porezag1996] D. Porezag, M. R. Pederson, *Phys. Rev. B* **54**, 7830–7836
  (1996), doi:10.1103/PhysRevB.54.7830 — infrared intensities and Raman
  activities in density-functional theory.
- [Gillet2013] Y. Gillet, M. Giantomassi, X. Gonze, *Phys. Rev. B* **88**,
  094305 (2013), doi:10.1103/PhysRevB.88.094305 — the Raman tensor at a
  laser: the complex susceptibility's change, its squared modulus; the
  broadening at least the spacing between transitions; a Raman derivative's
  slow k-point convergence.

Added 2026-10-27, each checked against Crossref and in `references.bib`:

- [Zayak2011] A. T. Zayak, Y. S. Hu, H. Choo, J. Bokor, S. Cabrini,
  P. J. Schuck, J. B. Neaton, *Phys. Rev. Lett.* **106**, 083003 (2011),
  doi:10.1103/PhysRevLett.106.083003 — chemical Raman enhancement on a
  periodic Au(111) slab; the far gold infinitely heavy.
- [Prandini2019] G. Prandini, G.-M. Rignanese, N. Marzari, *npj Comput.
  Mater.* **5**, 129 (2019), doi:10.1038/s41524-019-0266-0 — a metal's
  dielectric function, its intraband term and k-meshes.
- [Jensen2005a] L. Jensen, J. Autschbach, G. C. Schatz, *J. Chem. Phys.*
  **122**, 224115 (2005), doi:10.1063/1.1929740 — the polarizability with a
  finite lifetime.
- [Jensen2005b] L. Jensen, L. L. Zhao, J. Autschbach, G. C. Schatz, *J. Chem.
  Phys.* **123**, 174110 (2005), doi:10.1063/1.2046670 — Raman from
  derivatives of the frequency-dependent polarizability.
- [Saikin2009] S. K. Saikin, R. Olivares-Amaya, D. Rappoport, M. Stopa,
  A. Aspuru-Guzik, *Phys. Chem. Chem. Phys.* **11**, 9401 (2009),
  doi:10.1039/b906885f — benzenethiol on silver clusters; the damping's role.
- [Zhao2006] L. L. Zhao, L. Jensen, G. C. Schatz, *J. Am. Chem. Soc.* **128**,
  2911–2919 (2006), doi:10.1021/ja0556326 — pyridine–Ag₂₀; chemical,
  charge-transfer and field enhancement.
- [Aikens2006] C. M. Aikens, G. C. Schatz, *J. Phys. Chem. A* **110**,
  13317–13324 (2006), doi:10.1021/jp065206m — pyridine with Au₂₀.
- [Galperin2008] M. Galperin, S. Tretiak, *J. Chem. Phys.* **128**, 124705
  (2008), doi:10.1063/1.2876011 — a junction's optical response projected
  on the molecule.
- [Oren2012] M. Oren, M. Galperin, A. Nitzan, *Phys. Rev. B* **85**, 115435
  (2012), doi:10.1103/PhysRevB.85.115435 — Raman of a junction by charge
  transfer.
- [Ward2007] D. R. Ward, N. K. Grady, C. S. Levin, N. J. Halas, Y. Wu,
  P. Nordlander, D. Natelson, *Nano Lett.* **7**, 1396–1400 (2007),
  doi:10.1021/nl070625w — measured Raman in electromigrated gaps.
- [Taghizadeh2020] A. Taghizadeh, U. Leffers, T. G. Pedersen, K. S. Thygesen,
  *Nat. Commun.* **11**, 3011 (2020), doi:10.1038/s41467-020-16529-6 — Raman
  from the frequency-dependent dielectric function's finite differences.
- [Apra2020] E. Aprà, A. Bhattarai, E. Baxter, S. Wang, G. E. Johnson,
  N. Govind, P. Z. El-Khoury, *Appl. Spectrosc.* **74**, 1350–1357 (2020),
  doi:10.1177/0003702820923392 — uncoupled against coupled polarizabilities.
- [Halls1999] M. D. Halls, H. B. Schlegel, *J. Chem. Phys.* **111**,
  8819–8824 (1999), doi:10.1063/1.480228 — Raman intensities by method.
- [Heeger1988] A. J. Heeger, S. Kivelson, J. R. Schrieffer, W.-P. Su, *Rev.
  Mod. Phys.* **60**, 781–850 (1988), doi:10.1103/RevModPhys.60.781 — a
  half-filled chain's dimerization.
- [Frederiksen2007] T. Frederiksen, M. Paulsson, M. Brandbyge, A.-P. Jauho,
  *Phys. Rev. B* **75**, 205413 (2007), doi:10.1103/PhysRevB.75.205413 — a
  junction's vibrations in a dynamic region, the rest fixed.
- [SiestaOptical2021] the SIESTA developers, *Calculation of optical
  properties*, SIESTA school 2021 tutorial,
  docs.siesta-project.org/projects/siesta/en/school-2021/tutorials/optical-properties.
- [Greenler1966] R. G. Greenler, *J. Chem. Phys.* **44**, 310–315 (1966),
  doi:10.1063/1.1726462 — the infrared of adsorbed molecules on metal
  surfaces by reflection.
- [Hoffmann1983] F. M. Hoffmann, *Surf. Sci. Rep.* **3**, 107 (1983),
  doi:10.1016/0167-5729(83)90001-8 — infrared reflection–absorption
  spectroscopy of adsorbed molecules.
- [Zaitsev2018] N. L. Zaitsev, P. Jakob, R. Tonner, *J. Phys.: Condens.
  Matter* **30**, 354001 (2018), doi:10.1088/1361-648X/aad576 — the
  vibrations of PTCDA on Ag(111).
- [Zhou2021] X. Zhou, H. Vejayan, R. D. Beck, H. Guo, B. Jiang, *J. Phys.
  Chem. Lett.* **12**, 11164–11169 (2021), doi:10.1021/acs.jpclett.1c03342
  — the infrared activities of species adsorbed on metal surfaces.
- [Bengtsson1999] L. Bengtsson, *Phys. Rev. B* **59**, 12301–12304 (1999),
  doi:10.1103/PhysRevB.59.12301 — the dipole correction for surface
  supercells.
- [Resta1994] R. Resta, *Rev. Mod. Phys.* **66**, 899–915 (1994),
  doi:10.1103/RevModPhys.66.899 — the macroscopic polarization of
  crystalline dielectrics by the geometric phase.
- [RestaVanderbilt2007] R. Resta, D. Vanderbilt, in *Physics of
  Ferroelectrics: A Modern Perspective*, Topics in Applied Physics **105**,
  31–68 (2007), doi:10.1007/978-3-540-34591-6_2 — the modern theory of
  polarization.
- [Shockley1938] W. Shockley, *J. Appl. Phys.* **9**, 635–636 (1938),
  doi:10.1063/1.1710367 — currents induced in conductors by a moving
  charge.
- [Ramo1939] S. Ramo, *Proc. IRE* **27**, 584–585 (1939),
  doi:10.1109/JRPROC.1939.228757 — currents induced by electron motion.
- [Ward2008] D. R. Ward, N. J. Halas, J. W. Ciszek, J. M. Tour, Y. Wu,
  P. Nordlander, D. Natelson, *Nano Lett.* **8**, 919–924 (2008),
  doi:10.1021/nl073346h — conduction and Raman measured together in
  molecular junctions.
- [TroisiRatner2006] A. Troisi, M. A. Ratner, *Nano Lett.* **6**,
  1784–1788 (2006), doi:10.1021/nl0609394 — propensity rules for inelastic
  electron tunnelling spectra.
- [Paulsson2008] M. Paulsson, T. Frederiksen, H. Ueba, N. Lorente,
  M. Brandbyge, *Phys. Rev. Lett.* **100**, 226604 (2008),
  doi:10.1103/PhysRevLett.100.226604 — inelastic propensity rules for
  nanoscale junctions, unified.
- [HaidukeBruns2005] R. L. A. Haiduke, R. E. Bruns, *J. Phys. Chem. A*
  **109**, 2680–2688 (2005), doi:10.1021/jp045357u — a dipole derivative
  decomposed into charge, charge-flux and dipole-flux terms.
- [VesterOlsen2024] J. Vester, J. M. H. Olsen, *J. Chem. Theory Comput.*
  **20**, 9533–9546 (2024), doi:10.1021/acs.jctc.4c00882 — the partial
  Hessian approximation assessed in vibrational analysis.
- SIESTA 5.4.2 manual, `OpticalCalculation`, `Optical.*`, `Charge.Hirshfeld`,
  `BornCharge` and `PolarizationGrids`; its source `Src/optical.F`,
  `Src/partial_charges.F90`, `Src/grdsam.F`, `Src/siesta_analysis.F90`,
  `Src/siesta.F`; `Util/Optical/README`.
