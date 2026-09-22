# Normal modes — what a vibrational calculation counts, and what it must remove

**Role:** contract
**Domain:** science
**Companions:** [`overview.md`](?doc=science/overview.md) (the science domain's
promise), [`validation.md`](?doc=science/validation.md) (the runtime advisory
machinery that carries these rules to the user),
[`web/spectra.md`](?doc=web/spectra.md) § 8 (what freezing an atom means to the
Spectrum tab), [`engines/pyscf.md`](?doc=engines/pyscf.md) (the deck that
computes it).

This document owns **one fact**: *how many of the numbers a vibrational
calculation produces are vibrations, and how the rest are identified and
removed.* Every place in the codebase that counts modes, warns about modes, or
decides what goes into a spectrum or a thermochemistry sum reads its rule from
here. Before 2026-09-21 that fact had no home, was re-derived in three places,
and was wrong in all three (§ 8).

> **Vocabulary.** *Hessian* — the table of second derivatives of the energy with
> respect to moving each atom in each direction; its eigenvalues give the
> squared vibrational frequencies. *Stationary point* — a geometry where the
> forces on the atoms vanish, which is the only place a harmonic frequency means
> anything. *Mass-weighted* — each entry of the Hessian divided by the square
> roots of the two atoms' masses, which is what turns curvature into frequency.
> Wider terms (*SCF*, *DFT*, *CPHF*) are in the
> [`overview.md` glossary](?doc=science/overview.md).

---

## 1. The problem, in one molecule

Take 1,4-benzenedithiol (BDT): a benzene ring with an `–SH` group at each end,
14 atoms. Hold the two sulfur atoms still and ask for the vibrational spectrum —
the ordinary way to study a molecule bonded to a surface, where the anchor atoms
belong to the metal and are not free to move.

The calculation returns **36 numbers**. Thirty-five of them are vibrations.
The thirty-sixth is this:

> the whole ring turning, as one rigid body, about the line through the two
> sulfur atoms.

Nothing stretches, nothing bends, no bond changes length. The sulfurs sit *on*
that line, so holding them still does not prevent the turn. It costs no energy,
it is not a vibration, and it must not appear in a spectrum or in a
thermochemistry sum.

Measured (RHF/STO-3G, at the relaxed geometry): that mode comes out at
**−0.93 cm⁻¹** and accounts for **100.0 %** of the turning motion, while the
other 35 modes account for 0.0 % between them. It is one clean, spurious entry.

**The rule this document states** is what decides that the number is one — not
zero, not six.

---

## 2. Background — why a free molecule has 3N − 6 vibrations

An N-atom molecule has **3N** coordinates: an x, y and z for every atom.

Not all of them are vibrations, because some ways of moving the atoms do not
change the energy at all. Slide the whole molecule a centimetre to the left and
nothing about it is different — the energy is identical. Turn the whole molecule
around and, again, nothing is different. There are six such motions:

| motion | how many | why the energy does not change |
|---|---|---|
| translation | 3 (x, y, z) | empty space looks the same everywhere |
| rotation | 3 (about x, y, z) | empty space looks the same in every direction |

Each of these is a direction in which the energy has **zero curvature**, so the
mass-weighted Hessian has six zero eigenvalues. What is left is the vibrations:

```text
    n_vib  =  3N - 6
```

**The linear exception.** A straight molecule — CO₂, HCN, any diatomic — has
only two rotations that do anything. Spinning it about its own long axis moves
no atom at all, so that "motion" is not a motion; the direction is the zero
vector rather than a zero-curvature direction. Such a molecule has

```text
    n_vib  =  3N - 5
```

This is the standard treatment [Wilson1955]; the separation of vibration from
rotation is due to Eckart [Eckart1935].

**How it is done in practice.** You do not count — you *remove*. Write the six
motions out as explicit vectors at the current geometry, project them out of the
Hessian, and diagonalise what remains. PySCF's `harmonic_analysis` does exactly
this, and molbuilder's all-atoms-free path calls it:

```python
TR = _get_TR(mass, atom_coords)            # the six motions, from coordinates
q, r  = numpy.linalg.qr(TRspace.T)
P     = numpy.eye(natm*3) - q.dot(q.T)     # project them out ...
h     = reduce(numpy.dot, (bvec.T, h, bvec))
force_const_au, mode = numpy.linalg.eigh(h)   # ... THEN diagonalise
```

Two things to notice, because both matter later. The six vectors are built from
**coordinates alone** — no chemistry, no bond list, nothing a user supplies. And
the choice between 5 and 6 is made **numerically**, from the molecule's moments
of inertia, not from a flag.

---

## 3. The rule when some atoms are held still

Freezing atoms changes the question, and the usual answer `3N − 6` stops
applying. Call the held atoms **F** and the free atoms **A**. The calculation
now builds the Hessian block for the free atoms only and diagonalises that — a
**partial Hessian**, introduced by Head for adsorbates on surfaces [Head1997]
and named by Li & Jensen [LiJensen2002].

**Which method this is, among several.** Freezing part of a system is a family,
not one technique, and the members differ in what they do with the held part
[Tao2021]:

| method | what the held part is |
|---|---|
| **partial Hessian (PHVA)** — *what molbuilder does* | infinitely heavy: it does not move at all [Head1997, LiJensen2002, Besley2008] |
| mobile block Hessian (MBH) | a **rigid block of finite mass**, free to translate and rotate as a unit [Ghysels2007] |
| vibrational subsystem analysis (VSA) | its mass effect is averaged in adiabatically |
| generalised subsystem vibrational analysis (GSVA) | the subsystem's intrinsic modes, environment folded in [Tao2021] |

Naming this matters because the count in this section is **PHVA's count**. MBH
lets the held block move, so its bookkeeping is different, and a rule copied
between the two would be wrong in one of them.

**And the block that is sliced is the block of the full energy.** Besley &
Bryan put it exactly: the derivatives *"are computed for the adsorbate-molecule
system, and hence the interaction with the surface is included explicitly"*
[Besley2008]. The free atoms feel the held ones; what is dropped is only how
held-atom *motion* would couple back, and they do not move. So `H_AA` below is
the free–free block of the true Hessian, not of some isolated fragment.

**The honest cost of the approximation**, stated once so no one has to discover
it: every method in the table above shares *"the unphysical partitioning of the
full Hessian matrix, which causes information loss about the interaction between
the subsystem and its environment"* [Tao2021]. Freezing buys a smaller problem
and pays for it in accuracy. This document governs the **counting**, which must
be right whatever one thinks of the approximation.

There are `3N_A` numbers.

How many are not vibrations? The reasoning is three lines.

**Step 1 — which whole-body motions are still available?** Only those that leave
every held atom exactly where it is. Sliding the molecule sideways moves the
held sulfurs, so it is not available. Turning it about the line through both
sulfurs does not move them, so it is.

**Step 2 — such a motion is still a zero of the partial Hessian.** Write the
motion as a displacement `u` over all `3N` coordinates. Because it leaves the
held atoms alone, `u_F = 0`, and the energy change reduces cleanly:

```text
    uᵀ H u  =  u_Aᵀ H_AA u_A  +  2 u_Aᵀ H_AF u_F  +  u_Fᵀ H_FF u_F
                                          ↑                ↑
                                       both zero, because u_F = 0

            =  u_Aᵀ H_AA u_A
```

The left side is zero because the whole-body motion costs no energy. So the
right side is zero too: `u_A` is a zero mode of exactly the block the
calculation diagonalises.

**Step 3 — count them.** The answer depends only on *where the held atoms are*,
not on how many there are:

| held atoms | whole-body motions that leave them all in place | zero modes |
|---|---|---|
| none | all six (five if the molecule is straight) | 6 (or 5) |
| one | every rotation about that atom | **3** |
| two, distinct | rotation about the line through them | **1** |
| three or more, **all on one line** | rotation about that line | **1** |
| three or more, not all on one line | none | **0** |

> **Every "1" in that table is really "1, unless the surviving turn moves no
> free atom."** Hold both oxygens of CO₂: the table says one leftover, but the
> only free atom — the carbon — sits *on* the O···O line, so the turn moves
> nothing and the answer is **0**. Hold both carbons of acetylene and the same
> happens to all four atoms. Neither case is exotic, and a reader applying the
> table by hand gets both wrong. **The rank of § 3.1 gets them right without
> being told about them**, which is the whole argument for computing rather
> than tabulating.

So the vibration count is

```text
    n_vib  =  3·N_A  -  n_rigid(F)
```

and the free-molecule formula is the same statement with no atoms held: `F` is
empty, `n_rigid = 6` (or 5), and `3N − 6` falls out. **One rule, both cases** —
which is the point, because two rules are two things that can drift apart.

### 3.1 The count, written so it cannot be got wrong

The table above is a consequence, not the rule. Writing the table into code
means writing five branches, and five branches are five chances to be wrong —
which is how the codebase came to hold three different wrong answers (§ 8). The
rule is a rank, and it has no branches:

> Let **G** be the matrix whose columns are the whole-body motions **this
> system's energy is invariant under** (§ 3.1a decides which): three
> translations — the same unit vector repeated on every atom — and, for an
> isolated system only, three rotations, where the rotation column `a`, on atom
> `i`, is
>
> ```text
>     g_rot[a] |_i  =  ê_a × (R_i - R_c)
> ```
> Let **G_F** be the rows belonging to held atoms and **G_A** the rows belonging
> to free atoms. Then
> ```text
>     n_rigid(F)  =  rank( G_A · null(G_F) )
> ```

### 3.1a Which motions the energy is invariant under — periodic vs isolated

The six of § 2 are six only for a system floating in empty space. A **periodic**
calculation — a slab repeating sideways forever, which is how a metal surface is
modelled — has fewer:

- **Translation still costs nothing.** Move every atom by the same vector and the
  crystal is the same crystal, shifted. *(These are the three acoustic modes.)*
- **Rotation costs something.** The repeating box does not turn with the atoms.
  Rotate the contents and they no longer meet their neighbours in the next box
  the same way — a different structure, a different energy. There is no
  rotational invariance to exploit, and **no rotational null vectors to remove**.

So `G` has **six columns for an isolated system and three for a periodic one**,
and the table of § 3.2 gains two rows:

| the system | held atoms | `n_rigid` |
|---|---|---|
| periodic | none | **3** |
| periodic | **any at all** | **0** — translation would move a held atom |

**The junction case falls out as zero.** A slab with its lower layers held is the
second row: nothing survives, nothing is removed, and the spurious-mode problem
this document exists for does not arise there. It is a **molecular** problem.
That is not a reason to skip the rule for periodic systems — it is the rule
telling you, from the geometry, that there is nothing to do.

**Where the answer comes from:** the structure's own per-axis periodicity, which
molbuilder already records. It is not an engine property and must not be
inferred from which engine was picked — a molecule computed in a periodic box
and a molecule computed in free space are different models of the same molecule,
and the box is the thing that decides.

> **Do not "restore" the rotations for a molecule in a large periodic box.**
> They are near-symmetries there, broken by the box, so the corresponding
> directions are *not* exact null vectors. Removing them would delete
> directions that carry real, if small, energy — the very error § 3.1's
> caution warns about, arriving from the opposite side.

In words: first keep only the combinations of the six that do not move any held
atom (that is `null(G_F)`); then ask how many of those actually move a free atom
(that is the rank).

Check it against every row of the table:

| case | `rank(G_F)` | `dim null(G_F)` = 6 − rank | `n_rigid = rank(G_A · null)` |
|---|---|---|---|
| none held, ordinary molecule | 0 | 6 | **6** |
| none held, straight molecule | 0 | 6 | **5** — the axial rotation column is identically zero |
| none held, a lone atom | 0 | 6 | **3** — all three rotation columns vanish, so 3·1 − 3 = 0 vibrations |
| one held | 3 | 3 | **3** — or 2, if every atom lies on one line through it |
| two held, every free atom **on** their axis | 5 | 1 | **0** — the surviving turn moves nothing (CO₂ with both O held; acetylene with both C held) |
| two held | 5 | 1 | **1** |
| ≥3 collinear | 5 | 1 | **1** |
| ≥3 non-collinear | 6 | 0 | **0** |

Why `rank(G_F) = 5` for two held atoms: their three translation rows give 3, and
subtracting one atom's rotation rows from the other's leaves a `(R₁ − R₂) ×`
block, which has rank 2 — never 3, because a cross-product with a fixed vector
annihilates that vector. Adding a third held atom off the line contributes the
missing direction and takes the rank to 6.

The straight-molecule case, the lone-atom case and the collinear-anchor case all
fall out of the same rank. Nothing is special-cased, so nothing can be
special-cased *wrongly*.

**The origin `R_c` does not matter.** A rotation about a different origin differs
from one about `R_c` by a translation, and the translations are already columns
of `G`. The span is the same, so the rank is the same. Use the centre of mass,
as PySCF does, and the choice stops being a choice.

**What the rank must NOT remove — and why a blanket projection is wrong.**
There is a published warning against exactly the naive version of this fix, and
it is worth stating because the rank rule is what keeps us on the right side of
it. Vester & Olsen [Vester2024] studied partial Hessians for a molecule
surrounded by *frozen solvent* and found modes that resemble the molecule
translating and rotating as a whole — they call them **pseudotranslational and
pseudorotational**. Their conclusion is blunt: projecting out translation and
rotation, as one does for an isolated molecule, is *"invalid within the PHVA
approximation"*, because it *"can adversely affect other normal modes."*

**They are right, and the rank rule already agrees with them.** Their frozen
solvent is many atoms, not on one line, so `rank(G_F) = 6`, `null(G_F)` is empty
and `n_rigid = 0` — **nothing is projected**. The rank draws exactly the physical
distinction:

| the motion | does it move a held atom? | energy along it | what it is | what to do |
|---|---|---|---|---|
| ring turning about the S···S line, both S held | **no** | exactly unchanged | not a vibration | **project it out** |
| molecule sliding against a held slab or solvent shell | **yes** | changes, if only weakly | a real, hindered vibration | **keep it** |

The second row is a genuine mode — the frustrated translation of an adsorbate is
measurable — and a rule that removed it would be deleting physics. Only a motion
that leaves *every* held atom exactly in place has exactly zero energy along it,
and only those does `n_rigid` count.

Two further cautions from the same study, both aimed at § 5's instinct to look
at the answer: their pseudo-modes are *"low-frequency but not necessarily the
lowest ones"*, and *"more than six modes may have pseudotranslational and/or
pseudorotational character."* A soft mode is not evidence of a spurious one.

Note also that their reason for removal is different from ours, and both can
apply at once. We remove a motion because it is **not a vibration at all** — an
exact symmetry of the energy. They remove theirs because the approximation
**describes them badly**: the frozen environment cannot respond, so a collective
motion against it is not to be trusted. The first is a correctness rule and
belongs in the calculation; the second is a judgement about accuracy and belongs
to the person reading the spectrum.

> **What is cited here, and what is not.** The published work sets up every
> piece of this: the method [Head1997, LiJensen2002], the block that is sliced
> [Besley2008], the fact that only three zeros survive off a stationary point
> [Ghysels2008], and the warning against blanket projection [Vester2024]. The
> **rank rule of § 3.1 is our own derivation**, and it is written down here
> because the literature does not address the corner molbuilder runs into.
> Published PHVA freezes a *surface* or a *solvent* — many atoms, not on one
> line — where `n_rigid = 0` and the question never arises. Freezing one or two
> atoms of a molecule, which the Spectrum tab lets anyone do in two clicks, is
> the case nobody wrote down. The rule is derived in § 3, checked against every
> case in § 3.1, and measured in § 9; it is not attributed to anyone else.

### 3.2 The masses the two paths weigh with

A count is not the only thing the two paths can disagree about. Both mass-weight
the Hessian, and **both must use the same masses and the same normalisation**, or
the held-atom run reports different physics from the free one for no physical
reason. Two conventions exist in the wild for each, and picking differently on
the two paths is silent:

| | the two choices | what must be used, and why |
|---|---|---|
| **which masses** | whole mass *numbers* (H = 1, S = 32) or real isotope-averaged masses (H = 1.008, S = 32.06) | **isotope-averaged.** PySCF's `harmonic_analysis` and `thermo.thermo` both use them, and `thermo` takes no mass argument — so it is the only convention that agrees with itself. Note `atom_mass_list()` *defaults to the whole numbers*: the convention has to be asked for. |
| **how modes are normalised** | `Σₖ mₖ|Lₖ|² = 1` with masses in electron masses, or in amu | **amu.** The Gaussian/ORCA infrared prefactor `42.2561` is derived for amu; feed it a mode normalised in electron masses and every intensity is wrong by the ratio, **1823×**. |

Measured, BDT at one geometry, free versus both sulfurs held (§ 9): the ring C–H
stretches agree to **0.001 cm⁻¹** and their infrared intensities to **0.2 %**.
Mixing the mass tables would move those frequencies **14.9 cm⁻¹** apart; mixing
the normalisations would put the held run's intensities **1823×** low. Both
defects are invisible in a free-atom run, because a free-atom run only ever
takes one of the two paths.

The frequency conversion follows from the choice and is therefore derived, never
typed a second time: a Hessian in Hartree/Bohr² weighted by amu has eigenvalues
in Hartree/(Bohr²·amu), so

```text
    wavenumber (cm⁻¹)  =  sqrt(eigenvalue) × HARTREE_CM1 / sqrt(AMU_ELECTRON_MASS)
                       =  sqrt(eigenvalue) × 5140.487…
```

whereas the electron-mass convention would use `HARTREE_CM1` alone. Both are
correct; mixing them is not. The constant lives in `constants.py`, derived from
the two it is made of.

---

## 4. When the zero is exactly zero — and when it is not

The six motions of § 2 are not equally safe.

**Translations are exact everywhere.** Slide the molecule and the energy is
identical, whatever geometry you started from.

> This asymmetry is not a subtlety of our own: Ghysels *et al.* state the same
> thing about a partially optimised geometry in one line — *"The Cartesian
> Hessian has only 3 zero-eigenvalues instead of 6, implying that the rotational
> invariance is not manifest anymore. Spurious imaginary frequencies appear."*
> [Ghysels2008] Three survive, three do not, and which three is decided by the
> gradient.

**Rotations need a stationary point.** Turning is a curved path in Cartesian
space, so the second derivative along it picks up a first-derivative term:

```text
    d²E/dθ²  =  uᵀ H u  +  ∇E · x″
```

The left side is zero (turning costs nothing). So `uᵀHu = 0` — the rotation is a
true zero mode — **only when `∇E · x″ = 0`**, which is guaranteed at a
stationary point and not otherwise.

**The held-atom case is better than it looks.** At a *constrained* minimum the
free atoms have no force on them, while the held atoms generally do — they are
carrying the constraint. Both terms of `∇E · x″` still vanish:

* on the free atoms, `∇E = 0` by definition of the constrained minimum;
* on the held atoms, `x″ = 0`, because they lie on the rotation axis and a
  point on the axis does not move at all.

So **at a constrained minimum the surviving motion is an exact zero mode**, even
though the full gradient is not zero. This is worth stating plainly because the
obvious worry — "the forces are not zero, so the argument fails" — is answered
by *which* forces are not zero and *which* atoms move.

### 4.1 What this looks like in numbers

BDT with both sulfurs held, the same two sulfurs and the same 36 modes, run at
two geometries. The table shows how much of the turning motion each mode
contains (the decomposition is exact: the squares sum to 1.000000):

| | where the turning motion went |
|---|---|
| **at the minimum** (forces on free atoms zero) | **100.0 %** in one mode at **−0.93 cm⁻¹**; every other mode ≤ 0.0004 |
| **off the minimum** (‖∇E‖∞ = 1.9 × 10⁻² Eh/Bohr) | **96.5 %** in a mode at **96.78 cm⁻¹**; 2.5 % at 284.58; 0.7 % at 157.54 |

The second row is the one to remember. Off a stationary point the turning
motion **acquires a frequency of 97 cm⁻¹ and sits in the middle of the real
vibrations**, where nothing about its frequency marks it out.

---

## 4a. Intensities — how the strength of a band is obtained

A mode list is half a spectrum. The other half is **how strongly each mode
absorbs or scatters light**, and that is a different calculation with its own
rules. It belongs in this document because it consumes the mode vectors § 3
produces, and gets the wrong answer if their normalisation is wrong.

### 4a.1 What an intensity actually is

An infrared band is strong when the vibration **moves charge** — when the
molecule's dipole moment changes as the atoms move along that mode. So the
quantity wanted is the dipole's rate of change along the mode, and the
intensity in the standard unit is

```text
    I  =  42.2561 × |dμ/dQ|²        km/mol,  with μ in Debye and Q in Å·√amu
```

Raman is the same idea one level out: a band scatters when the mode changes how
*polarisable* the molecule is, so the quantity is dα/dQ, combined by Placzek's
formula into an activity in Å⁴/amu.

> **This is where § 3.2's normalisation becomes load-bearing.** That `42.2561`
> is derived for modes normalised with masses in **amu**. Hand it a mode
> normalised in electron masses and every intensity is wrong by the ratio —
> **1823×** — which is exactly the defect measured on 2026-09-21. The mode
> count and the intensity scale are not separate concerns; they are the same
> convention, read twice.

### 4a.2 Three ways to get dμ/dR, and what each costs

| route | how | cost | availability |
|---|---|---|---|
| **analytic** | dμ/dR is the *same* response equations the Hessian already solves, contracted with dipole integrals — so asking for both costs one solve, not two | **+14 %** over the Hessian alone | needs `pyscf.prop.infrared`, which **has never been released to PyPI** — master branch only |
| **finite-difference dipoles** | nudge each atom, read the converged dipole, take the difference | **+486 %** — it needs 6N extra SCF solves | always available |
| **rides along with Raman** | the Raman sweep already runs 6·N_free displaced SCFs for dα/dR; the dipole at each point is a one-line integral on an already-converged wavefunction | **free** | whenever Raman is on |

*(Measured on NH₃/PBE0/6-31G, 2026-09-11. The two dμ/dR tensors agree to
**0.02 %**, so the choice is a cost choice, not an accuracy one.)*

**The third row is why the rule is what it is.** If Raman is being computed, the
displacement loop is already being paid for, so the analytic route would buy
nothing. Hence: *analytic only when infrared is the only intensity asked for.*

### 4a.3 Why the route is chosen at run time, not when the deck is written

`pyscf.prop.infrared` is a property of **the environment the deck lands in**, not
of the machine that wrote it. A deck composed here may run on a cluster whose
environment was installed from the package index and has no analytic route at
all. So the deck **tries** the analytic route and falls back, loudly, printing
why — and records which one ran as `ir_route` in the results.

That has a consequence for the write-up: the Methods paragraph is composed
before the run, so it must stay **route-neutral**, and the sentence naming the
route is added afterwards by whoever holds the results. A paragraph that claimed
"analytic" at compose time would be making a promise the composer cannot keep.

### 4a.4 Two corrections the analytic route needs, both measured

Taking the analytic route means using upstream's Hessian object, and twice that
silently differs from the Hessian the no-infrared path would produce. **Asking
for intensities must not move the frequencies**, so both are corrected:

| | what upstream does | the error | the fix |
|---|---|---|---|
| **density fitting** | its Hessian class is hardcoded to the **non**-DF variant, so on a density-fitted SCF it builds a non-DF Hessian of a DF density | **7.2 × 10⁻⁵** Eh/Bohr² — a **0.11 cm⁻¹** shift | hand it the SCF's own `mf.Hessian()`; agreement then 3.6 × 10⁻¹² — *and it is faster* |
| **dispersion** | computes only the electronic + nuclear terms and overwrites the result with them, dropping the dispersion Hessian | **7.2 × 10⁻⁴** Eh/Bohr² — a **3.7 cm⁻¹** shift on *every* frequency, invisible on any functional without a dispersion correction | add the term back; agreement then 5.7 × 10⁻¹² |

Neither is a rounding artefact. The second is the larger and the more dangerous,
because a 3.7 cm⁻¹ shift on every band looks like a plausible answer.

### 4a.5 Where infrared is not well defined

**Charged molecules.** A dipole moment is origin-independent only for a neutral
system. For an ion, the dipole depends on where the origin is put, so its
derivative picks up a term that is bookkeeping rather than physics. This is true
of any code, not just ours; the honest position is to compute it, say so, and
treat absolute values with suspicion.

**Systems with held atoms — with a caveat worth stating.** The intensity formula
itself is fine: it consumes the free-atom mode vectors and the free-atom dipole
derivatives, both of which exist. What changes is *interpretation* — a held atom
contributes no dipole derivative, so charge flow through the anchor is missing
from the band strength. For a molecule anchored at one or two atoms that is a
small correction; for a molecule on a metal surface, where the substrate screens
and the interface carries much of the charge transfer, it is not small. The
number is computed; how much it means is the reader's judgement.

### 4a.6 SIESTA: not offered, and why that is a statement about this tool

The SIESTA path (§ 3.1a, and the design document) computes **frequencies and
mode shapes only**. Infrared and Raman controls are **not drawn** when SIESTA is
the engine — not drawn rather than defaulted off, because a control that
silently does nothing is worse than an absent one: the user believes they asked
for something.

This is not a claim that SIESTA cannot do it. Infrared intensities are reachable
there by a different route entirely — **Born effective charges** from the
Berry-phase polarisation machinery, which answers "how does the polarisation
respond to moving an ion" for a periodic system, where a molecular dipole moment
is not even well defined. That is a separate feature with separate physics and
separate validation, and if it is ever wanted it should be designed as one
rather than implied by leaving a molecular checkbox on the page.

**And for the junction work the more useful quantity is a different one again.**
What modulates a current is not how a mode couples to light but how it couples
to the *electrons* — ∂H/∂Q rather than ∂μ/∂Q. That is the electron–vibration
coupling, and it is the thing SIESTA's force-constant machinery can hand over
directly [Frederiksen2007, Galperin2007].

### 4a.7 Validation status, stated plainly

The projection mathematics and the km/mol prefactor are textbook, and the
implementation is **band-level validated**: water at B3LYP/def2-SVP against
literature windows, with the right band ordering, and a CO₂ run reproducing the
mutual-exclusion rule of a centrosymmetric molecule (653.45 cm⁻¹ Raman-silent /
infrared-active at 32.85 km/mol; 1388.81 Raman-active at 14.74 Å⁴/amu;
2460.11 the asymmetric stretch at 613.04 km/mol, which is *the* band of the CO₂
infrared spectrum).

**What has not been done: a mode-by-mode cross-check against an external code**
(Gaussian, ORCA, Turbomole). Absolute intensities should be quoted with that
caveat until it is.

---

## 5. Why the spurious mode cannot be recognised from the answer

It is natural to ask whether the calculation's own output gives the mode away —
it would avoid building the six vectors at all. Three candidate tests, and what
each actually does:

**"It has a frequency near zero."** Fails, in both directions. Measured at
96.78 cm⁻¹ above; meanwhile genuine soft modes — torsions, floppy rings — live
at a few cm⁻¹ and would be thrown away.

**"It will stand out by symmetry, or by being orthogonal to the rest."** Carries
no information. The Hessian is symmetric by construction, so *every* mode is
orthogonal to every other; that is as true of a C–H stretch as of the rotation.
Point-group labels would not separate them either — a rotation and a real
vibration can share an irreducible representation — and they are unavailable in
any case, because the deck turns symmetry off (displacements leave the point
group).

**"It changes no bond length."** This one is real, and it is worth seeing why it
does not help. The set of displacements that leave every interatomic distance
unchanged to first order **is** the set of whole-body motions — that is what
rigidity means. So this is the same test as § 3.1, expressed as `N²` pair
constraints instead of six vectors. Measured, as the largest first-order change
in any interatomic distance per unit displacement:

| | spurious mode | nearest real mode | largest over all 36 |
|---|---|---|---|
| at the minimum | 3.4 × 10⁻⁶ | 6.2 × 10⁻⁵ | 2.98 |
| off the minimum | 2.9 × 10⁻¹ | **3.6 × 10⁻¹** | 2.87 |

At the minimum there is a factor-of-18 gap and a threshold would work. Off the
minimum the gap is **1.2×** — unusable. It degrades in the same circumstance the
frequency test does, and for the same reason: both ask the *answer* to be clean,
and the answer is only clean when the geometry is.

**So: remove the motion before diagonalising, do not detect it afterwards.**
Three reasons, each independent:

1. **It works off a stationary point.** Projection does not care what the
   gradient is.
2. **Deleting a mode afterwards leaves the contamination behind.** Off the
   minimum the turning motion is 96.5 % in one mode and **3.5 % spread across
   four others**. Removing the offender leaves that 3.5 % in the modes you keep.
   Projecting first gives 35 clean vibrations.
3. **It is already what the free path does.** § 2's code listing is the
   all-atoms-free path of this very codebase. Freezing an atom should not change
   the *kind* of calculation being done.

**Is it legitimate to project off a stationary point, where the direction is not
an exact zero?** Yes, and for a stated reason: the whole-body motions are known
not to be vibrations from the exact symmetry of the energy. Any curvature along
them is an artefact of the geometry or of arithmetic, never physics. PySCF makes
the same judgement for translations and rotations on every free run.

---

## 6. The rules

These are the testable statements. Code that disagrees with one of them is
wrong; prose that restates one of them must cite this section rather than
re-derive it.

**R1 — One derivation.** `n_rigid(F)` is computed in exactly one place, by the
rank of § 3.1. No call site re-derives it, tabulates it, or branches on
`len(F)`.

**R2 — One formula for the mode count.** `n_vib = 3·N_free − n_rigid(F)`, for
every system. The free-molecule `3N − 6` / `3N − 5` is this formula with `F`
empty, not a separate case.

**R3 — Project, then diagonalise.** Both paths remove the surviving whole-body
motions from the Hessian *before* diagonalising. The frozen path and the free
path differ in which motions survive, never in whether the removal happens.

**R4 — What is reported is what is left.** Every mode in the results is a
vibration. A spectrum, a mode animation and a thermochemistry sum all read the
same list, and none of them needs a filter to protect itself from a non-mode.

**R5 — Stationarity is judged in the subspace that is diagonalised.** The check
that the input geometry is a stationary point looks at the forces on the **free**
atoms. Held atoms carry constraint forces by definition, and those forces say
nothing about whether the partial Hessian is meaningful.

**R6 — A prediction that cannot match the run is not written.** Prose stating a
mode count either cites the run's own list or uses R2, which agrees with it by
construction. Two derivations of one number is the defect, not the remedy.

**R7 — The user is told what survived, and what it is.** When `n_rigid(F) > 0`
the surface says how many spurious motions there are and what they are (a
rotation about which line), before the run is paid for — not a guess at the
number, and not silence.

**R8 — A cost claim is a measurement, not an expectation.** No surface tells a
user that freezing saves compute unless the code it describes actually skips
that compute. Today the analytic-Hessian path does not (§ 8), so either the
partial Hessian is built over the free atoms only — as Q-Chem's is [QChemPHVA] —
or the claim is withdrawn and the real saving (the Raman displacement loop) is
named instead.

---

## 7. What the run must show — the acceptance test

The rules are checked through the results, not by looking for a function call.
For a structure with held atoms:

1. Build the whole-body motions that leave the held atoms in place (§ 3.1).
2. Mass-weight them, and decompose each over the reported modes. **In the
   correct metric** — the modes are orthonormal with the masses in, not in plain
   Euclidean length; § 4.1's decomposition sums to 1.000000 only when that is
   respected.
3. **Every reported mode has zero component along every such motion.**
4. `len(modes) == 3·N_free − n_rigid(F)`.

**The systems that exercise every branch**, all computed and agreeing:

| system | held | `n_rigid` | what it pins |
|---|---|---|---|
| water | — | 6 | the ordinary case |
| CO₂ | — | 5 | straight, with no linearity flag anywhere |
| a lone atom | — | 3 | an atom does not vibrate |
| water | O | **3** | half of six reported numbers are not vibrations |
| CO₂ | both O | **0** | ⚠ the table of § 3.2 says 1 |
| acetylene | both C | **0** | ⚠ the same trap, all four atoms on the axis |
| NH₃ | three H | 0 | three anchors not in a row pin everything |
| periodic slab | — / any | 3 / 0 | § 3.1a |
| two waters 20 Å apart | one of them | **0** | **the over-removal guard** — nothing is projected, and the free molecule keeps its six soft modes. A rule that removed them here would be committing the published error of § 3.5 |

Point 3 is the assertion worth having: it states the physical property rather
than the shape of the implementation, and it fails loudly on the defect this
document exists to close.

---

## 8. Why this document exists

On 2026-09-21 an end-to-end run of BDT — free, then with both sulfurs held, at
one shared geometry, driven through the web UI — measured the spurious mode
directly (§ 1, § 4.1). Three places in the codebase claimed to know how many
such modes there were. All three disagreed with each other and with the
measurement:

| site | said | truth |
|---|---|---|
| `pyscf/vibration_emitters.py` | *"the 6 (or 5) zero-frequency modes … simply do not exist here. **All 3·N_FREE eigenvalues are physical.**"* | one of them is a rotation |
| `spectra/methods.py::_mode_count` | `3·n_free − (5 if linear else 6)` → **30** | 35 vibrations among 36 numbers |
| `validation/spectra.py` | `6 − 2·len(frozen)` → *"2-ish spurious near-zero modes"* | **1** |

The emitter's claim is the root: it treats "some atoms are held" as a switch that
removes all six motions, when what is removed depends on **where** the held atoms
are. The other two are independent guesses at a number nothing owned.

Three further consequences fell out of the same gap, and each is a rule above:

* the spurious mode reaches the reported spectrum (R4);
* the thermochemistry drops it only because numerical noise happened to put it
  at −0.93 rather than +0.93 cm⁻¹ — a mode at +0.93 cm⁻¹ contributes
  **6.4 k_B ≈ 12.7 cal mol⁻¹ K⁻¹** of entropy, about 3.8 kcal/mol in −TS at
  298 K, from a motion that is not a vibration (R4);
* the stationarity check reads the force on every atom including the held ones,
  so a correctly-converged constrained minimum is reported as *"not a stationary
  point"* (R5).

**One more promise the code does not keep**, found in the same review and
recorded here because it is what a user *buys* when they freeze an atom.
`engines/overview.md` and the preflight advisory both say freezing cuts the cost
sharply. For a frequencies-or-IR run it does not: `dipole_derivatives` calls
`mf.Hessian().kernel()`, the **full** `N × N × 3 × 3` solve, and the held rows
are discarded afterwards at `HESS[_free_idx][:, _free_idx]`. Only the Raman
finite-difference loop actually scales with the free count.

The saving is real and achievable — it is simply not implemented. Q-Chem's
manual describes the same feature working the other way: *"only the part of the
Hessian matrix comprising the second derivatives of a subset of the atoms
defined by the user is computed … This results in a significant decrease in the
cost of the calculation"* [QChemPHVA]. Measured here: 10.6 s free versus 10.1 s
with 2 of 14 atoms held. Either the code earns the claim or the claim goes;
**R8** decides which is acceptable.

**The lesson is the one `constants.py` already records**: a fact about the
physical world belongs in one place. Eight spellings of the Bohr radius put the
same file 4 × 10⁻⁷ Å apart from itself; three derivations of the rigid-motion
count put the same run's mode budget 6 apart from itself.

---

## 9. Worked example — BDT, free and held, end to end

One geometry (RHF/STO-3G minimum, `E = −1014.2531093438 Ha` in both runs), one
structure, the only difference being whether the two sulfurs are held.

**Mode budget.**

| | `3N − 6` / `3·N_free − n_rigid` | produced | of which vibrations |
|---|---|---|---|
| free | 3·14 − 6 = 36 | 36 | 36 |
| S held | 3·12 − 1 = 35 | 36 | 35 + one rotation |

**The C–H stretches are untouched**, as they must be — sulfur is two bonds away
and barely moves in a ring C–H stretch:

| free (cm⁻¹) | S held (cm⁻¹) | difference |
|---|---|---|
| 3703.7286 | 3703.7296 | +0.0009 |
| 3711.6442 | 3711.6449 | +0.0007 |
| 3725.2537 | 3725.2532 | −0.0005 |
| 3728.4226 | 3728.4220 | −0.0006 |

This is also the check that both paths use **the same mass table** and the same
normalisation (§ 3.2): had the held path used whole mass numbers (H = 1 instead
of 1.008), these would sit **+14.9 cm⁻¹** higher; had it normalised in electron
masses, the intensities beside them would be **1823×** low.

**The S–H stretches move, by exactly the amount they should.** Holding sulfur
changes the pair's reduced mass from

```text
    free:  μ = m_S·m_H / (m_S + m_H) = 0.977278 amu
    held:  μ = m_H                   = 1.008000 amu
```

so the frequency must fall by `√(μ_free/μ_held)`:

| free (cm⁻¹) | S held (cm⁻¹) | measured ratio |
|---|---|---|
| 3285.2349 | 3234.8834 | 0.984673 |
| 3285.3623 | 3234.9786 | 0.984664 |

Predicted **0.984643**; measured differs by **3 × 10⁻⁵**. The two-body limit is
crude and the agreement is not — which is the point: the held atom really is
being treated as infinitely heavy, exactly as a partial Hessian assumes.

**And the 36th number is the rotation**: −0.93 cm⁻¹, 100.0 % of the turning
motion about the S···S line, changing no interatomic distance to 3 × 10⁻⁶.
Under R3 it is removed before diagonalisation and the run reports 35.

---

## 10. Glossary

| term | plain meaning |
|---|---|
| **normal mode** | one pattern of atoms moving together, with a single frequency; a vibrational spectrum is a list of these |
| **Hessian** | the table of second derivatives of the energy — how stiff the molecule is against every way of moving each atom |
| **partial Hessian** | the same table, restricted to the atoms that are free to move, used when some atoms are held still |
| **mass-weighting** | dividing the Hessian by the square roots of the atoms' masses; turns stiffness into frequency |
| **stationary point** | a geometry where no atom feels a force — the only place a harmonic frequency is defined |
| **constrained minimum** | a stationary point of the *free* atoms only; the held atoms still feel force, and that is expected |
| **whole-body motion** | sliding or turning the entire molecule without changing its shape; costs no energy, so it is not a vibration. Group theory calls the set of these that leave the held atoms in place the *stabiliser*; the name is not needed to use the rule |
| **rank** | how many genuinely independent directions a set of vectors spans — the arithmetic that replaces counting cases in § 3.1 |

---

## 11. References

Cited by key; entries live in
[`references.bib`](?doc=science/references.bib).

All five were checked against Crossref and the publishers' records on
2026-09-21; the bibliography records each check. Access status and the one full
text we may redistribute are in [`refs/README.md`](?doc=science/refs/README.md).

* **[Wilson1955]** Wilson, Decius & Cross, *Molecular Vibrations* (McGraw-Hill,
  1955) — the standard treatment of normal-mode analysis and the 3N−6 / 3N−5
  count, and the source for the stationary-point condition in § 4.
* **[Eckart1935]** Eckart, *Phys. Rev.* **47**(7), 552–558 (1935) — the
  separation of vibration from overall translation and rotation (the Eckart
  conditions). Cited for the separation itself, and for nothing more: the
  stationary-point argument of § 4 is the standard Hessian one and belongs to
  Wilson1955.
* **[Head1997]** Head, *Int. J. Quantum Chem.* **65**(5), 827–838 (1997) — the
  **origin** of the partial-Hessian approach, for adsorbates on surfaces.
* **[LiJensen2002]** Li & Jensen, *Theor. Chem. Acc.* **107**(4), 211–219
  (2002) — names and analyses *partial Hessian vibrational analysis*; it does
  not originate it.
* **[Ghysels2007]** Ghysels *et al.*, *J. Chem. Phys.* **126**(22), 224102
  (2007) — the mobile block Hessian, an extension of Head's method to partially
  optimised systems; its stated aim is to avoid artificial imaginary
  frequencies while keeping track of global translation and rotation, which is
  the concern § 3 governs.
* **[Besley2008]** Besley & Bryan, *J. Phys. Chem. C* **112**(11), 4308–4314
  (2008) — PHVA in practice on Si(100), and the clearest statement of *which*
  Hessian is sliced: the interaction with the held atoms is included explicitly.
* **[Ghysels2008]** Ghysels *et al.*, conference abstract, SimBioMa 2008 — cited
  for one corroborating sentence only: at a partially optimised geometry the
  Cartesian Hessian has three zero eigenvalues, not six.
* **[Vester2024]** Vester & Olsen, *J. Chem. Theory Comput.* **20**(21),
  9533–9546 (2024), CC-BY — the **counter-case** of § 3.1: for a molecule in
  frozen solvent, projecting out translation and rotation is invalid and damages
  other modes. Cited wherever the removal is justified, so that the limit of the
  removal is cited with it.
* **[Tao2021]** Tao, Zou, Nanayakkara, Freindorf & Kraka, *Theor. Chem. Acc.*
  **140**(3), 31 (2021) — places PHVA among MBH, VSA and GSVA, and states the
  limitation they share.
* **[QChemPHVA]** *Q-Chem 6.3 User's Manual* § 10.7.4 — a shipping partial
  Hessian that computes only the block, and reports the cost saving molbuilder
  promises but does not take (§ 8, R8).
