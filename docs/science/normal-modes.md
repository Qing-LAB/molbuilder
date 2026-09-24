# Normal modes — what a vibrational calculation counts, and what it must remove

**Role:** contract
**Domain:** science
**Companions:** [`engines/vibration.md`](?doc=engines/vibration.md) — **the
calculation's master contract**: how each engine obtains the block this
document reasons about, the script and the deck, the result file, the
invariants the code is checked against, and what is built and owed;
[`overview.md`](?doc=science/overview.md) (the science domain's promise),
[`validation.md`](?doc=science/validation.md) (the runtime advisory machinery
that carries these rules to the user), [`web/spectra.md`](?doc=web/spectra.md)
§ 8 (what freezing an atom means to the Spectrum tab).

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
> translations — the same unit vector repeated on every atom — and the
> rotations the lattice permits: three with no lattice, one for a wire, none for
> a slab or a crystal, where the rotation column `a`, on atom `i`, is
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

**The rule has no cases** *(stated case-free 2026-09-23; a two-row table stood
here and got a wire wrong)*: a rotation generator is an antisymmetric matrix
`A`, and the rotation is a symmetry of the energy only if it maps every lattice
vector onto itself — `A·a = 0` for every lattice vector `a` of a `periodic`
axis. The space of such `A` has dimension **3, 1, 0, 0** for **0, 1, 2, 3**
periodic axes: no lattice leaves all three turns; a wire keeps the turn about its
own axis; a slab or a crystal keeps none. Translations are always three. So `G`
has six columns for an isolated system, four for a wire, three for a slab or a
crystal — and the table of § 3 gains these rows:

| the system | held atoms | `n_rigid` |
|---|---|---|
| periodic along one axis (a wire) | none | **4** |
| periodic along two or three axes | none | **3** |
| periodic, any number of axes | **any at all** | **0** — translation would move a held atom |

Which axes are `periodic` is the structure's own `axis_kind`, per axis — never
the boolean `pbc()`, which cannot say *which* axis repeats. An axis that
continues (`transport`) binds a turn exactly as a repeating one does.

**What "moves nothing" means, since a rank needs a tolerance.** A surviving
motion is dropped when its root-sum-square displacement of the atoms in
question, per unit motion, is below **10⁻³ Å**: a molecule bent by numerical
noise is straight, and a molecule bent by a tenth of an Ångström is not. A
tolerance of a millionth would hand a real bend to the projection on a
geometry converged to ordinary criteria; PySCF's own linearity test sits at a
moment of inertia of 5·10⁻⁶ amu·Å², the same order of off-axis distance for a
light atom. The one home is `spectra/normal_modes.py`, and the four-molecule
gate against PySCF on free molecules ([`engines/vibration.md`](?doc=engines/vibration.md)
§ 4.5) is what pins it.

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
| **how modes are normalised** | `Σₖ mₖ\|Lₖ\|² = 1` with masses in electron masses, or in amu | **amu.** The Gaussian/ORCA infrared prefactor `42.2561` is derived for amu; feed it a mode normalised in electron masses and every intensity is wrong by the ratio, **1823×**. |

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

### 3.3 What every other code does — checked 2026-09-21, and why the testing weight sits on the rank rule

Before the rule was built, four places were checked for an implementation to
borrow, so that this would be a borrowing where one was available:

- **PySCF has nothing.** Not one of the eight entry points in
  `pyscf.hessian.thermo` takes a frozen list, an atom list or a mask;
  `harmonic_analysis(mol, hess, …)` reads the whole molecule and the whole
  curvature table. Four phrasings searched on its issue tracker, nothing; no
  such module in `pyscf-forge` or `pyscf/properties`; the installed tree
  grepped for `partial_hessian|phva|frozen_atoms`, nothing. *(And the obvious
  trick fails instructively: give the held atoms a huge mass and call
  `harmonic_analysis` anyway. It computes the centre of mass from those
  masses, so the centre lands on the held atoms, and it then removes six
  motions built around that centre. Six is wrong — for BDT with both sulfurs
  held the answer is one, for a held slab zero — the published error
  [Vester2024] warns against, arrived at by a shortcut.)*
- **ASE does what the old two-branch code did.** The most widely used
  implementation of held-atom vibrational analysis [ASE2017],
  `Vibrations(atoms, indices=[…])`, displaces only the chosen atoms and then
  `omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)`:
  mass-weight, diagonalise, **no projection**, `3n` modes for `n` displaced
  atoms — line for line the frozen branch this document retired. The
  leftover-motion problem is a community-wide wart, not a molbuilder
  peculiarity.
- **Others have hit the wart and bolted projection on.** A pull request
  against an ASE-based toolchain, *"Project translations/rotations out of
  Hessians for all calculators"*, gives the reason this document measured:
  the rigid-body modes otherwise stay in the frequency list *"contaminated by
  residual gradients, grid noise or finite-difference error"*, and with
  projection they *"vanish to machine precision"* — but it projects all six,
  which is right for a free molecule and wrong when atoms are held.
- **The literature says it from the physics side.** Ghysels *et al.*,
  comparing partial-Hessian techniques: *"Although the PES is still invariant
  under the six global translations and rotations, the zero eigenvalues
  corresponding to global rotations may be lacking."* [Ghysels2010]

| | what it does when atoms are held |
|---|---|
| ASE [ASE2017] | projects nothing — the leftover stays in the list |
| the ASE-based pull request | projects all six — right for a free molecule, **wrong** when atoms are held |
| Vester & Olsen [Vester2024] | identifies and removes after the fact, by character |
| **the rank rule of § 3.1** | computes the right number from the geometry: 6, 5, 4, 3, 1 or 0 |

**What this settled** *(the reason Option A was chosen, recorded 2026-09-21 and
worth keeping)*: the risk of writing the rule ourselves is **not** "replacing
a well-tested routine with ours", because there is no upstream held-atom
implementation whose correctness would be second-guessed — there is none to
replace. The real risk is the opposite one: the rank rule is ahead of every
published implementation, so nobody else's testing covers it. That is why the
whole testing weight sits on it (§ 7): it must reproduce the textbook answer
everywhere a textbook has one — the gate against PySCF on free molecules —
before it is trusted on the one case only it handles.

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

### 4a.2 How the derivatives are obtained, and the rule that follows

Which route produces dμ/dR (the Hessian's own response, contracted with
dipole integrals; a dipole at each nudged geometry; or the Raman sweep's
points read for free), what each was measured to cost, the two corrections
the analytic route needs so that asking for intensities never moves a
frequency, and the fact that the route is chosen at **run time** because the
analytic module is a property of the environment the deck lands in — all of
that is the implementation contract's,
[`engines/vibration.md`](?doc=engines/vibration.md) § 4.6. The rule this
document keeps is the one that follows from the measurements: **analytic
only when infrared is the only strength asked for and no atom is held**, and
the Methods paragraph stays route-neutral until the results say which route
ran.

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

### 4a.5b The SIESTA route

Built 2026-09-23 and described with the PySCF route, side by side, in § 4b.3:
force constants by central differences of SIESTA's forces over the free atoms
only, read back on the host and put through the same harmonic path.
Intensities are not computed on it (§ 4a.6).

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

### 4a.7 Validation status

Band-level validated (water in the literature windows with the right
ordering; CO₂ reproducing mutual exclusion), and not cross-checked mode by
mode against an external code — the numbers and the caveat are
[`engines/vibration.md`](?doc=engines/vibration.md) § 9.

---

## 4b. The theory the two engines share, a two-atom example, and the discussion this started from

§ 3 says *what* is computed: the free–free block of the true Hessian, then the
surviving whole-body motions removed, then the mass-weighted eigenproblem.
This section restates that in the terms of the discussion the work started
from, shows the two engines meeting at one function, works the smallest
example with real numbers, and records where this contract agrees with that
discussion and where it goes further. *How* each engine obtains the block —
the script, the deck, the pseudocode, the cost read from the code — is
[`engines/vibration.md`](?doc=engines/vibration.md) §§ 4–5.

### 4b.1 The theory, in five lines

Near a geometry `R₀`, write the atoms' displacement as one long vector `u`
(three numbers per atom). The energy is a quadratic bowl,

```text
    E(u)  ≈  E₀  +  ½ uᵀ H u ,            H_{Iα,Jβ} = ∂²E / ∂R_{Iα} ∂R_{Jβ}
```

and the force is minus its gradient — the many-coordinate form of Hooke's law:

```text
    F  =  −∇E  =  −H u          (F_x = −H_xx·x − H_xy·y − … : moving one atom
                                  pushes on the others; that coupling IS the
                                  off-diagonal Hessian)
```

Split the coordinates into free `A` and held `F`, and impose `u_F = 0`:

```text
    ┌ F_A ┐       ┌ H_AA  H_AF ┐ ┌ u_A ┐            F_A = −H_AA u_A
    │     │  = −  │            │ │     │      ⇒
    └ F_F ┘       └ H_FA  H_FF ┘ └  0  ┘            F_F = −H_FA u_A   ≠ 0
```

The second line is the one to keep in mind: **a held atom's displacement is
zero, its interaction is not.** It feels the free atoms move; the constraint
only forbids it from answering. So `H_AA` is a slice through the full energy
surface with every atom present — never the Hessian of a molecule with the held
atoms deleted [Besley2008].

Mass-weight the block and solve the eigenproblem:

```text
    D_AA  =  M_A^{-1/2} H_AA M_A^{-1/2} ,        D_AA e_ν  =  ω_ν² e_ν
```

What this contract adds to that textbook picture, and why (§ 3, § 4):

1. **Not every eigenvector of `D_AA` is a vibration.** A turn of the free atoms
   about one held atom, or about the line through two, costs no energy and is a
   zero of `H_AA` — it comes out of the eigenproblem looking like a mode, and
   off a stationary point it acquires a frequency (§ 4.1: 97 cm⁻¹, in the
   middle of the spectrum). The rank rule of § 3.1 finds these motions from
   the geometry, and they are projected out **before** `eigh` (R3). With three
   held atoms not on one line there are none, and nothing is removed.
2. **Stationarity is asked of the free atoms only** (R5): `∇_A E = 0`. The
   held atoms carry the constraint force `−H_FA u_A`; that force is expected
   and says nothing about whether `H_AA` is meaningful.
3. **Masses and normalisation are one convention on both routes** (§ 3.2):
   isotope-averaged masses, `Σ m_k|L_k|² = 1` in amu.

Both engines hand the same function the same three things — the block, the
masses, the geometry with its held set — and get the same answer back:

```mermaid
flowchart LR
  subgraph P["PySCF — analytic second derivatives, isolated molecule"]
    direction TB
    P1["gto.M: every atom present,<br/>held ones too"] --> P2["one whole-system SCF"]
    P2 --> P3["relax the FREE atoms (geomeTRIC, $freeze)<br/>check max|F| over the free atoms"]
    P3 --> P4["hess_elec(atmlst=A) + hess_nuc(atmlst=A)<br/>+ D3 block [A,A]  →  H_AA"]
  end
  subgraph S["SIESTA — central differences of forces, periodic or not"]
    direction TB
    S1["prep: sort a COPY held-first,<br/>record atom-permutation.json"] --> S2["deck: Geometry.Constraints ·<br/>MD.TypeOfRun FC · FC.First..FC.Last = the free range"]
    S2 --> S3["run: SCF at R₀, then ±δ on each free coordinate<br/>→ forces on every atom → &lt;label&gt;.FC"]
    S3 --> S4["summarize: read .FC, central difference,<br/>symmetrise → H_AA · invert the permutation"]
  end
  P4 --> M["spectra/normal_modes.vibrational_modes<br/>mass-weight H_AA · remove the surviving whole-body<br/>motions (the rank of § 3.1) · diagonalise"]
  S4 --> M
  M --> R["&lt;label&gt;.spectra.json — the one artifact:<br/>modes · what was removed · thermochemistry ·<br/>intensities (PySCF) or null (SIESTA)"]
```

### 4b.2 A two-atom example with real numbers

The discussion this started from worked the textbook case: two equal atoms on a
spring, Hessian

```text
    H  =  ┌  k  −k ┐        eigenvalues 0 and 2k
          └ −k   k ┘        eigenvectors (1, 1)  — both atoms move together: a translation
                                         (1, −1) — against each other: the stretch
```

Here is the same molecule out of the measured fixture (`tests/fixtures/siesta_fc`,
H₂ at 0.741 Å, SIESTA GGA/DZP), with `k = 41.713 eV/Å²` read from the `.FC` file
and `m = 1.008 amu`:

| | what is diagonalised | motions removed | modes | ω |
|---|---|---|---|---|
| **both atoms free** | the 6×6 block; along the bond, the 2×2 above | 5 (three slides, two turns; the axial turn moves nothing) | 1 | `√(2k/m)` = **4744 cm⁻¹** |
| **one atom held** | the 3×3 block of the free atom: `[k]` along the bond | 2 (the two turns about the held atom: the free atom moving sideways costs nothing) | 3·1 − 2 = 1 | `√(k/m)` = **3355 cm⁻¹** |

The held case is the discussion's `(1, −1)` stretch with one end nailed down:
the mode is now `(1)` on the free atom alone, and the frequency is lower by
exactly `√2`, because the partner no longer recoils — the reduced mass is `m`
instead of `m/2`. This is the "constrained Hessian is not the full spectrum"
point made quantitative: holding an atom is a choice about the physics, and
the two numbers above are both right for the question each asks. The two
sideways motions of the free atom are the surviving whole-body motions of
§ 3.1 (`n_rigid = 2` for one held atom on a line of two), and without R3 they
would be reported as two modes near zero. The run through jobset
(`tests/test_siesta_vibration_e2e.py`; `vibration.md` § 5.5) reports exactly
one mode, two motions removed, and the free atom by its input number.

### 4b.3 The discussion this started from, and where this contract differs

The work began from a discussion of a molecule anchored on a metal surface
(*"a large array of metal atoms fixed, a few metal atoms allowed to participate
with the adsorbed molecule"*), which set out the picture § 4b.1 restates. The
cross-check, point by point (the implementation sections cited are
[`engines/vibration.md`](?doc=engines/vibration.md)'s):

| the discussion said | this contract |
|---|---|
| keep every metal atom in the quantum calculation; take the Hessian only with respect to the coordinates allowed to move (`H_AA`); "deleting the frozen rows" means deleting *degrees of freedom*, not atoms | **the same**, and it is the built behaviour on both engines (`vibration.md` § 4.4, § 5.5); on PySCF the held rows are never computed at all |
| `hess.kernel(atmlst = active_atoms)` | **differs, by measurement**: `kernel(atmlst=)` adds a full-size dispersion term and the density-fitted Hessian class refuses the list, so the deck sums `hess_elec + hess_nuc + D3[A,A]` on a plain mean field (`vibration.md` § 4.4) |
| mass-weight `H_AA` and diagonalise; the eigenvectors are the modes | **goes further**: the surviving whole-body motions are removed first (R3, § 3.1). For a slab or three anchors off a line nothing survives and the two agree; for one or two held atoms the difference is measured — water with its oxygen held reported six numbers, three of them turns, and the two loudest infrared bands were among them (§ 8; `vibration.md` § 11) |
| the condition is `∇_A E = 0`; the frozen atoms may carry force | **the same** (R5) — and it was a live defect here until 2026-09-23: the check read every atom and warned on every converged constrained minimum |
| the constrained Hessian makes the substrate infinitely rigid; converge the answer by adding active metal layers (molecule only, +1 layer, +2, +3) and watching the molecular modes | **the same judgement, and it is the person's**: the artifact records the free set and `hessian_scope`, so two runs that differ only in which atoms are held are comparable by their own files. § 3's honest-cost paragraph and § 4a.5 say the same about intensities |
| frozen is not cheap electronically — "a 200-Au cluster remains a 200-Au electronic-structure problem" | **the same, and sharpened from the code** (`vibration.md` § 4.4, § 5.7): on PySCF the response equations and the derivative integrals shrink with the free atoms, the exchange-correlation derivative matrices do not, and at hundreds of atoms those matrices are the wall; on SIESTA the count of force evaluations shrinks and each stays whole-system |
| PySCF is a finite-system code; a metal surface is a periodic solid | **the same, and it is why there are two routes**: the isolated molecule goes to PySCF, the slab or junction to SIESTA, and both hand the same block to the same harmonic path |
| whether PySCF can give infrared or Raman intensities for the constrained system "depends on the property implementation" | **settled** (§ 4a; `vibration.md` § 4.6): the analytic dipole-derivative route takes no atom list, so with atoms held infrared goes by central differences of the dipole over the free atoms, and Raman by central differences of the polarizability over the free atoms — the same numbers by the other route |
| a fixed atom's displacement is zero, its interaction is not: `F_F = −H_FA u_A ≠ 0` | **the same**, and it is the sentence a reader of a held-atom result should carry: the held atoms shape every number in `H_AA` |

What the discussion did not need and this contract had to add: the count of
motions that survive a hold, computed rather than tabulated (§ 3.1, because
the Spectrum tab lets anyone hold one atom of a molecule, the case the surface
literature never meets); the reorder machinery SIESTA's contiguous range
forces, and the record that undoes it (`vibration.md` § 5.2); and the artifact's honesty
about absent numbers (`web/spectra.md` § 9b.3), so a SIESTA file cannot be
read as a PySCF file with zero intensities.

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

**R3 — Project, then diagonalise, at the Γ point.** Both paths remove the
surviving whole-body motions from the Hessian *before* diagonalising. The
frozen path and the free path differ in which motions survive, never in whether
the removal happens. **At q = Γ only** *(ruled 2026-09-23)*: `n_rigid` is a
property of one geometry, and a phonon dispersion `D(q ≠ 0)` carries real
curvature along its acoustic branches that this rule must not remove — the
SIESTA path is Γ-only by design, and a dispersion is a different feature.

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

**R8 — A cost claim is read from the code, and what the code cannot skip is
stated beside it.** No surface tells a person that holding atoms saves compute
unless the code it describes skips that compute.
[`engines/vibration.md`](?doc=engines/vibration.md) § 4.4 and § 5.7 say, from
the code text, which pieces of each route run over the free atoms and which
run over every atom — and the piece that does not shrink (PySCF's
exchange-correlation derivative matrices) is stated as plainly as the pieces
that do. The partial Hessian is built over the free atoms only, as Q-Chem's is
[QChemPHVA]. A timing is a check on this statement, never its source.

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
| water | O + one H | **1** | two held atoms leave one turn that moves the free hydrogen — the old two-branch code reported 3 here where R2 gives 2 |
| CO₂ | both O | **0** | ⚠ the table of § 3 says 1 |
| acetylene | both C | **0** | ⚠ the same trap, all four atoms on the axis |
| NH₃ | three H | 0 | three anchors not in a row pin everything |
| periodic slab or crystal | — / any | 3 / 0 | § 3.1a |
| periodic wire (one axis) | — | **4** | § 3.1a — the turn about the wire's own axis survives |
| two waters 20 Å apart | one of them | **0** | **the over-removal guard** — nothing is projected, and the free molecule keeps its six soft modes. A rule that removed them here would be committing the published error § 3.1 warns against |

Point 3 is the assertion worth having: it states the physical property rather
than the shape of the implementation, and it fails loudly on the defect this
document exists to close. **Pinned**: tier 1 in `tests/spectra/test_normal_modes.py`
(every row above, no engine); the four-molecule equivalence with PySCF and the
held-oxygen water run through the whole described road in
`tests/test_vibration_e2e.py`.

---

## 8. Why this document exists

*(This section is the record of what was found on 2026-09-21. Every defect it
names was fixed by 2026-09-23 — [`engines/vibration.md`](?doc=engines/vibration.md)
§ 10 says what stands — and the present tense below is the record's, kept so
the reason for each rule stays visible.)*

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
* the stationarity check read the force on every atom including the held ones,
  so a correctly-converged constrained minimum was reported as *"not a
  stationary point"* (R5; fixed 2026-09-23).

**One more promise the code does not keep**, found in the same review and
recorded here because it is what a user *buys* when they freeze an atom.
`engines/overview.md` and the preflight advisory both said freezing cuts the
cost sharply. For a frequencies-or-IR run it did not: `dipole_derivatives`
called `mf.Hessian().kernel()`, the **full** `N × N × 3 × 3` solve, and the
held rows were discarded afterwards at `HESS[_free_idx][:, _free_idx]`. Only
the Raman finite-difference loop scaled with the free count.

The saving was real and achievable — and it was not implemented until
2026-09-23, when the Hessian became the free atoms' block
([`engines/vibration.md`](?doc=engines/vibration.md) § 4.4). Q-Chem's manual
describes the same feature working the other way: *"only the part of the
Hessian matrix comprising the second derivatives of a subset of the atoms
defined by the user is computed … This results in a significant decrease in the
cost of the calculation"* [QChemPHVA]. Measured here, before the fix: 10.6 s
free versus 10.1 s with 2 of 14 atoms held. Either the code earns the claim or
the claim goes; **R8** decides which is acceptable, and now reads the claim
from the code.

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
  promised before it took it (§ 8, R8).
* **[ASE2017]** Larsen *et al.*, *J. Phys.: Condens. Matter* **29**, 273002
  (2017) — the Atomic Simulation Environment, whose `Vibrations(indices=…)`
  is the community's held-atom analysis: mass-weight and diagonalise, no
  projection (§ 3.3).
* **[Ghysels2010]** Ghysels *et al.*, comparing partial-Hessian techniques —
  cited for one sentence in § 3.3: at a partially held geometry the zero
  eigenvalues of the global rotations may be lacking.
