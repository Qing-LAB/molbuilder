# Vibration calculations — the audit, and what is left

**Role:** audit — the follow-up list
**Domain:** science · engines · web
**Opened:** 2026-09-22, from an end-to-end run driven through the web interface
**The science:** [`science/normal-modes.md`](?doc=science/normal-modes.md)
**The design:** [`plans/2026-09-21-normal-mode-unification-design.md`](?doc=plans/2026-09-21-normal-mode-unification-design.md)
**Plan row:** **V1** in [`plan.md`](?doc=plans/plan.md) § 2

Every item below was **measured**, not inferred, unless it says otherwise. Each
says what it is, what actually goes wrong, where it lives, and what fixing it
means. Plain language throughout — this is a list to work from, months from now,
by someone who was not in the conversation.

---

## 0. How this was found, in one paragraph

1,4-benzenedithiol — a benzene ring with an `–SH` at each end, 14 atoms — was
built through the web interface and run **twice at one geometry**: once with
nothing held, once with both sulfur atoms held. Same structure, same method
(Hartree–Fock/STO-3G), same energy to twelve digits, so every difference between
the two runs is the freezing and nothing else. That pair is the evidence behind
almost every row here.

---

## 1. Already fixed — do not redo these

| | what it was | how it was proven |
|---|---|---|
| **the mass tables disagreed** | held-atom runs used whole mass numbers (H = 1), free runs used real masses (H = 1.008). Frequencies off ~15 cm⁻¹; **infrared intensities 1823× wrong**, because the standard prefactor assumes one convention and the code fed it the other | with both sulfurs held, ring C–H stretches now match the free run to **0.001 cm⁻¹** and their intensities to **0.2 %** |
| **runs wrote bare `.xyz` files** | `_initial.xyz` / `_optimized.xyz` were written with no companion JSON, so a run's geometry carried no labels, no cell, no identity | the run now writes the pair; `frozen_atoms: [6, 7]` survives into the output, and molbuilder's own codec reads it back |

Both are uncommitted at the time of writing. Full non-e2e suite after them:
**9426 passed, 3 skipped, 6 xfailed.**

---

## 2. The physics — what is still wrong

### 2.1 A rotation is still reported as a vibration ⚠ the main one

**What happens.** Hold the two sulfur atoms and the run returns **36 numbers**.
Thirty-five are vibrations. The thirty-sixth is *the whole ring turning about
the line joining the two sulfurs* — nothing stretches, nothing bends, no bond
changes length. The sulfurs sit *on* that line, so holding them does not prevent
the turn, and it costs no energy.

**The evidence.** It comes out at **−0.93 cm⁻¹** and accounts for **100.0 %** of
that turning motion while all 35 other modes account for 0.0 % between them.

**Why it can't be spotted afterwards.** At a slightly-off geometry the same
motion comes out at **96.78 cm⁻¹**, sitting among real vibrations with nothing
about it looking wrong — and it smears 3.5 % of itself across four genuine
modes. So it must be removed *before* the solve, not detected after.

**Where.** `pyscf/vibration_emitters.py`, the frozen branch of the analysis
block, which asserts *"All 3·N_FREE eigenvalues are physical."* That claim is
false whenever fewer than three non-collinear atoms are held.

**What fixing it means.** Option A of the design: one code path that computes
how many whole-body motions survive the freeze, and removes them. Not a patch —
it replaces the two-branch structure that allowed the drift.

### 2.2 The reported free energy depends on numerical luck ⚠

**What happens.** The thermochemistry protects itself with *"skip anything
imaginary, skip anything not above zero"*. Our fake mode was excluded **only
because noise put it at −0.93 rather than +0.93 cm⁻¹**.

**What it would cost.** A mode at +0.93 cm⁻¹ contributes **6.4 k_B ≈ 12.7
cal mol⁻¹ K⁻¹** of entropy — about **3.8 kcal/mol** in the free energy at 298 K
— from a motion that is not a vibration. Nothing warns.

**Where.** `pyscf/vibration_deck.py::_vib_thermo_block`.

**What fixing it means.** It disappears for free once 2.1 is fixed: if every
reported mode is a vibration, the filter has nothing to protect against and is
deleted. A filter that works by luck is not a filter.

### 2.3 "Is this geometry relaxed?" asks about the wrong atoms — *derived, not yet measured*

**What happens.** When a user asserts a geometry is already relaxed, the deck
checks it by taking the largest force **over every atom, including the held
ones**.

**Why that's wrong.** A held atom is *supposed* to feel force — that is what
holding means, like a nail holding a stretched rubber band. So the ordinary
workflow (relax with the substrate held, then compute frequencies at that
geometry) will print *"input geometry is not a stationary point"* on a geometry
that is perfectly correct for the atoms that can actually move.

**Where.** `pyscf/vibration_deck.py::_vib_gradient_check` —
`np.abs(_g0).max()` over all atoms.

**Status.** Read from the code path; **not reproduced**. Our runs froze at the
unconstrained minimum, where all forces vanish, so it never fired. Reproducing
it costs about a minute: relax with the sulfurs held, then a frequency run at
that geometry.

**The rule it should follow.** The forces must vanish on the atoms that
*vibrate*. And a related rule worth writing down while we are here: **the set
frozen for the spectrum should be a subset of the set frozen for the
relaxation.** Freezing *more* for the spectrum is always safe; freezing *less*
puts you off a stationary point in the vibrating subspace.

### 2.4 Freezing atoms does not make the job cheaper, though two places promise it does

**What happens.** The code computes the **full** second-derivative table for
every atom and then throws the frozen rows away.

**Measured.** 10.6 s free versus 10.1 s with 2 of 14 atoms held — no saving
beyond noise.

**Who promises otherwise.** `engines/overview.md` (*"freezing a slab or anchor
cuts the cost sharply"*) and the pre-run advisory in
`validation/spectra.py` (*"the cost saving is typically large"*). For a **Raman**
run the advice is true — that loop really does scale with the free count. For a
frequencies-or-infrared run it is false, and that is the tab's default.

**The saving is real and reachable.** PySCF's Hessian takes an atom list, and
that list reaches the expensive inner solve, so asking for four atoms out of
forty really does cost about a tenth. Q-Chem's manual describes the same feature
working properly. Two cautions: the result is numbered by position in the list
you passed, not by the atom's own index; and no upstream test exercises a
partial list, so it owes us a check against compute-everything-and-slice.

**The catch.** The analytic infrared route does not accept that list — so today
it is the cost saving *or* analytic infrared. **Decided: this becomes an option
in the run setup**, with the run stating which way it went. The default is still
unpicked (§ 6).

---

## 3. What the run *says* about itself — still wrong

### 3.1 The Methods paragraph names a method that was not run ⚠ publication-facing

**What happens.** Every run's Methods text reads *"performed at the
**B3LYP/sto-3g** level with the **D3BJ** dispersion correction"* and cites two
papers for them. Our runs were **plain Hartree–Fock** with neither applied —
`mf = _scf.RHF(mol)`, no `mf.xc`, no `mf.disp` anywhere in the deck. The very
next paragraph of the same text correctly says `pyscf.hessian.rhf`, so the
document contradicts itself.

**Why.** `spectra/methods.py::_paragraph_vibrational` reads `cfg.functional` and
`cfg.dispersion` unconditionally, with no reference to `cfg.method`. Whether
they are *applied* is decided elsewhere, in the deck. Two places deciding one
fact — **the same disease the mass fix cured.**

**What fixing it means.** One function returning the *effective* level of theory
— method, functional, dispersion, basis — consulted by both the SCF construction
and the prose. Plus an advisory when a functional or dispersion is set under a
Hartree–Fock method, since the form offers them silently today.

### 3.2 Three places disagree about how many modes there are

| | says | truth |
|---|---|---|
| the deck | *"all 3·N_FREE eigenvalues are physical"* | one is a rotation |
| `spectra/methods.py::_mode_count` | **30** | 36 produced, 35 real |
| `validation/spectra.py` | *"2-ish spurious near-zero modes"* | exactly **1** |

Three independent guesses at one number. Also in the same three lines:
`if n_free < 2: return 0` is wrong (one free atom anchored by three
non-collinear held atoms has **3** vibrations), and `_is_linear(struct)` asks
about the whole structure when only the free subset could matter.

**And its own headline rule never runs.** `_mode_count`'s docstring says *"THE
CALCULATION IS THE AUTHORITY when there is one"* — but the only production
caller is the deck composer, which renders **before** the run with
`results=None`. That arm is exercised by its own fixture and nothing else, and
the prediction is what lands in the results file.

**Two traps worth knowing**, because a hand-written table of cases gets them
wrong and the rank rule gets them right: **CO₂ with both oxygens held** and
**acetylene with both carbons held** both leave **zero** spurious modes, not
one, because the free atoms lie *on* the axis so the surviving turn moves
nothing.

### 3.3 The run banner always says "all atoms free"

The frozen run printed `Constraints : (no frozen_atoms -- all atoms free)`
while freezing two atoms. `runwrap.py:816` greps the generated deck for a
comment line — `# Source: Structure.frozen_atoms` — that **no deck writes**.

Note for whoever fixes it: the canonical carrier is
`FROZEN_INDICES_USER = [6, 7]  # 0-based`, and a naive digit count of that line
returns **3**, because of the trailing `# 0-based`. The wrapper is regenerated
with every deck, so grepping generated source buys no robustness at all — the
count should be interpolated at prep time from the set prep already holds, the
way `warm_files_label` already is.

### 3.4 The results file's structure identity has the job name baked into it

`spectra.json`'s `structure_hash` is built from `[f'{N_ATOMS}', f'{JOB}']` plus
coordinates — **the job name is line 1**. So the same molecule under two job
names has two identities, and it never matches the `.molstruct.json` sidecar's
hash of the same structure, which uses a different scheme entirely.

Its own docstring says it exists *"so the parser can refuse to merge results
from a different molecule"*. It cannot do that reliably.

*(Note: the pair a **run writes out** was fixed on 2026-09-22 and now hashes the
document's bytes. This is the other one, inside the results.)*

---

## 4. Structure identity — the design, and one gap in the code

Settled in discussion, **not yet written into any contract**:

- **Two hashes, one home.** A **geometry hash** over the atom lines (verbatim
  text — the numbers are already text, so there is no float question), plus the
  lattice and per-axis periodicity from the sidecar, with `none` as a real
  hashed value when there is no lattice. And a **broad hash** over labels,
  regions and identity columns, for provenance.
- **Never the comment line.** It is a human title from one writer and a
  *derived* `Lattice=` from another; the parse layer already says adopting that
  would *"promote a DERIVED value into a stored one."*
- **Out:** the cell origin (a gauge choice — shifting it moves no periodic
  image), labels, timestamps, the job name.
- **Joins check the geometry hash only.** Labels are bookkeeping; the physics
  needs the same atoms in the same places. A broad-hash difference is *reported*,
  never blocking.
- **Minted at three gates** — Modify → *Save to project*, Results → export, and
  the CLI on request — then **carried and never recomputed**. Checked at three
  points: prep, results load, and joining two runs.
- **Derived structures** (e.g. a reordered copy) mint their own identity *and*
  record the parent's plus the permutation between them.
- **A run's output is a record, not a gate.** It becomes a calculation input
  only by passing through a gate. *(Ruled 2026-09-22.)*

**The gap in the code:** `workingcopy_structure.py:265` —
`keep_sidecar = (not _metadata_is_default(meta))`. A structure whose metadata is
all default gets **no companion**, so *Save to project* can still emit a bare
`.xyz` — at the very gate meant to guarantee the pair. Both of our saves got one
only because they happened to carry labels.

---

## 5. Documentation that is no longer true

| where | says | reality |
|---|---|---|
| `engines/overview.md` ¹ | spectra consumes *"the **form value**… the form stays authoritative"* | that form field retired at P2; the frozen set travels with the structure, and `web/spectra.md` § 8 says so |
| `engines/overview.md` ¹ | *"computes second derivatives only for the free atoms, so freezing cuts the cost sharply"* | false for frequencies/infrared — § 2.4 |
| `pyscf/vibration_emitters.py:1093` | the canonical normalisation is *"mass in atomic units"* | it is **amu** — and that exact confusion is the 1823× bug |
| the Methods text **and** the pre-run advisory | *"the pre-Hessian relaxation holds them fixed (geomeTRIC `$freeze`)"* | said unconditionally; with *already relaxed* asserted there is no relaxation and no constraints file |

---

## 6. Decisions outstanding

1. **The cost-versus-analytic-infrared default** (§ 2.4). Analytic is today's
   behaviour, so keeping it changes nothing for existing users; the cheap route
   is what anyone freezing a slab will want.
2. **Sequencing.** Does the SIESTA engine get built now or after the
   unification? Recommendation: after — adding a second engine to today's
   two-branch code gives four branches to keep in step.
3. **Whether to measure § 2.3** before acting on it — about a minute of compute.

---

## 7. Not built at all

- **Option A**, the one-path unification. Chosen; nothing written. The order of
  work and the test systems are in the design document, § 7.5 and § 7.6, and
  step 2 there is a **gate**: if the new rule does not reproduce PySCF's
  free-molecule answer, nothing after it gets built.
- **The SIESTA engine.** The Spectrum tab offers `choices: ("pyscf",)`, and
  SIESTA's force-constant machinery appears nowhere in the codebase — although
  `vibra`, which turns force constants into modes, is already installed in our
  SIESTA environment, and the project's own directory layout already separates
  `frequency/` from `spectrum/`, which is exactly the split wanted.
- **The identity rule** of § 4.
