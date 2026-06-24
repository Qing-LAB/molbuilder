# Optimization parameter tuning — cross-engine guide

**Audience**: anyone running a molbuilder-generated SIESTA or PySCF
script who wants to know *what to set, for what purpose, and why*.

**Scope**: the parameters that genuinely depend on what you're using
the result for — optimizer algorithm, convergence thresholds, SCF
tolerance, mesh / basis / k-grid quality, step caps.  Parameters
whose answer is the same across all tiers (charge auto-detect, log
file naming, etc.) are not covered here.

This doc is the **canonical reference** the form fields' `help`
strings and the stage-table presets point at.  When in doubt about
what value a knob should carry, the answer should be derivable from
this doc.

---

## 1. The four-tier framework

Every parameter in this guide has values in **at most four tiers**.
The tier names are stable across engines:

| Tier | When to use | Result quality |
|---|---|---|
| **screening** | Triaging tens of candidate structures, debugging a workflow, verifying a builder produced sane geometry. | Wrong basis / loose SCF; structures within ~0.05 Å, energies within ~5 kcal/mol of the real answer. **Never publish.** |
| **loose preopt** | Stage 1 of a multi-stage relaxation — fixing obvious geometry sins (bad bond lengths, eclipsed conformers) before the expensive functional sees them. | Geometry close enough that the publishable tier won't waste cycles undoing bad initial guesses. |
| **publishable** | The default tier for any paper on simple organic chemistry / single-stage surface relaxation.  This is what Gaussian's `OPT` defaults give you; what reviewers expect. | Geometry to ~10⁻³ Å, energies to ~0.1 kcal/mol, forces below ~0.04 eV/Å (≈ 0.001 Ha/Bohr). |
| **tight (vib/IR/NEB)** | Vibrational analysis (real-mode requirement), IR/Raman intensities, transition-state search, NEB barrier heights to chemical accuracy. | Forces below ~10⁻⁵ Ha/Bohr (Gaussian `OPT=Tight`); needs the SCF + mesh equally tight. |

**Decision recipe (read in order, take the first match):**

1. Reporting *energy differences* across structures to better than
   1 kcal/mol, *or* doing vibrational/IR analysis, *or* searching a
   transition state → **tight**.
2. Reporting a single-structure relaxed geometry / total energy /
   simple property (HOMO-LUMO, Mulliken, dipole) for a paper →
   **publishable**.
3. Need a sensible geometry as input to a later, expensive stage
   (e.g. handoff from gas-phase to TranSIESTA) → **loose preopt**
   for the first stage, **publishable** for the next.
4. Sanity-checking that the builder didn't put atoms on top of each
   other → **screening**.

The molbuilder PySCF stage-table widget (task #534) bakes tiers
2–4 in as the default `stage1` / `stage2` / `stage3` rows.  Tier 1
("screening") is what the bare `cfg.basis = "def2-SVP"` default
gives without a staged ladder; not exposed as a stage.

---

## 2. Parameter-by-parameter tuning

Each section has the same shape: **what it controls (physics)** /
**per-tier values + rationale** / **per-engine keyword map** /
**references**.

### 2.1 Optimizer algorithm

**What it controls.**  How the geometry is updated between SCF
calls.  The algorithm only sees the gradient at the current
geometry (sometimes the previous few); it has to decide what step
to take.

| Algorithm | Family | Strengths | Weaknesses |
|---|---|---|---|
| **CG** (Polak-Ribiere or Fletcher-Reeves) | Conjugate gradient | No memory cost; well-behaved when forces are large (far from minimum). | Oscillates across a basin near a minimum, especially on stiff/coupled systems (metals, interfaces, vdW). [^1] |
| **Broyden** (modified, à la D. D. Johnson) | Quasi-Newton | Builds a Hessian approximation across moves; converges in few steps near a minimum where CG oscillates. | Needs to keep `M` history vectors (memory: M × 3N floats); poisoned by bad early steps. [^2] |
| **BFGS / L-BFGS** | Quasi-Newton | Like Broyden but with the BFGS update — Gaussian's default since 1982; geomeTRIC uses it. | Same caveats as Broyden. |
| **FIRE** | MD-inspired | Robust on rough landscapes (random-built initial geometries); always descends in energy. | Slower than quasi-Newton near a minimum. [^3] |
| **GDIIS / RFO** | Subspace methods | Excellent for TS search + tight minima. | Not exposed by SIESTA / current molbuilder PySCF defaults. |
| **Verlet / Nose** | MD integrators | Not relaxation — finite-T MD only. | Don't use for geometry opt. |

**Per-tier choice:**

| Tier | Optimizer | Rationale |
|---|---|---|
| screening | CG or FIRE | Far from minimum; quasi-Newton has no good Hessian to fit; cheap is fine. |
| loose preopt | **CG** | Same as screening + you want predictable per-step cost.  CG won't surprise you. |
| publishable | **Broyden (SIESTA) / geomeTRIC=BFGS (PySCF)** | You're near the minimum; CG's oscillation is the dominant cost. |
| tight | **Broyden / BFGS** | Same as publishable + more iterations.  Don't switch back to CG. |

**Engine keywords:**

| Engine | Keyword | Notes |
|---|---|---|
| SIESTA | `MD.TypeOfRun CG` / `Broyden` / `FIRE` / `Verlet` / `Nose` / `none` | `none` = single-point (skip MD block). |
| PySCF | `cfg.optimizer = "geometric"` (default) / `"berny"` | `geomeTRIC` uses BFGS internally; `berny` is built into PySCF but less robust on biomolecules.  **In molbuilder's PySCF staged-opt (#534) only `geometric` is supported** — the per-stage convergence kwarg list (`convergence_drms`/`convergence_dmax`) isn't accepted by berny. |

**Worked example.**  The 2026-06-23 BDT/Au(111) junction case
(444 atoms, organic-on-metal, vdW interface, MaxForceTol = 0.04
eV/Å): CG bounced between 0.087 and 0.54 eV/Å for 20+ moves over
12 hours with no monotonic trend.  Broyden on the same system with
`MD.MaxCGDispl = 0.02 Å` is expected to converge in ≤ 30 moves.

[^1]: Hestenes, M. R. & Stiefel, E. *J. Res. Natl. Bur. Stand.* **49**, 409 (1952).  Polak, E. & Ribière, G. *Rev. française d'inf. rech. opér.* **3**, 35 (1969).
[^2]: Johnson, D. D. "Modified Broyden's method for accelerating convergence in self-consistent calculations." *Phys. Rev. B* **38**, 12807 (1988).
[^3]: Bitzek, E. *et al.* "Structural Relaxation Made Simple." *Phys. Rev. Lett.* **97**, 170201 (2006).

---

### 2.2 Step displacement cap

**What it controls.**  Hard ceiling on how far any single atom can
move in one optimizer step.  Catches line-search over-shoot and
keeps the optimizer from leaving the basin you started in.

| Tier | Value | Rationale |
|---|---|---|
| screening | 0.30 Å | Big steps cover ground fast on a clean PES. |
| loose preopt | **0.20 Å** | SIESTA's default; OK while gradients are large. |
| publishable | **0.05 Å** | Once forces are ~0.1 eV/Å, a 0.2 Å step routinely over-shoots into the next line-search regime.  See worked example in §2.1. |
| tight | **0.02 Å** | Smallest steps that still make detectable progress at SCF noise floor. |

Gaussian's `OPT` defaults `MaxStep` to 0.30 Bohr (≈ 0.16 Å); the
`Tight` keyword tightens it to 0.02 Bohr (≈ 0.011 Å). [^4]

**Engine keywords:**

- SIESTA: `MD.MaxCGDispl` — **universal across CG / Broyden / FIRE
  in SIESTA 5.4.2** despite the CG-prefixed name.  The phantom
  variants `MD.MaxDispl` (for Broyden/FIRE) listed in some older
  references are NOT applied to Broyden's per-step cap (recognized
  as a fdf key but silently mis-applied) — see decision-log
  2026-06-23 in `design.md`.
- PySCF / geomeTRIC: not exposed as a direct cap — controlled via
  `convergence_dmax` (max displacement at convergence) + the
  optimizer's own line search.  See §2.4.

[^4]: Frisch, M. J. *et al.* "Gaussian 16 User's Reference", `OPT` keyword. [link.gaussian.com/g16src](https://link.gaussian.com/g16src)

---

### 2.3 Force convergence threshold

**What it controls.**  The "we're done" criterion.  Maximum
absolute force on any atom (constrained atoms are excluded
automatically).

#### 2.3.1 Design considerations (system-type-aware tier framework)

**The 2026-06-23 realignment.**  Pre-2026-06-23 the molbuilder
"tight" tier carried geomeTRIC's `GAU_TIGHT` preset
(`gmax = 1.5 × 10⁻⁵ Ha/Bohr ≈ 0.00077 eV/Å`).  That is Gaussian's
*very-tight* setting — designed for small-molecule vibrational
analysis / IR intensities / transition-state search / NEB barrier
heights to chemical accuracy.  **For any system with more than
~50 metal atoms it chases SCF noise and never converges.**  This
surfaced concretely during BDT-on-Au(111) junction debugging when
the user asked "is this practical for large crystal systems?".

**The framework now in use** acknowledges that "tight" means
something different across system types:

| System type | Tight = | Reason | Reference |
|---|---|---|---|
| **Crystal / surface / interface (≥ 50 atoms)** | **0.01 eV/Å max-force** | SCF noise floor on extended systems grows with system size; chasing < 0.005 eV/Å is futile.  The community-accepted production threshold for surface DFT is 0.01 eV/Å. | VASP default `EDIFFG = -0.01`; QE tight `forc_conv_thr 2 × 10⁻⁴ Ry/Bohr ≈ 0.005 eV/Å`; broad surface-DFT literature [^4a]. |
| **Molecule (≤ 50 atoms) for vib/IR/TS/NEB** | **0.001 eV/Å max-force** | Vibrational analysis needs forces below the lowest physical mode's noise floor (~1 cm⁻¹); IR intensities need stable Hessian eigenvectors. | Gaussian `OPT=Tight` / geomeTRIC `GAU_TIGHT` [^4b]. |
| **Molecule production (≤ 50 atoms, energetics + geometry only)** | **0.04 eV/Å max-force (SIESTA) / 0.023 eV/Å (PySCF)** | The "publishable" tier — what Gaussian's `OPT` default gives + what reviewers expect for non-vib papers. | Gaussian-OPT default [^4]. |

**Why two tiers under one name was the bug.**  The old single
"tight" label conflated two regimes that need different numbers.
A 444-atom Au junction "tight" at 0.001 eV/Å is wrong (won't
converge); a small-molecule "tight" at 0.01 eV/Å is also wrong
(insufficient for vib).  Splitting the tier explicitly and
labeling the system-type axis fixes this.

**Cross-engine caveat (per the §2.3 footnote below).**  PySCF /
geomeTRIC checks all 5 convergence criteria (gmax / grms / dmax /
drms / etol — AND).  SIESTA checks only `MD.MaxForceTol`.  At the
same numerical max-force threshold, a PySCF "converged" geometry
is generally tighter than a SIESTA "converged" one.  When picking
matching thresholds across engines for a cross-validation, expect
PySCF to take more iterations to declare success.

**Default-value implementation.**  molbuilder PySCF #534 stage3
(the disabled-by-default "tight" tier in `_default_stages()`)
carries the crystal/surface production values as of 2026-06-23:
`gmax 2 × 10⁻⁴`, `grms 1 × 10⁻⁴` (Ha/Bohr), `dmax 1 × 10⁻³`,
`drms 5 × 10⁻⁴` (Å), `etol 1 × 10⁻⁶` (Hartree).  Users targeting
molecule vib/IR work override these explicitly via the form's
stage-table widget or `--stages-json`; see §2.4 for the full per-
knob value table including the molecule-vib (very-tight) column.

The SIESTA generator emits the same tier guidance in the FDF's
verbose comments (`siesta/input.py::_emit_md_block`).  When SIESTA
staged-opt lands (#542), per-stage `SiestaStageSpec` defaults will
mirror this same framework — stage1 = CG loose, stage2 = Broyden
publishable, stage3 = Broyden tight (crystal-practical).

---


| Tier | Value | Rationale |
|---|---|---|
| screening | 0.10 eV/Å | Within 1% of a typical bond force; geometry "looks right." |
| loose preopt | 0.05 eV/Å | Default SIESTA `MD.MaxForceTol` is 0.04 — close enough. |
| publishable | **0.04 eV/Å (SIESTA) / 4.5 × 10⁻⁴ Ha/Bohr ≈ 0.023 eV/Å (PySCF)** | Gaussian-OPT default; what papers cite without explanation.  [^4] |
| **tight (crystal/surface production)** | **0.01 eV/Å / ≈ 2 × 10⁻⁴ Ha/Bohr** | **Community-standard production threshold for metals / interfaces / large unit cells.  Matches VASP `EDIFFG = -0.01` (the de-facto solid-state convention) and Quantum ESPRESSO's tight `forc_conv_thr 2e-4 Ry/Bohr ≈ 0.005 eV/Å`.  Safe for 100s-of-atoms systems where Gaussian's GAU_TIGHT would chase SCF noise.** [^4a] |
| very-tight (molecule vib/IR/TS/NEB) | 0.001 eV/Å / 1.5 × 10⁻⁵ Ha/Bohr | **Gaussian's `GAU_TIGHT` preset.  Defensible Hessian eigenvalues + IR intensities on small molecules; NEB barrier heights to chemical accuracy.  DO NOT use for crystal/surface systems — for 100+ atom metals this never reaches the SCF noise floor and the run never converges.** [^4b] |

[^4a]: VASP documentation, "EDIFFG" tag — default negative-value
    convention is force tolerance in eV/Å.  For Au(111) surface +
    adsorbate studies the canonical practice is EDIFFG = -0.01
    (max-force ≤ 0.01 eV/Å); see e.g. Hammer + Nørskov,
    *Adv. Catal.* **45**, 71 (2000) and the broad surface-DFT
    literature.  Quantum ESPRESSO's tight `forc_conv_thr =
    2 × 10⁻⁴ Ry/Bohr` converts to ≈ 0.005 eV/Å.
[^4b]: Schlegel, "Geometry optimization" *WIREs* **1**, 790 (2011)
    §3 documents the Gaussian `OPT=Tight` (`GAU_TIGHT`) criteria as
    a vibrational-analysis prerequisite; the multi-100-atom crystal
    case is explicitly out of scope.

**Engine keywords:**

- SIESTA: `MD.MaxForceTol` (single criterion — only the max
  unconstrained force is checked).
- PySCF / geomeTRIC: **five criteria, all must pass** —
  `convergence_gmax` (max grad), `convergence_grms` (RMS grad),
  `convergence_dmax` (max disp), `convergence_drms` (RMS disp),
  `convergence_energy` (energy step).  This is modelled on
  Gaussian's `OPT` convention. [^5]

At the same numerical threshold a PySCF "converged" structure is
generally tighter than a SIESTA "converged" one (SIESTA checks
one, PySCF checks five).

[^5]: Schlegel, H. B. "Geometry optimization." *WIREs Comput. Mol. Sci.* **1**, 790 (2011).

---

### 2.4 geomeTRIC RMS gradient / displacement (PySCF only)

The five-criteria check from §2.3 has these supplementary
companions to `gmax`.  Default values mirror Gaussian's `OPT` set.

| Knob | Default (publishable) | **Tight (crystal/surface, molbuilder #534 stage3)** | Very-tight (molecule vib/IR — opt-in via override) | Units |
|---|---|---|---|---|
| `convergence_gmax` | 4.5 × 10⁻⁴ | **2.0 × 10⁻⁴** | 1.5 × 10⁻⁵ | Ha/Bohr |
| `convergence_grms` | 3.0 × 10⁻⁴ | **1.0 × 10⁻⁴** | 1.0 × 10⁻⁵ | Ha/Bohr |
| `convergence_dmax` | 1.8 × 10⁻³ | **1.0 × 10⁻³** | 6.0 × 10⁻⁵ | Å |
| `convergence_drms` | 1.2 × 10⁻³ | **5.0 × 10⁻⁴** | 4.0 × 10⁻⁵ | Å |
| `convergence_energy` | 1.0 × 10⁻⁶ | 1.0 × 10⁻⁶ | 1.0 × 10⁻⁶ | Hartree |

The publishable set is geomeTRIC's `GAU` preset (Gaussian-OPT
default).  The **Tight column is the molbuilder #534 stage3 default
as of 2026-06-23** — community-standard production thresholds for
crystal / surface / interface DFT (max-force ≈ 0.01 eV/Å, matching
VASP `EDIFFG=-0.01`).  The Very-tight column is geomeTRIC's
`GAU_TIGHT` preset (gradients tighten 10×, displacements 20×) and
is **molecule-only** territory — opt in explicitly via the form's
stage-table or `--stages-json` for vib/IR/TS/NEB work on small
molecules.  Using GAU_TIGHT on a 100+ atom metal system reliably
chases SCF noise and never converges. [^6]

Pre-2026-06-23 stage3 carried the GAU_TIGHT values — surfaced as a
real bug during BDT/Au(111) junction debugging when the user asked
"is this practical for large crystal systems?".  The realignment
keeps GAU_TIGHT reachable (via user override) but no longer ships
as the default tight tier.

**All five criteria flow end-to-end through molbuilder's PySCF
staged-opt** (since #534 commit 7a, 2026-06-23).  Per-stage values
of `gmax`/`grms`/`dmax`/`drms`/`etol` reach the rendered script's
`STAGES = [...]` literal (geomeTRIC consumes them via
`optimize(...)` kwargs) AND the generated `.molwatch.log` header's
`_CONVERGENCE_TARGETS` nested dict.  The Results-tab trajectory
inspector reads the nested dict and draws per-stage threshold
lines on the force / RMS-force / max-displ / RMS-displ / energy-
step plots; no value the user sets in the stage table is silently
dropped before reaching the visualization.

[^6]: Wang, L.-P. & Song, C. "Geometry optimization made simple with translation and rotation coordinates." *J. Chem. Phys.* **144**, 214108 (2016).

---

### 2.5 SCF tolerance

**What it controls.**  How tightly the *electronic* problem is
solved at each *geometry* step.  Forces are derived from the
converged density — sloppy SCF gives noisy forces, which makes the
optimizer thrash.

**Rule of thumb**: the SCF tolerance should be ~10× tighter than
the force-precision target you want at the end.  E.g. publishable
force tolerance ≈ 0.04 eV/Å ≈ 10⁻³ Ha/Bohr → SCF needs to give
forces stable to ~10⁻⁴ Ha/Bohr → SCF tolerance ~ 10⁻⁹ Ha (energy)
or 10⁻⁴ (density-matrix delta).

| Tier | SIESTA (`DM.Tolerance`, dimensionless) | PySCF (`scf_conv_tol`, Ha) |
|---|---|---|
| screening | 1 × 10⁻³ | 1 × 10⁻⁷ |
| loose preopt | 1 × 10⁻⁴ | 1 × 10⁻⁷ |
| publishable | **1 × 10⁻⁴** | **1 × 10⁻⁹** |
| tight | 1 × 10⁻⁵ | 1 × 10⁻¹⁰ |

**Why looser SCF on warm-up.**  When forces are 1 eV/Å (typical
preopt), tightening SCF from 10⁻⁷ to 10⁻⁹ Ha buys you nothing —
the geometry step is dominated by the gradient direction, not its
precision.  Tightening matters as `force ≪ 0.1 eV/Å`.

[^7]: Pulay, P. & Fogarasi, G. "Geometry optimization in redundant internal coordinates." *J. Chem. Phys.* **96**, 2856 (1992).

---

### 2.6 Real-space mesh cutoff (SIESTA only)

**What it controls.**  SIESTA discretises the Hartree + XC
potentials on a 3D real-space grid.  `MeshCutoff` sets the grid
spacing via the plane-wave-equivalent kinetic-energy cutoff.

| Tier | `MeshCutoff` (Ry) | Rationale |
|---|---|---|
| screening | 150 | Sanity-check only. |
| loose preopt | 200–250 | Forces accurate to ~1%. |
| publishable | **350** | Forces converged to better than 0.01 eV/Å on organic + Au systems. |
| tight (vib/phonons) | 500 (or 600 for first-row elements) | Mesh-egg-box noise below 0.001 eV/Å. |

The exact converged value depends on the basis sets in use — DZP
NAOs converge faster than long-tail PAOs.  Test by varying
`MeshCutoff` by ±50 Ry; the relative geometry should be stable to
within your tolerance. [^8]

[^8]: Soler, J. M. *et al.* "The SIESTA method for *ab initio* order-N materials simulation." *J. Phys.: Condens. Matter* **14**, 2745 (2002).

---

### 2.7 k-grid (SIESTA periodic systems)

**What it controls.**  Brillouin-zone sampling for periodic
systems.  Number of k-points needed depends on cell size + electronic
structure (metal vs insulator).

| System type | Recipe |
|---|---|
| Molecule in vacuum (>10 Å vacuum padding) | **Γ only** (`1 1 1`).  All other points are equivalent by translation. |
| Organic-on-metal junction (BDT/Au(111)) | **4 × 4 × 1** for screening, **6 × 6 × 1** publishable, **8 × 8 × 1** for current-voltage characteristics. |
| Bulk metal | **12 × 12 × 12** publishable; Monkhorst-Pack symmetry-reduced. [^9] |
| Bulk semiconductor / insulator | **6 × 6 × 6** publishable.  Gap requires fewer k-points than DOS. |

The mantra: the k-spacing × the lattice constant should be ~0.04 Å⁻¹
for publishable accuracy.

[^9]: Monkhorst, H. J. & Pack, J. D. "Special points for Brillouin-zone integrations." *Phys. Rev. B* **13**, 5188 (1976).

---

### 2.8 Basis set (PySCF; SIESTA uses NAOs by default)

| Tier | Basis | Rationale |
|---|---|---|
| screening | `def2-SVP` | Gives ~30% errors on bond lengths in conjugated systems; never publish. |
| loose preopt | `def2-SVP` | Same as screening — basis switch costs more than it buys. |
| publishable | **`def2-TZVP`** | The modern standard for organic chemistry; ECPs bundled for heavy elements up to Rn. [^10] |
| tight | `def2-TZVPP` or `def2-QZVP` | For energy comparisons across structures; final single-point after publishable-tier geometry. |

**RI-J auxiliary basis.**  For *every* hybrid-DFT calculation
(B3LYP, PBE0, M06-2X, ωB97M-V, ...) molbuilder defaults to
`mf = mf.density_fit(auxbasis="def2-universal-jfit")`.  This gives
~5–10× SCF speedup at <0.1 kcal/mol error. [^11]

[^10]: Weigend, F. & Ahlrichs, R. "Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H to Rn." *Phys. Chem. Chem. Phys.* **7**, 3297 (2005).
[^11]: Weigend, F. "Accurate Coulomb-fitting basis sets for H to Rn." *Phys. Chem. Chem. Phys.* **8**, 1057 (2006).

---

### 2.9 Functional + dispersion (DFT)

This is **not** tier-dependent — pick once based on the chemistry,
keep it across all stages.  Tier varies the convergence, not the
level of theory.

| Choice | When | Reference |
|---|---|---|
| **`b3lyp` + `mf.disp = "d3bj"`** | Default for organic chemistry; the most-cited combo of the last decade. | [^12] [^13] |
| `wb97m-v` | Charge-transfer character (donor-acceptor, transport junctions, π-stacked complexes).  Range-separated meta-GGA + VV10 nonlocal correlation.  Ships dispersion in the functional definition — no separate `mf.disp`. | [^14] |
| `pbe0` | Non-Becke alternative; common in solid-state work. | Adamo & Barone 1999. |
| `m06-2x` | Non-covalent interactions; thermochemistry benchmarks. | Zhao & Truhlar 2008. |
| `r²scan` + `mf.disp = "d3bj"` | Newer meta-GGA at ~B3LYP accuracy with GGA cost; emerging standard. | Furness *et al.* 2020. |

**Plain `b3lyp` without dispersion is no longer publishable** for
any molecule > 10 atoms.  Reviewers will flag it.

**Spelling note.**  The strings above are the *exact* tokens PySCF
2.x's `parse_dft` accepts.  In paper prose write the conventional
form ("B3LYP-D3(BJ)", "ωB97M-V"); in generated code write the PySCF
token.  The publication guide § "Functional" documents the
parenthesised-vs-split spelling traps in detail.

[^12]: Becke, A. D. *J. Chem. Phys.* **98**, 5648 (1993).
[^13]: Grimme, S. *et al.* "Effect of the damping function in dispersion corrected density functional theory." *J. Comp. Chem.* **32**, 1456 (2011).  And: Grimme, S. *et al.* *J. Chem. Phys.* **132**, 154104 (2010) for D3.
[^14]: Mardirossian, N. & Head-Gordon, M. *J. Chem. Phys.* **144**, 214110 (2016).

---

### 2.10 Maximum iteration counts

**What it controls.**  Per-stage budget for the OUTER (geometry)
loop.  Per-stage budget for the INNER (SCF) loop.

| Knob | Tier | Value | Rationale |
|---|---|---|---|
| **Geometry steps** | loose preopt | 50 | If the warm-up needs more, the starting geometry is wrong; stop and inspect. |
| | publishable | 200 | Typical organic relax converges in 30–80 steps; 200 is the safety margin. |
| | tight | 100 | Starts from a publishable-converged geometry; only fine-tunes. |
| **SCF iterations** | All tiers | 100 | Should be plenty for any well-conditioned SCF.  If you hit 100 without converging, the *system* is the problem (broken-symmetry open-shell, level-shift needed, init_guess wrong) — bumping max_iter to 500 doesn't help. |

**Engine keywords:**

- SIESTA: `MD.NumCGsteps` / `MD.NumBroydenSteps` / `MD.NumFIRESteps`
  per `MD.TypeOfRun`; `MaxSCFIterations` for inner loop.
- PySCF: `STAGE['max_steps']` in the per-stage table; `mf.max_cycle`
  for inner SCF.

---

## 3. Cross-engine parameter map

For users who know one engine and want the equivalent in the other.

| Concept | SIESTA | PySCF / geomeTRIC | Units |
|---|---|---|---|
| Geometry algorithm | `MD.TypeOfRun` | `cfg.optimizer` (`"geometric"` ≈ BFGS) | enum |
| Force convergence | `MD.MaxForceTol` | `convergence_gmax` | SIESTA eV/Å; PySCF Ha/Bohr |
| RMS-grad convergence | *(not checked by default)* | `convergence_grms` | Ha/Bohr |
| Disp convergence | *(implicit via MaxCGDispl)* | `convergence_dmax` / `convergence_drms` | Å |
| Energy-step convergence | *(implicit via SCF tol)* | `convergence_energy` | Hartree |
| Max step | `MD.MaxCGDispl` / `MD.MaxDispl` | *(geomeTRIC-internal line search)* | Å |
| SCF tolerance | `DM.Tolerance` | `mf.conv_tol` | dimensionless / Hartree |
| Max geom steps | `MD.NumCGsteps` etc. | `STAGE['max_steps']` (or geomeTRIC `maxsteps`) | integer |
| Max SCF cycles | `MaxSCFIterations` | `mf.max_cycle` | integer |
| Discretisation | `MeshCutoff` (Ry) | *(basis-determined)* | Ry / N/A |
| Basis | NAO via `PAO.Basis` (default DZP) | `cfg.basis` (Gaussian) | string |
| k-grid | `kgrid_Monkhorst_Pack` | *(gas-phase only by default)* | tuple |

---

## 4. Tier preset summary table

The values that drop straight into the molbuilder PySCF stage-table
widget (and, when SIESTA staged-opt ships, will drop into its
analog).

| Knob | Screening | Loose preopt | **Publishable** | Tight (vib/IR) |
|---|---|---|---|---|
| Optimizer (SIESTA) | CG | CG | **Broyden** | Broyden |
| Optimizer (PySCF) | geometric | geometric | **geometric** | geometric |
| `MaxForceTol` (eV/Å, SIESTA) | 0.10 | 0.05 | **0.04** | 0.01 |
| `gmax` (Ha/Bohr, PySCF) | 2 × 10⁻³ | 2 × 10⁻³ | **4.5 × 10⁻⁴** | 1.5 × 10⁻⁵ |
| `MaxCGDispl` (Å, SIESTA) | 0.30 | 0.20 | **0.05** | 0.02 |
| `dmax` (Å, PySCF) | 1 × 10⁻² | 7.2 × 10⁻³ | **1.8 × 10⁻³** | 6 × 10⁻⁵ |
| SCF tol (PySCF, Ha) | 1 × 10⁻⁷ | 1 × 10⁻⁷ | **1 × 10⁻⁹** | 1 × 10⁻¹⁰ |
| `DM.Tolerance` (SIESTA) | 1 × 10⁻³ | 1 × 10⁻⁴ | **1 × 10⁻⁴** | 1 × 10⁻⁵ |
| `MeshCutoff` (SIESTA, Ry) | 150 | 200 | **350** | 500 |
| Basis (PySCF) | def2-SVP | def2-SVP | **def2-TZVP** | def2-TZVPP |
| Max geom steps | 30 | 50 | **200** | 100 |

The **PySCF stage-table defaults** (`_default_stages()` in
`molbuilder/config/pyscf.py`) already encode rows for *loose preopt*
(stage1) / *publishable* (stage2) / *tight* (stage3, disabled by
default).  See § 5.

---

## 5. Restart / continuation strategy

| Scenario | SIESTA | PySCF |
|---|---|---|
| Resume an interrupted relaxation from the last accepted step | `MD.UseSaveCG .true.` + `MD.UseSaveXV .true.` (KEEP both — CG history + geometry) | Re-run; `_mb_outfile(JOB + ".chk")` + the chkfile-init shim molbuilder always emits |
| Restart same stage with new geometry but reset optimizer history | `MD.UseSaveXV .true.` + `MD.UseSaveCG .false.` (KEEP geometry, RESET CG basis) | Drop the chkfile or use `--cold` flag (runwrap) |
| Switch optimizer (CG → Broyden) on stage boundary | Use the appropriate `MD.Use*Save*` for the new optimizer; the previous one's history file is now stale | N/A — only geometric supported in stages |
| Diff-restart for diagnostics | Copy the run dir; do NOT carry forward `.CG` / `.BR` history — they're per-trajectory | Same |

**Why "keep CG history" on a continuation but "reset" on a tier
switch.**  CG's conjugate basis is built up over moves; throwing
it out wastes ~5–10 moves of warm-up.  But when you tighten
`MD.MaxCGDispl` (or switch CG → Broyden), the old conjugate basis
is *poisoned* by the over-shoot pattern that triggered the tier
switch in the first place — keeping it re-creates the same
oscillation.

---

## 6. Worked decision: BDT/Au(111) junction

Walking through the 2026-06-23 case with the framework above:

| Question | Answer | Why |
|---|---|---|
| What's the target use? | Transport (NEGF on relaxed geometry) | → **publishable** tier for the relax (the NEGF itself is its own quality axis) |
| Material class? | Organic-on-metal interface, vdW component | → Broyden (§ 2.1); not CG |
| Cell? | 4 × 4 × 1 surface supercell | → 6 × 6 × 1 k-grid (§ 2.7) |
| Current state? | CG oscillating 0.09–0.5 eV/Å for 20+ moves | → optimizer + step-cap are wrong; not the threshold |
| Recommended stage-2 fdf | `MD.TypeOfRun Broyden`, `MD.MaxCGDispl 0.02 Å`, `MD.MaxForceTol 0.04 eV/Å`, `DM.Tolerance 1e-4`, `MeshCutoff 350 Ry` | All from § 4 publishable column. |

---

## References

(In citation order; see the footnote marker in the relevant
section for context.)

1. Hestenes & Stiefel 1952; Polak & Ribière 1969 — CG algorithm.
2. Johnson, D. D. *Phys. Rev. B* **38**, 12807 (1988) — modified Broyden.
3. Bitzek *et al.* *Phys. Rev. Lett.* **97**, 170201 (2006) — FIRE.
4. Frisch *et al.* — Gaussian 16 `OPT` keyword reference.
5. Schlegel, H. B. *WIREs Comput. Mol. Sci.* **1**, 790 (2011) — geometry-optimization review.
6. Wang & Song *J. Chem. Phys.* **144**, 214108 (2016) — geomeTRIC.
7. Pulay & Fogarasi *J. Chem. Phys.* **96**, 2856 (1992) — redundant internals.
8. Soler *et al.* *J. Phys.: Condens. Matter* **14**, 2745 (2002) — SIESTA method paper.
9. Monkhorst & Pack *Phys. Rev. B* **13**, 5188 (1976) — k-grid sampling.
10. Weigend & Ahlrichs *Phys. Chem. Chem. Phys.* **7**, 3297 (2005) — def2-TZVP family.
11. Weigend *Phys. Chem. Chem. Phys.* **8**, 1057 (2006) — def2-universal-jfit auxiliary.
12. Becke *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP.
13. Grimme *et al.* *J. Comp. Chem.* **32**, 1456 (2011) — D3-BJ.  Grimme *et al.* *J. Chem. Phys.* **132**, 154104 (2010) — D3.
14. Mardirossian & Head-Gordon *J. Chem. Phys.* **144**, 214110 (2016) — ωB97M-V.

PySCF: Sun, Q. *et al.* *J. Chem. Phys.* **153**, 024109 (2020).

---

## See also

* [`pyscf-publication-guide.md`](pyscf-publication-guide.md) — the PySCF-specific publishable recipe + methods-section template.  This guide subsumes its parameter-tuning table in § 4; the publication-guide retains the methods-section text + functional / basis caveats specific to PySCF.
* [`pyscf.md`](pyscf.md) — what the generated PySCF script *does* (output files, log contract, stages-loop shape).
* [`siesta.md`](siesta.md) — SIESTA script-contract.
* [`../protocols/scientific-validation.md`](../protocols/scientific-validation.md) — the broader scientific-correctness review process.
