# PySCF — publication-quality parameter guide + staged-opt design

**Audience**: someone running a molbuilder-generated PySCF script for a paper, AND the maintainer planning the staged-optimization feature (task #534).

**Captured 2026-06-20** during the PDT (pentanedithiol) discussion.
Don't lose this on context compaction — the parameter table + the
methods-section text below are the load-bearing artefacts.

---

## TL;DR — what to change in a current molbuilder PySCF script for publication quality

The current generator defaults are screening-tier (`def2-SVP`, `B3LYP`
with no dispersion).  For a defensible paper on **simple organic
molecules** (PDT-class: 10–60 atoms, ground-state geometry), edit
the generated `pyscf_relax.py` like this:

```python
# In gto.M(...) — line ~258 in current generator output
mol = gto.M(
    ...
    basis      = "def2-TZVP",   # was "def2-SVP" — THE single most important upgrade
    ...
)

# In your mf setup (where scf.RKS or scf.UKS is constructed):
mf = scf.RKS(mol).density_fit(auxbasis="def2-universal-jfit")  # RI-J, 5–10× speedup, no accuracy loss
mf.xc = "B3LYP-D3(BJ)"   # was "B3LYP" — Grimme dispersion correction
mf.conv_tol = 1e-9        # already at publication standard
mf.max_cycle = 100        # already plenty
# convergence_gmax = 4.5e-4 Ha/Bohr (geomeTRIC default) — already publication standard
```

That's the minimum-viable change.  Convergence thresholds in the
current script are already Gaussian-OPT defaults — what reviewers
expect.

---

## Parameter tiers (Gaussian convention, the de-facto standard)

| Knob | Default (publishable) | TIGHT (vib/IR/NEB) | VERY-TIGHT (kinetics) |
|---|---|---|---|
| `convergence_energy` (Ha) | 1e-6 | 1e-7 | 1e-8 |
| `convergence_gmax` (Ha/Bohr) | **4.5e-4** (≈ 0.023 eV/Å) | 1.5e-5 | 2e-6 |
| `convergence_grms` (Ha/Bohr) | 3.0e-4 | 1.0e-5 | 1e-6 |
| `convergence_dmax` (Bohr) | 1.8e-3 | 6.0e-5 | 6e-6 |
| `convergence_drms` (Bohr) | 1.2e-3 | 4.0e-5 | 4e-6 |
| `mf.conv_tol` (Ha) | 1e-9 | 1e-10 | 1e-11 |

The "default" column = Gaussian's OPT default = geomeTRIC's default.
>90% of published organic-chemistry structures use these.  Reviewers
don't question them.

Escalate to TIGHT when you also need **vibrational analysis**,
**IR/Raman frequencies**, **zero-point energies for thermochemistry**,
or any **transition-state search**.  VERY-TIGHT for NEB barrier
heights to chemical accuracy (~1 kcal/mol).

---

## Basis + functional choices

### Basis set
| Tier | Choice | Use case |
|---|---|---|
| Screening / debug | `def2-SVP` | first pass on a hand-built structure; never publish |
| **Publishable** | **`def2-TZVP`** | the modern standard for organic chemistry |
| High accuracy | `def2-TZVPP` or `def2-QZVP` | for energy comparisons across structures; final single-point after geometry |
| ECP-required (heavy atoms) | `def2-TZVP` (built-in ECP) | covers all elements up to Rn |

`def2-SVP` gives ~30% errors on bond lengths in conjugated systems;
`def2-TZVP` is the floor for credible work.

### Density-fitting auxiliary
Always use RI-J for non-hybrid + hybrid DFT:
```python
mf = scf.RKS(mol).density_fit(auxbasis="def2-universal-jfit")
```
~5–10× speedup; errors <0.1 kcal/mol; no reason to skip it for
production.

### Functional
| Choice | When |
|---|---|
| **`B3LYP-D3(BJ)`** | most organic chemistry; the most-cited combo of the last decade |
| `ωB97X-D` | when charge-transfer character is significant (donor–acceptor, charge transport, π-stacked complexes).  D included; no need for explicit -D3. |
| `PBE0` | when you want a non-Becke alternative; common in solid-state work |
| `M06-2X` | non-covalent interactions; thermochemistry benchmarks |
| `r²SCAN-D3(BJ)` | newer choice; meta-GGA; ~B3LYP accuracy at GGA cost — emerging standard |

Plain `B3LYP` (no dispersion) is no longer publishable for any
molecule >10 atoms.  Reviewers will flag it.

### When you also need vibrational analysis
- Re-optimize the final structure with TIGHT criteria above.
- After convergence, run `mf.Gradient().kernel()` then a Hessian
  (`Freq.kernel()` in PySCF) — but that's a separate post-opt step,
  not in the molbuilder geometry script today.

---

## Methods-section template

Paste-and-edit version for the paper:

> Geometry optimizations were carried out at the
> **B3LYP-D3(BJ)/def2-TZVP** level of theory using PySCF [1] with
> the geomeTRIC optimizer [2].  Gaussian's default convergence
> criteria were applied (gmax = 4.5 × 10⁻⁴ Ha/Bohr, grms = 3.0 × 10⁻⁴
> Ha/Bohr, dmax = 1.8 × 10⁻³ Bohr, drms = 1.2 × 10⁻³ Bohr,
> ΔE = 1 × 10⁻⁶ Ha).  The SCF self-consistency threshold was set to
> 1 × 10⁻⁹ Ha.  Resolution-of-the-identity Coulomb-fitting (RI-J) [3]
> with the def2-universal-jfit auxiliary basis was applied for
> computational efficiency.

### Citations to include

| Component | Cite |
|---|---|
| PySCF | Sun, Q. *et al.* "Recent developments in the PySCF program package." *J. Chem. Phys.* **153**, 024109 (2020). DOI: 10.1063/5.0006074 |
| geomeTRIC | Wang, L.-P. & Song, C. "Geometry optimization made simple with translation and rotation coordinates." *J. Chem. Phys.* **144**, 214108 (2016). DOI: 10.1063/1.4952956 |
| B3LYP | Becke, A. D. *J. Chem. Phys.* **98**, 5648 (1993); Lee, Yang, Parr, *Phys. Rev. B* **37**, 785 (1988). |
| D3(BJ) | Grimme, S. *et al.* "Effect of the damping function in dispersion corrected density functional theory." *J. Comp. Chem.* **32**, 1456 (2011). DOI: 10.1002/jcc.21759 |
| def2-TZVP | Weigend, F. & Ahlrichs, R. *Phys. Chem. Chem. Phys.* **7**, 3297 (2005). DOI: 10.1039/B508541A |
| RI-J auxiliary basis | Weigend, F. *Phys. Chem. Chem. Phys.* **8**, 1057 (2006). DOI: 10.1039/B515623H |
| ωB97X-D (if used) | Chai, J.-D. & Head-Gordon, M. *Phys. Chem. Chem. Phys.* **10**, 6615 (2008). |

---

## SIESTA ↔ PySCF cross-engine equivalence

For users who know SIESTA and want the same convergence rigor in PySCF:

| Layer | What it controls | SIESTA knob | PySCF/geomeTRIC equivalent |
|---|---|---|---|
| **SCF** (per geom step) | electronic "solved enough" for forces | `DM.Tolerance` (DM max change) | `mf.conv_tol` (energy, Ha) + `mf.conv_tol_grad` (orbital gradient) |
| **Geometry** (across steps) | when forces / displacements small enough to stop | `MD.MaxForceTol` (eV/Å) | `convergence_grms` / `convergence_gmax` (Ha/Bohr) + `convergence_drms` / `convergence_dmax` (Bohr) + `convergence_energy` (Ha) |

| Tier | SIESTA `MD.MaxForceTol` | PySCF `convergence_gmax` |
|---|---|---|
| Loose (debug) | 0.05 eV/Å | 2e-3 Ha/Bohr |
| **Publishable** | **0.04 eV/Å** | **4.5e-4 Ha/Bohr** (Gaussian default; ≈ 0.023 eV/Å) |
| TIGHT (vib, TS) | 0.01 eV/Å | 1.5e-5 Ha/Bohr |
| VERY-TIGHT (NEB) | 0.001 eV/Å | 2e-6 Ha/Bohr |

PySCF is **stricter** than SIESTA at the same tier — geomeTRIC
requires **5 criteria** (energy + rms grad + max grad + rms step +
max step) to ALL be satisfied (modelled on Gaussian's OPT
convention).  SIESTA only checks **max force** by default.  At the
same numerical threshold, a PySCF "converged" structure is
generally tighter than a SIESTA "converged" one.

---

## In-script staged optimization — design plan (task #534)

PySCF has an architectural advantage over SIESTA: we generate the
runnable script, so we can put a multi-stage loop INSIDE it.  No
manual "run stage1, then stage2" cycles.

### Design agreed with user (2026-06-20)

* Each PySCF script generates **3 stages** internally.
* Each stage can be **enabled / skipped** independently from the UI.
* Per-stage outputs use `<JOB>_stageN_*` suffix so the Results-tab
  file picker shows them as discrete files.
* Geometry from stage N feeds stage N+1 via in-Python `mol` carry-
  over (no disk roundtrip; no manual file rename).

### Parameter defaults (3-stage, simple organic molecules)

| Stage | Default-enabled | `mf.conv_tol` | `convergence_gmax` (Ha/Bohr) | `max_steps` | Purpose |
|---|---|---|---|---|---|
| 1 (loose pre-opt) | True | 1e-7 | 2.0e-3 | 50 | Get the structure roughly right cheaply |
| 2 (**publishable**, Gaussian default) | True | 1e-9 | 4.5e-4 | 200 | What papers cite |
| 3 (TIGHT, vib/IR/NEB) | False (opt-in) | 1e-10 | 1.0e-4 | 100 | Only when you need accurate Hessians |

Stage 3's `grms`/`dmax`/`drms`/`etol` scale together at 10× tighter
than Stage 2 per Gaussian-TIGHT convention.

### UI shape

Per-parameter inline stage controls with a "Stage strategy" preset
dropdown that fills sensible defaults; "Custom" mode for full
control.  Example:

```
Force tolerance (Ha/Bohr)
  Stage 1  [☑]  2.0e-3   (loose pre-opt)
  Stage 2  [☑]  4.5e-4   (publishable)   ← Gaussian default
  Stage 3  [☐]  1.0e-4   (tight, vib/IR)
```

Most users tick stages 1+2, leave 3 disabled.  Power users custom-
set any value per stage.

### Generated-script sketch

```python
STAGES = [
    {"name": "stage1", "enabled": True,  "conv_tol": 1e-7,  "gmax": 2.0e-3, "max_steps":  50},
    {"name": "stage2", "enabled": True,  "conv_tol": 1e-9,  "gmax": 4.5e-4, "max_steps": 200},
    {"name": "stage3", "enabled": False, "conv_tol": 1e-10, "gmax": 1.0e-4, "max_steps": 100},
]

mol = gto.M(atom=..., basis=..., ...)
for stage in STAGES:
    if not stage["enabled"]:
        continue
    mf = scf.RKS(mol).density_fit(auxbasis="def2-universal-jfit")
    mf.xc = "B3LYP-D3(BJ)"
    mf.conv_tol = stage["conv_tol"]
    prefix = f"{JOB}_{stage['name']}"
    _molwatch = MolwatchEmitter(f"{prefix}.molwatch.log", prefix, mol,
                                convergence_targets={...stage targets...})
    mol = optimize(mf, prefix=f"{prefix}_geom",
                   convergence_gmax=stage["gmax"],
                   convergence_grms=stage["gmax"] * 0.67,
                   convergence_dmax=stage["dmax"],
                   convergence_drms=stage["drms"],
                   convergence_energy=stage["etol"],
                   maxsteps=stage["max_steps"])
    # mol updated in place by optimize(); carries forward to next stage
```

### Implementation touches

* `molbuilder/pyscf/input.py` — generator loops over enabled stages,
  threads `mol` forward, names outputs `<JOB>_stageN_*`
* `molbuilder/config/pyscf.py` (or equivalent) —
  `PySCFConfig.stages: List[StageSpec]` dataclass
* Web blueprint + form schema for per-stage controls
* JS for "Stage strategy" preset dropdown + per-stage enable/value rows
* `MolwatchEmitter` usage updated to take per-stage
  `convergence_targets` (the header carrying stage-specific values)
* L2 tests for per-stage script rendering, "skip stage" semantics,
  carry-over geometry
* Doc update: this file + `docs/engines/pyscf.md` for the per-stage
  output filename rule

### Why molbuilder needs this

Without staged optimization, the user has to manually re-edit the
script after stage 1 converges, change the tolerances, re-run.
That's error-prone (forgetting to tighten SCF, forgetting to change
output filenames so stage 1's `_geom_optim.xyz` isn't overwritten).
The in-script loop removes all that by construction; user clicks
"publishable 2-stage" and gets a finished, paper-ready geometry +
all intermediate files separately preserved.

---

## Why this lives here (not in design.md or roadmap.md)

* `design.md` is the project's high-level decisions log.  This is
  engine-specific user/maintainer guidance.
* `roadmap.md` is for items with timeline.  This isn't time-bound.
* `docs/engines/pyscf.md` is the PySCF script contract — what the
  script DOES.  This is what the user should CHOOSE.

Sibling docs to read next:
* `docs/engines/pyscf.md` — the script-output contract
* `docs/engines/siesta.md` — the SIESTA equivalent
* `docs/engines/transport.md` — TranSIESTA-specific guidance
* `docs/protocols/scientific-validation.md` — broader scientific-correctness review process
