# TranSIESTA workflow — scientific basis & a consistency-first strategy

> **Design doc.** Captures (a) the *received view* of the Au–BDT–Au
> transport workflow as assembled from a multi-agent discussion + standard
> practice, (b) a critical scientific assessment — what holds, what to
> correct, with justification/estimations/literature, and (c) the
> **molbuilder strategy**: derive all coupled runs from one descriptor and
> *enforce the cross-run consistency contract* that is where these
> calculations actually break.  Builds on the existing emitter
> (`molbuilder/transport/transiesta.py`), the `*-electrode` region
> convention (`region-labels.md`), and the modern `TS.Elec`/`TS.ChemPots`
> syntax (`docs/engines/transport.md`).

---

## § 0 Scope

The target is a publication-grade zero-bias conductance (and `T(E)`, then
finite-bias `I–V`) for a single-molecule junction: `Au(111) | BDT |
Au(111)`, BDT = 1,4-benzenedithiolate (dehydrogenated thiols, S–Au
contacts), 6×6 lateral Au(111), 6 layers/side (3 frozen "electrode" + 3
relaxed "interface"). The same logic applies to any two- or
multi-terminal junction.

---

## § 1 The physics in one page

A junction is an **open-boundary** problem: a finite scattering region C
between two semi-infinite leads L, R extending to ±∞ along transport (z).
TranSIESTA solves it with the **Non-Equilibrium Green's Function (NEGF)**
method [Brandbyge 2002; Papior 2017]. The leads enter C only through
energy-dependent **self-energies** Σ_{L,R}(E) = V_{Ci}(E·S_i − H_i)⁻¹V_{iC}
built from the *pristine bulk* lead Hamiltonian/overlap (H_i, S_i); the
device Green's function is G(E) = [E·S_C − H_C − Σ_L − Σ_R]⁻¹ and the
transmission T(E) = Tr[Γ_L G Γ_R G†], with zero-bias conductance
G = G₀·T(E_F), G₀ = 2e²/h.

**Three consequences that drive every parameter choice:**
1. Σ comes from a **separate, pristine bulk-lead calculation** — *not*
   from frozen atoms in the device (frozen = a geometry constraint only).
   Hence the workflow is **three runs**: relax → bulk electrode → NEGF.
2. The **transport direction is open** in the NEGF run → its k-grid is
   `kz = 1` (any `kz>1` re-imposes Bloch periodicity along the wire and
   destroys the transport physics).
3. **Geometry fixes the physical model; the k-grid only sets integration
   accuracy.** Adding lateral vacuum makes a *cluster* (a different
   Hamiltonian); a denser k-grid of the *same* geometry only integrates it
   better. These are independent axes.

---

## § 2 The received view (faithful summary of the discussion)

The multi-agent discussion converged on:
- **R1.** Three runs: (1) SIESTA relaxation of the full junction (outer 3
  Au layers/side frozen, inner 3 + BDT relaxed); (2) a *separate* pristine
  bulk-Au electrode run → `.TSHS`; (3) TranSIESTA on the relaxed geometry
  reading the electrode `.TSHS`.
- **R2.** `kz = 1` for the device; **zero vacuum in z**.
- **R3.** Transverse k for the device "≈ 4×4×1 for a 6×6 cell," with the
  later (correct) caveat that this must be *converged*, not assumed.
- **R4.** Geometry sets the model, k-grid sets accuracy (the cluster-vs-
  slab vs Γ-vs-dense distinction) — the strongest and most correct point.
- **R5.** Electrode = **3 Au(111) layers** (one ABC repeat), single-point
  (`MD.NumCGsteps 0`), a **geometric clone** of the device's frozen
  layers ("Golden Rule").
- **R6.** Electrode `kz`: the first pass said `4×4×~40`; the **final
  summary said `4×4×1`**.
- **R7.** Basis: **DZP** for C/H/S + interface Au; optionally **SZP** for
  bulk/frozen Au; "start DZP-everywhere, downgrade bulk Au only if
  needed." `MeshCutoff ≥ 300 Ry`.

R1, R2, R4 are correct and well-argued. R3 (with the convergence caveat),
R5-partial, and R7-ordering are mostly right. R5-thickness and R6 contain
the substantive errors below.

---

## § 3 Assessment — what holds (agree)

- **3-run NEGF structure (R1)** — exactly the TranSIESTA design
  [Brandbyge 2002; Papior 2017]. Frozen atoms ≠ self-energy. ✓
- **`kz = 1` for the NEGF run (R2)** — required by the open boundary. ✓
- **Geometry vs k-grid (R4)** — the load-bearing conceptual point, fully
  correct: vacuum changes H; k changes only BZ sampling. ✓
- **Converge the transverse k (R3, corrected form)** — metals have a sharp
  Fermi surface, so forces/E_F/T(E) are k-sensitive; `4×4×1` is a
  *candidate*, not a law. ✓
- **Geometric clone + single-point electrode (R5-partial)** — mandatory. ✓

---

## § 4 Assessment — corrections (with justification)

### § 4.1 Electrode thickness: 3 layers is likely too thin (largest risk)
TranSIESTA requires the electrode **principal layer** to couple *only* to
its immediate neighbor, i.e. the electrode cell along z must exceed the
range of the Hamiltonian/overlap (and, more strictly, the density-matrix)
matrix elements; otherwise the self-energy is wrong and the code's
electrode/Hartree-flatness check fails [Papior 2017; TranSIESTA manual].

**Estimation.** With the emitter's default `PAO.EnergyShift 0.01 Ry`
(`transiesta.py:374`), the diffuse Au 6s orbital has a confinement radius
r ≈ 3.2–3.7 Å, so the H/S element range ≈ 2r ≈ 6–7 Å. The Au(111)
interlayer spacing is a/√3 = 4.08/1.732 ≈ **2.355 Å**. Thus the overlap
already spans **~3 layers**, and the density-matrix range is longer. A
3-layer electrode (~4.7 Å, two interlayer gaps) lets boundary orbitals
reach the *next-nearest* image → non-zero second-neighbor coupling →
ill-defined principal layer.

**Correction.** The geometric repeat is 3 layers, but the *electronic*
principal layer for Au(111) DZP is typically **2 ABC units ≈ 6 layers**
(consistent with common junction setups). Do not hardcode 3; **size the
electrode from the orbital range** and confirm with TranSIESTA's electrode
check. Practical default: 6 frozen electrode layers/side (so the device
needs ≥6 electrode + a few interface layers).

### § 4.2 The electrode `kz` (R6): the final "4×4×1" is wrong
The electrode `.TSHS` run is a **periodic bulk** calculation — its z-axis
*is* real bulk gold, so it needs a **converged, dense `kz`**. A thin
3–6-layer cell has a large Brillouin zone along z → many points required.
The first pass's `4×4×~40` is the right spirit; the **final summary's
`4×4×1` under-samples the bulk lead and yields a wrong electrode
Hamiltonian.** (The "semi-infinite extension is wrt the electrode cell"
remark is true but separate — that is the Green's-function tiling, not the
SCF `kz`.) **Correct contract:** electrode = (transverse k *matching* the
device) × (converged dense `kz`, e.g. start 60–100, converge); device =
(same transverse k) × `kz = 1`.

### § 4.3 The "Golden Rule" must be a full *numerical contract*
The discussion requires geometric + basis identity between the electrode
and the device frozen layers, but **omits** that the **XC functional,
pseudopotentials, `MeshCutoff`, and `PAO.EnergyShift`** must *also* be
identical across the electrode and device runs. They must — otherwise H_L
and H_C are computed on different numerical footings and the Fermi-level
alignment / self-energy matching is invalid. Also, **`MeshCutoff 300 Ry`
is likely under-converged for Au**: the Pseudo-Dojo Au pseudo treats
**5s 5p 5d 6s** explicitly (semicore 5s5p; verified
`Au.psml: 5s2 5p6 5d10 6s1`), and semicore states are sharp → expect
**350–500 Ry**. Converge once, then **freeze the identical value** across
all three runs.

### § 4.4 Mixed SZP/DZP basis — a transport-specific risk none flagged
For total energy a basis "step" is benign. For *transport* the observable
*is* the transmission, and a **SZP→DZP discontinuity inside the metal acts
as a spurious scatterer** (a basis-set reflection that contaminates T(E)).
**Default to DZP everywhere**; downgrade bulk/frozen Au to SZP only after
confirming on a small test that (i) T(E) is unchanged and (ii) the SZP/DZP
boundary sits well behind the contact. The discussion's ordering ("start
DZP-everywhere") is right; I make it the firm default.

### § 4.5 Screening depth (a gap)
The electrostatic potential must reach its bulk value at the electrode
boundary, else Σ is applied where the molecule still perturbs the leads.
6 layers/side (3 frozen + 3 relaxed) is **marginal**; TranSIESTA reports
the boundary potential — **verify it is flat**, and add Au layers if not.

### § 4.6 Convergence is a requirement, not a number; and a level-alignment caveat
Every "magic" value above (transverse k, MeshCutoff, electrode thickness,
EnergyShift, vacuum-if-cluster) must be **convergence-tested**. Separately,
note for "publication quality": **plain LDA/GGA-DFT+NEGF systematically
overestimates single-molecule conductance** by ~1–2 orders of magnitude
because the DFT HOMO–LUMO gap is too small and level alignment to E_F is
off — for Au–BDT, experiment is ≈ **0.011 G₀** [Xiao 2004] while GGA-NEGF
commonly gives ~0.1–0.4 G₀. The geometry/k/electrode contract here gives a
*numerically correct* GGA-NEGF result; closing the gap to experiment needs
beyond-DFT corrections (self-energy/DFT+Σ, scissors, hybrid). The framework
should report this honestly, not imply DFT-NEGF == experiment.

---

## § 5 Recommended baseline (defensible start; all to be converged)

| Quantity | Baseline | Note |
|---|---|---|
| XC | GGA-PBE | identical across all 3 runs |
| Pseudos | Pseudo-Dojo PBE (Au/C/S/H), validated | [van Setten 2018]; `pseudo check` gate |
| Basis | **DZP everywhere** | SZP bulk-Au only after a T(E) check (§ 4.4) |
| `MeshCutoff` | 400 Ry (converge 300→500) | semicore Au (§ 4.3); identical across runs |
| `PAO.EnergyShift` | 0.01 Ry | sets orbital range → electrode thickness (§ 4.1) |
| Transverse k | converge 2×2 → 4×4 → 6×6 | identical & commensurate device⇄electrode |
| Device `kz` | **1** | open boundary |
| Electrode `kz` | converge (start ~80) | dense bulk z-sampling (§ 4.2) |
| Electrode thickness | **~6 Au(111) layers** | electronic principal layer (§ 4.1) |
| z-vacuum | **0** | slab-junction model; nonzero ⇒ cluster model |
| Force tol | 0.02 eV/Å | relaxation only |

---

## § 6 The molbuilder strategy

The discussion itself reveals the design principle: **TranSIESTA
correctness is a *consistency* problem.** One numerical contract + one
geometry must appear, intact, across three coupled runs, with commensurate
k, a geometric clone, and a thick-enough electrode. Humans break exactly
these couplings. So the framework's value is **not** emitting `.fdf` text
(the engine already does, with modern `TS.Elec` syntax) — it is to
**derive all runs from one descriptor and *enforce the cross-run
contract*.** This is the same philosophy as the benchmark workflow
(`benchmark-workflow.md`): one source of truth, contracts not hand-tuning,
versioned portable data.

### § 6.0 Data flow, file sets & I/O contracts (the exact design)

**Diagram — what each run consumes and produces (file level).** The single
numerical contract (§ 6.8) is baked, *identical*, into all three fdfs; only
the geometry and the open-vs-bulk boundary (`kz`, `SolutionMethod`) differ.

```
 SOURCE SET (user owns)            DERIVED RUNS (driver owns)                          TARGET SET (results)
 ─────────────────────            ─────────────────────────────────────              ─────────────────────
 device.molstruct.json ┐
   regions: L/R-electrode,│   ┌─▶ relax.fdf ──SCF/CG──▶ relaxed coords ──┐
   interface, frozen     ├──▶ │   (MD.CG, kz=1)          (device.XV)       │
 numerical contract      │   │                                            ├─▶ device.fdf ──NEGF──▶ device.TSHS ─┐
   {XC, MeshCutoff,      │   │                                            │   (kz=1, SolutionMethod  device.out  ├─▶ tbtrans
    EnergyShift, basis,  │   └─▶ electrode.fdf ─SCF────▶ electrode.TSHS ──┘    transiesta,           (E_F, SCF)  │   *.TBT.AVTRANS
    transverse k}        │       (clone of frozen layers,  (E_F^lead,           TS.Elec→electrode.TSHS)          ▼
 pseudopotentials (.psml)┘        dense kz, NumCGsteps 0,    H_lead, S_lead)                            transport-result.json
                                  SaveHS / TS.HS.Save)                                                  {E_F, G₀=T(E_F),
                                                                                                         conv status, caveat}
       │                                  │                                       │
       └── consistency PREFLIGHT ─────────┴── enforces the INVARIANT SET (§ 6.8) ─┘  before any binary runs
```

**Source set** (the *only* things a user authors): a region-labeled device
structure (`molstruct.json`, `region-labels.md` convention), the numerical
contract values, and the pseudopotential files. Everything below is derived.

**Intermediate set** (driver-owned, handed stage→stage): `relax.fdf` →
`device.XV` (relaxed coordinates); `electrode.fdf` → `electrode.TSHS` (the
bulk-lead Hamiltonian/overlap + its E_F). These are *consumed* by the device
run — the relaxed coords replace the device coordinate block, the
`electrode.TSHS` becomes the `TS.Elec.<name>` HS reference.

**Target set** (results, versioned): `device.TSHS` + `device.out` (device
SCF, its E_F), the `tbtrans` transmission files (`*.TBT.AVTRANS_*`,
`*.TBT.nc`), and `transport-result.json` (the portable summary, § 6.6).

**Per-stage I/O contract** (what each stage *requires* in / *guarantees* out):

| Stage | Requires (in) | Guarantees (out) | Boundary knobs |
|---|---|---|---|
| **relax** | labeled device geom + contract + frozen list | relaxed coords; `max|F| < tol` | `MD.CG`, `kz=1` |
| **electrode** | clone of frozen-electrode layers + *same* contract + dense `kz` + `SaveHS` | `electrode.TSHS` (H_lead, S_lead, E_F^lead) | `NumCGsteps 0`, `kz` dense |
| **transiesta** | relaxed device coords + *same* contract + `electrode.TSHS` | `device.TSHS`, device E_F, converged SCF | `kz=1`, `SolutionMethod transiesta` |
| **tbtrans** | `device.TSHS` (+ electrode reference) | `T(E)`, `G₀ = T(E_F)` | bias = 0 (or swept) |

The **preflight** (`§ 6.3`, now built) is the gate that sits on the
device↔electrode edge of this diagram and refuses to proceed unless the
invariant set holds.

### § 6.1 One descriptor → three runs (single source of truth)
A `transport-plan` carries: the labeled device (`L/R-electrode` +
`interface` + relaxed/frozen, via the `*-electrode` convention,
`region-labels.md`), the **numerical contract** (XC, pseudos, MeshCutoff,
EnergyShift, per-species basis, transverse k), and the electrode spec
(thickness, `kz`). The framework *derives* the **relaxation fdf**, the
**electrode fdf**, and the **TranSIESTA fdf** from it — so § 4.3's
numerical contract and the geometry are **identical by construction**, not
by the user remembering to copy them.

### § 6.2 Electrode auto-extraction with a guaranteed clone (the "wizard")
Extract the `*-electrode` layers' *exact* coordinates + lateral cell from
the device and write the electrode fdf (single-point, `MD.NumCGsteps 0`) —
removing the #1 footgun (geometric identity, § 4.1/§ 4.5 "Golden Rule").
Size the electrode from the orbital range and **warn when too thin**
(§ 4.1); set its `kz` dense while the device stays `kz=1` (§ 4.2). This is
the planned "electrode wizard" (`project_transport_electrode_bias_workflow`).

### § 6.3 A consistency *preflight* (the highest-value piece)
Before any run, assert and refuse on violation:
- device ⇄ electrode transverse k **commensurate** (TBtrans requirement);
- electrode `kz` dense, device `kz = 1`;
- **identical** XC / pseudo / MeshCutoff / EnergyShift / Au-basis-tier;
- electrode atoms are a **clone** of the device frozen layers (within tol);
- electrode **thick enough** for the principal layer (orbital-range test);
- **zero z-vacuum** (or emit "this is a cluster model" so the user *chose*
  it, per § 1.3);
- frozen-device-Au basis **==** electrode-Au basis (§ 4.4 matching).
This turns the prose "Golden Rule" into automated gates — the single
biggest correctness lever, since these are exactly the silent failures.

### § 6.4 Orchestration that passes outputs between stages
A driver runs **relax → electrode → transiesta** in order, feeding the
relaxed coordinates into the device fdf and the `electrode.TSHS` into the
device's `TS.Elec.<name>` HS reference **automatically** — replacing
today's documented *manual* rename/copy step (`transiesta.py` "Electrode
.fdf workflow"). Then `tbtrans` for T(E)/G₀, and the finite-bias loop
(`TS.Voltage` is one value/run → the planned bias driver sweeps it and
stitches `*.TBT.AVTRANS_*` into I–V).

### § 6.5 Convergence-sweep mode (reuse the benchmark sweep machinery)
Sweep transverse-k {2,4,6}, MeshCutoff {300,400,500}, electrode-thickness
{3,6,9 layers}; report where T(E_F), E_F and forces stop moving. This
operationalizes § 4.6 ("converge it") instead of trusting a number, and
reuses the per-point-isolation + summarize pattern already built for the
GPU benchmark.

### § 6.6 Versioned data contracts
`transport-plan.json` (the descriptor) and `transport-result.json` (E_F,
G₀ = T(E_F), the convergence status, and the § 4.6 DFT-NEGF caveat flag) —
same persistence pattern as `environment.json` / `bench-result.json`,
portable to plots / the Results tab.

### § 6.7 The invariant set (what must stay identical, and across which runs)

This is the contract the framework *keeps* — break any row and the
transmission is silently wrong. "Across" = which runs must agree; "Gate" =
the preflight check `id` that enforces it (✗ = not yet machine-checked,
relies on the wizard's clone-by-construction or human review).

| # | Invariant | Across | Why (physics) | Gate |
|---|---|---|---|---|
| I1 | XC functional + authors | relax = electrode = device | one H footing; mixing shifts E_F | `contract.xc` |
| I2 | Pseudopotentials (per species) | all three | different core = different atom | wizard clones species ✓ |
| I3 | MeshCutoff | electrode = device | real-space grid must align for the NEGF coupling | `contract.meshcutoff` |
| I4 | PAO.EnergyShift | all three | sets orbital range = basis radius | `contract.energyshift` |
| I5 | Basis tier, per species | frozen-electrode-Au **==** device-Au | a basis step = spurious scattering (§ 4.4) | `contract.basis` |
| I6 | Lateral cell (a, b) | electrode = device | the lead tiles the device cross-section | `cell.transverse` |
| I7 | Transverse k (kx, ky) | electrode **commensurate** device | TBtrans projects lead k onto device k | `kgrid.transverse` |
| I8 | Device kz = 1 | device | open boundary along transport (no periodicity) | `kgrid.device_kz` |
| I9 | Electrode kz dense (converged) | electrode | it is a *periodic bulk* run; thin cell → large BZ | `kgrid.electrode_kz` |
| I10 | Electrode geom = device frozen layers | electrode ⇆ device | Σ self-energy must map atom-for-atom onto the device | wizard clones atoms ✓ |
| I11 | Electrode thickness ≥ principal layer | electrode | Σ assumes only nearest principal layers couple (§ 4.1) | `electrode.thickness` (warn) |
| I12 | z-vacuum ≈ 0 at the leads | device | a gap = strained/severed lead, not a junction (§ 1.3) | `device.z_vacuum` (warn) |
| I13 | Electrode writes its HS | electrode | the device run needs `electrode.TSHS` to exist | `electrode.saveHS` (warn) |

### § 6.8 Scientific-validation map (gate → principle → literature)

Each gate traces to a physical requirement and a reference (verified DOIs,
§ 8) — so the design is auditable, not asserted.

| Gate / invariant | Scientific principle | Doc § | Reference |
|---|---|---|---|
| `kgrid.device_kz` (I8) | NEGF open boundary: no Bloch periodicity along transport | § 1, § 4 | Brandbyge 2002 (10.1103/PhysRevB.65.165401) |
| `kgrid.electrode_kz` (I9) | bulk-lead BZ must be converged to get the right band structure | § 4.2 | Papior 2017 (10.1016/j.cpc.2016.09.022) |
| `contract.{xc,meshcutoff,energyshift}` (I1,I3,I4) | a single self-consistent Hamiltonian/Fermi reference across runs | § 4.3 | Brandbyge 2002; Soler 2002 (10.1088/0953-8984/14/11/302) |
| `contract.basis` (I5) | basis-set discontinuity ⇒ artificial backscattering | § 4.4 | Brandbyge 2002 |
| `cell.transverse` / `kgrid.transverse` (I6,I7) | lead Σ maps onto the device cross-section; k-projection | § 4.5 | Papior 2017 |
| `electrode.thickness` (I11) | principal-layer screening: only nearest layers couple | § 4.1 | Papior 2017 |
| Au valence 5s5p5d6s (semicore) ⇒ MeshCutoff 350–500 | semicore d needs a fine grid | § 4.3 | van Setten 2018 (10.1016/j.cpc.2018.01.012) |
| result caveat flag (G₀ vs exp ≈ 0.011 G₀) | DFT-NEGF overestimates conductance ~1–2 orders | § 4.6 | Xiao 2004 (10.1021/nl035000m) |

The preflight encodes I1,I3–I9,I11–I13 as machine checks (14 tests); I2 and
I10's atom-level clone are now guaranteed-by-construction by the electrode
wizard (§ 6.2, `transport electrode`), which derives the lead geometry +
lateral cell + numerical contract from the device — verified by a round-trip
test that feeds wizard output back through the preflight.

---

## § 7 Build order (what exists vs proposed)

| Piece | State |
|---|---|
| Device `.fdf` emitter (modern `TS.Elec`/`TS.ChemPots`), `*-electrode` detection | **built** (`transiesta.py`) |
| Real-binary smoke test | **built** (`test_transiesta_siesta_smoke_l4.py`) |
| **Electrode wizard** (clone + thickness/`kz`) | **built** (`wizard.py`, `transport electrode`) |
| **Consistency preflight** | **built** (`preflight.py`, `transport preflight`) |
| **3-run orchestration** (auto coord/TSHS hand-off) | **built** (`orchestrate.py`, `transport bundle`) |
| **Convergence sweep** | proposed (§ 6.5) |
| **Bias driver** + I–V stitch | proposed (§ 6.4) |
| `transport-plan` / `transport-result` schemas | proposed (§ 6.6) |

**Recommended first build: the consistency preflight (§ 6.3)** — it
captures § 4.1, § 4.2, § 4.3, the clone, and commensurate-k in one place
and immediately prevents the silent errors, before any orchestration.

---

## § 8 References

DOIs verified 2026-06-27; arXiv preprint links given for open access.

1. M. Brandbyge, J.-L. Mozos, P. Ordejón, J. Taylor, K. Stokbro,
   "Density-functional method for nonequilibrium electron transport,"
   *Phys. Rev. B* **65**, 165401 (2002). — original TranSIESTA / NEGF.
   doi:[10.1103/PhysRevB.65.165401](https://doi.org/10.1103/PhysRevB.65.165401)
   · arXiv:[cond-mat/0110650](https://arxiv.org/abs/cond-mat/0110650)
2. N. Papior, N. Lorente, T. Frederiksen, A. García, M. Brandbyge,
   "Improvements on non-equilibrium and transport Green function
   techniques: The next-generation TranSIESTA," *Comput. Phys. Commun.*
   **212**, 8–24 (2017). — modern TranSIESTA; electrode/principal-layer
   requirements, multi-electrode chemical potentials.
   doi:[10.1016/j.cpc.2016.09.022](https://doi.org/10.1016/j.cpc.2016.09.022)
   · arXiv:[1607.04464](https://arxiv.org/abs/1607.04464)
3. J. M. Soler, E. Artacho, J. D. Gale, A. García, J. Junquera,
   P. Ordejón, D. Sánchez-Portal, "The SIESTA method for ab initio
   order-N materials simulation," *J. Phys.: Condens. Matter* **14**,
   2745–2779 (2002). — SIESTA, PAO basis, EnergyShift, MeshCutoff.
   doi:[10.1088/0953-8984/14/11/302](https://doi.org/10.1088/0953-8984/14/11/302)
   · arXiv:[cond-mat/0111138](https://arxiv.org/abs/cond-mat/0111138)
4. M. J. van Setten, M. Giantomassi, E. Bousquet, M. J. Verstraete,
   D. R. Hamann, X. Gonze, G.-M. Rignanese, "The PseudoDojo: Training and
   grading a 85 element optimized norm-conserving pseudopotential table,"
   *Comput. Phys. Commun.* **226**, 39–54 (2018).
   doi:[10.1016/j.cpc.2018.01.012](https://doi.org/10.1016/j.cpc.2018.01.012)
   · arXiv:[1710.10138](https://arxiv.org/abs/1710.10138)
5. X. Xiao, B. Xu, N. J. Tao, "Measurement of Single Molecule Conductance:
   Benzenedithiol and Benzenedimethanethiol," *Nano Lett.* **4**, 267–271
   (2004). — Au–BDT ≈ 0.011 G₀ (the DFT-NEGF overestimation benchmark,
   § 4.6).
   doi:[10.1021/nl035000m](https://doi.org/10.1021/nl035000m)

(Estimations in § 4.1/§ 4.3 — orbital range vs Au(111) spacing, semicore
MeshCutoff — are order-of-magnitude arguments to be replaced by the § 6.5
convergence sweep on the actual system.)
