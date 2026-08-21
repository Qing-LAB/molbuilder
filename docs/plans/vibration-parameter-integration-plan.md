# The 22 shown-but-unread parameters — categories, integration paths, and the hint mechanism

**Role:** plan
**Domain:** engines (PySCF) + validation
**Started:** 2026-08-21
**Companions:** [`engines/pyscf.md`](?doc=engines/pyscf.md) § 7a (the SCF
dresser contract this plan's code is checked against);
[`science/validation.md`](?doc=science/validation.md) (the layers the kind
validator joins); `docs/science/references.bib` (every citation named here
resolves there, by test).

*(2026-08-21; follows the science-validation review.  The user's ruling:
these are not clutter to hide — they mark a GAP in the minimum
implementation.  Categorize them, work out the scientifically correct way
to honor each, and design how the user is guided — hints and real
literature references, "not a guarantee of correctness, but the system
senses what problem the user is solving and gives reasonable
recommendations."  Nothing below is implemented until its row is ruled
on.)*

**The measurement** (2026-08-21, regex over the deck + emitters, spot-verified):
the vibration form shows 55 parameters; 22 are never read by the render.
They fall into three categories with three different truths.

---

## Category 1 — SCF machinery (9 items): honor them; the answer does not change, the ROBUSTNESS does

These steer *how* each SCF converges, not *what* it converges to.  A
vibration deck runs many SCFs (equilibrium + 6N displaced points + the
per-mode ES pairs), so a convergence aid matters MORE here than in a
single optimization — one stuck displaced point poisons a Hessian column.

| item | what honoring it means in the deck | the hint the form should give | reference (to add to `references.bib`) |
|---|---|---|---|
| `scf_init_guess` | set on every `mf` the deck builds (equilibrium + `_build_mf_at`) | "`atom` often converges where `minao` fails for open-shell metals" | Lehtola, *J. Chem. Theory Comput.* **15**, 1593 (2019) — assessment of SCF initial guesses |
| `level_shift` | ditto; the converged minimum is unchanged once reached | "stabilizes oscillating SCF; remove or reduce if convergence is already clean" | Saunders & Hillier, *Int. J. Quantum Chem.* **7**, 699 (1973) |
| `damp` | ditto (early-iteration density damping) | "pairs well with level_shift for hard cases; slows easy ones" | same family; PySCF manual |
| `diis_space` | ditto (Pulay subspace size) | "8 is standard; raise for near-degenerate systems" | Pulay, *Chem. Phys. Lett.* **73**, 393 (1980); *J. Comput. Chem.* **3**, 556 (1982) |
| `scf_soscf` | wrap `mf` in Newton (`mf.newton()`) exactly as the optimization deck does | "quadratic convergence for the last digits; expensive per step" | Bacskay, *Chem. Phys.* **61**, 385 (1981) |
| `scf_conv_tol_grad` | pass through beside `scf_conv_tol` | interacts with the FD noise floor — the amplitude×tolerance advisory already guards this coupling | — (the existing gate's Mills1972 note) |
| `on_nonconvergence` | policy for a failed SCF.  **Scientific note:** for vibration the right default is stricter than optimization's — a silently-unconverged displaced point corrupts the Hessian, so `stop` should be the kind's default | "warn-and-continue is for surveys; frequencies from an unconverged point are not frequencies" | — |
| `chkfile` | equilibrium SCF writes it — cross-run restart.  *(A claim corrected here 2026-08-21: displaced SCFs do NOT seed from the equilibrium density today — measured, `kernel()` runs bare; `dm0` seeding is a worthwhile future improvement, recorded, not assumed.)* | "restart a killed run without repaying the equilibrium SCF" | — |
| `auxbasis` | pass to `density_fit(auxbasis=…)` where DF is on | "auto-selected aux sets are fine for def2 bases; override for exotic elements" | Weigend, *Phys. Chem. Chem. Phys.* **4**, 4285 (2002) |

**Recommendation: implement the whole category as one unit** — mechanical
pass-through into the two `mf`-construction sites, catalogue help gains the
hint sentence + `refs`, and the honesty test (below) turns each row green as
it lands.

## Category 2 — the physical model (3 items): ✅ DELIVERED 2026-08-21 on the measured support matrix

**The matrix, probed live against pyscf 2.13** (every verdict measured,
none recalled; water/STO-3G/B3LYP probes, then the full E2E):

| model | SCF | grad | analytic Hessian | polarizability (Raman) | verdict |
|---|---|---|---|---|---|
| gas | ok | ok | ok | ok | baseline |
| **PCM** (IEF-PCM/C-PCM) | ok | ok | **ok** — RKS *and* UKS, *and* with density fitting | **ok, and the solvent ENTERS the response** (tensor shifts 9e-2 a.u.; not a gas-phase number under a water label) | **HONORED end to end** |
| SMD | FAIL — this build compiled without `-DENABLE_SMD` | — | — | — | informed refusal naming the build flag + [Marenich2009] |
| ddCOSMO | ok | ok | FAIL — no analytic Hessian (`AttributeError`) | — | informed refusal + [Lipparini2014], suggests PCM |

Also measured: the dielectric parameter reaches the model (eps 78 vs
2.27 shifts E); a 0.005 Å displacement drops water's group C2v → Cs,
so re-symmetrization under a derivative would move the frame.

**What landed:**
- one spelling for the PCM decoration in `pyscf/scf_setup.py`
  (`SOLVENTS` table + `emit_solvent_lines`), the optimization deck's
  inline block switched onto it, and the vibration deck emits
  `_mb_apply_solvent(mf)` once and decorates EVERY construction —
  equilibrium, displaced, relaxation — because consistency is the
  science;
- `symmetry` honored on the **already-relaxed path only** (equilibrium
  SCF + Hessian under the group, PCM included; the displaced builder
  forces it off; with in-deck relaxation the kind validator refuses
  with the measured C2v→Cs reason).  Mode irrep labels remain the
  planned follow-up;
- the kind validator carries the whole matrix as guidance: PCM → an
  info note with the equilibrium-solvation caveat + [Tomasi2005,
  Cances1997]; SMD/ddCOSMO → the refusals above; solvent+use_gpu →
  refused pending a GPU-side probe; a method with no solvent → a warn
  that it methods nothing;
- **the live bar**: `test_water_in_water_runs_the_solvated_chain_end_
  to_end` — relax, Hessian, IR *and* Raman under one PCM Hamiltonian;
  the bands red-shift (1622 vs 1639 cm⁻¹ bend) and the Raman
  activities shift (65.5 vs 76.9 Å⁴/amu) — the solvent is in the
  physics, not the label.

**The honesty gate's open list is now EMPTY.**

### The original analysis (kept for the record)

| item | the scientific requirement | the honest path |
|---|---|---|
| `solvent`, `solvent_method` | implicit solvation must apply to the SAME Hamiltonian everywhere — relaxation, Hessian, dipole/polarizability derivatives.  Mixing gas-phase derivatives with a solvated SCF is not an approximation, it is an inconsistency | **investigate first**: what PySCF 2.13 actually provides (PCM/SMD/ddCOSMO gradients exist; analytic *Hessians* under solvation may not).  Then either full consistent integration, or the kind's validator refuses with the reason and the reference — never a silent gas-phase answer under a solvent label.  Equilibrium-solvation caveat stated when supported |
| `symmetry` | point-group symmetry is not only speed: modes carry irreducible-representation labels, and IR/Raman activity follows selection rules from them | worth implementing as a FEATURE (mode irrep labels in the artifact + viewer), not just a passthrough; relaxation must be checked to preserve the declared group |

References to add: Tomasi, Mennucci & Cammi, *Chem. Rev.* **105**, 2999
(2005) — the continuum-solvation review; Marenich, Cramer & Truhlar,
*J. Phys. Chem. B* **113**, 6378 (2009) — SMD; Cancès, Mennucci & Tomasi,
*J. Chem. Phys.* **107**, 3032 (1997) — IEF-PCM; Lipparini *et al.*,
*J. Chem. Phys.* **141**, 184108 (2014) — ddCOSMO.  Symmetry/selection
rules: Wilson, Decius & Cross (already `Wilson1955`).

**Recommendation: its own phase.**  Step 1 is a support-matrix
investigation against PySCF 2.13 (what exists for solvated/symmetric
Hessians and property derivatives), because the correct workflow cannot be
designed from memory of an API.  Until it lands, the two solvent items and
`symmetry` get an explicit validator line: *"declared but not yet honored
by the vibration deck — refused so a label cannot mislead"* (a refusal,
not a hidden field: the user LEARNS the gap exists).

## Category 3 — workflow and provenance (10 items): honor the useful, exclude the contradictory

| item | verdict proposed |
|---|---|
| `optimize` | **exclude from the kind** (`calculations = ["optimization"]`): the D3 ruling made relaxation non-optional here — `already_relaxed` is the one legitimate skip, and two switches for one fact is drift |
| `optimizer` | honor: the relax block hardcodes geomeTRIC today; berny is a legitimate choice (though geomeTRIC's constraint support is why it is the default — the hint says so) |
| `geom_etol`, `geom_continue_retries` | honor: thread into the relax block's convergence dict / retry wrapper beside the four criteria it already reads |
| `save_initial_xyz`, `save_optimized_xyz` | honor: trivial file writes, matching the optimization deck's behavior |
| `write_molwatch_log` | honor, and it is a real win: the relaxation phase streams the same molwatch log the optimization deck writes, so the live watcher covers the vibration run's first phase too |
| `write_trajectory`, `log_file`, `verbose_comments` | honor: same provenance semantics as the optimization deck |

## The hint mechanism — framework-level, one home

The user's ask: recommendations with *actual confirmed scientific
references the user can keep investigating*.  Proposal:

1. catalogue items gain an optional **`refs = ["Pulay1980", …]`** key;
2. the form's help expander renders the citations under the help text;
3. a doc-claims-style test pins every `refs` key to an entry in
   `docs/science/references.bib` — an invented citation fails CI;
4. the Methods generator may fold the same keys into the bibliography,
   so what guided the user's choice is what the manuscript cites.

## The honesty gate

A test (formalizing the measurement that found the 22): **every parameter
the vibration form shows is read by the vibration render, or carries an
explicit validator refusal naming the gap.**  It fails when a new item
leaks in unhonored — the leak class this whole plan exists to close.

## The implementation design — layers first, code checked against them

*(Written before the code, per the user's process ruling 2026-08-21:
"start from the contract update and the documentation of the design…
a structured framework that logically organizes things in well-designed
layers and dependencies and uses data or config files to drive the
pipeline."  The CONTRACT is `engines/pyscf.md` § 7a; this section only
sequences the work against it.)*

The discovery that shaped it: the framework ALREADY drives SCF emission
from data — `layout.SCF_SECTION` names the machinery items and
`layout.line` spells each one; the optimization deck applies them through
the Sections machinery.  The vibration deck's gap is not missing
features, it is a BYPASS: its lifted emitters hand-spell their own SCF
lines at three `mf`-construction sites.  So the work is un-forking, not
patching:

1. **`pyscf/scf_setup.py`** — the generator of the emitted
   `_mb_configure_scf(mf)` function body, from `SCF_SECTION` +
   `layout.line`.  New module, one producer, no per-knob code anywhere
   else.
2. **The vibration emitters call the door**: the three construction
   sites emit `_mb_configure_scf(mf)` calls plus their role-specific
   lines per the § 7a role table (chkfile at equilibrium; dm0 seeding
   and halt-with-index at displaced points; policy advisories from the
   kind validator).
3. **The catalogue rows** for the machinery items gain `refs` (the
   Pulay/Saunders–Hillier/Bacskay/Lehtola/Weigend keys, already in the
   bib) and the hint sentences of Category 1's table.
4. **The honesty gate lands as a test** and the audit numbers in this
   plan update.
5. Category 3's workflow knobs follow the same shape (the live-watch
   emitter reused from the optimization deck's one home, the optimizer
   choice honored, `optimize` excluded by `calculations`), then
   Category 2's investigation phase.

## Open decisions (each blocks only its own category)

- **D-i**: Category 1 as one implementation unit — proceed?
- **D-ii**: Category 2's support-matrix investigation first, refusal
  lines in the meantime — proceed?
- **D-iii**: Category 3 verdicts as tabled (`optimize` excluded, the rest
  honored) — proceed?
- **D-iv**: the `refs` key + bib-pinned test — proceed?
