# Chemistry helpers — charge, protonation, hydrogens

**Role:** contract
**Domain:** model
**Companions:** `structure.md` (the `Structure` these operate on),
`science/chemistry-correctness.md` + `science/validation.md` (the
**correctness** half of `chemistry.py` — spin/charge parity, open-shell metals,
ECP resolution, the analyzer + engine adapters; migrating in the science wave),
`engines/siesta.md` + `engines/pyscf.md` (the emitters that consume the
resolved net charge).

`molbuilder/chemistry.py` (tests: `tests/test_chemistry.py`) analyzes a
`Structure` and cleans it up: it works out the **net charge**, adds missing
**hydrogens**, and relieves atomic **clashes**. All functions are pure (no
global state, no I/O) unless noted.

> **Two halves, two domains.** The module also holds the scientific-correctness
> machinery — spin/charge parity, open-shell-metal detection, PySCF ECP
> resolution, and the `analyze_structure` analyzer + its engine-parameter
> adapters. That half is a *science* concern (it decides whether a calculation
> setup is physically valid) and lives in `science/` (see the pointer in § 4).
> This doc covers the **structure-chemistry helpers**.

---

## 1. Net charge

Charge is resolved in **one** place so the SIESTA and PySCF emitters — whose
config fields are named differently (`cfg.net_charge` vs `cfg.charge`) — don't
each carry their own logic.

```mermaid
flowchart LR
    IN["resolve_net_charge(struct, explicit_charge)"]
    Q{"explicit_charge<br/>is None?"}
    OV["use it verbatim<br/>(0 = force neutral,<br/>disables auto-detect)"]
    AUTO["formal_charge_from_phosphates(struct)<br/>(count deprotonated phosphate O⁻)"]
    IN --> Q
    Q -- no --> OV
    Q -- yes --> AUTO
```

- **`resolve_net_charge(struct, explicit_charge)` → int** — the one resolver.
  An **explicit override wins**; `0` is meaningful (forces neutral, disables
  auto-detection), and only `None` triggers the phosphate heuristic below.
- **`formal_charge_from_phosphates(struct)` → int** — the auto-detect
  heuristic. It looks **only at phosphate groups**. For each phosphorus:
  1. Find non-bridging oxygen neighbours (an O whose only heavy neighbour is
     this P). Adjacency is distance-based: `_HX_CUT = 1.30 Å` (X–H),
     `_XX_CUT = 1.95 Å` (heavy–heavy).
  2. Count them (`n_nb`) and how many already carry an H (`n_h`). One
     non-bridging O is the implicit `P=O` (contributes **0**); each remaining
     bare O without an H contributes **−1**. So per phosphorus the
     contribution is **`−max(0, n_nb − 1 − n_h)`** — pure arithmetic, no
     atom-name sorting (*which* O is left bare is a protonation choice, § 2).

  It does **not** count carboxylates (Asp/Glu), protonated amines (Lys/Arg),
  histidine pKa, sulfonates/sulfates/nitrates, or metal coordination — those
  groups are invisible to it. A user with such a system **overrides** via
  `cfg.charge` (PySCF) / `cfg.net_charge` (SIESTA); the docstring and the
  emitter specs say so.

Two related counters:
- **`total_electrons(struct, charge=0)` → int** — sum of atomic numbers minus
  the charge (electron count, used by spin-parity checks).
- **`expected_pH7_peptide_charge(struct)` → int | None** — estimates a
  peptide's net charge at physiological pH: **only** Asp/Glu −1 and Lys/Arg +1.
  His, Cys, Tyr and the free N-/C-termini contribute **0** (His is ambiguous at
  pH 7; the termini cancel for a free peptide). Returns `None` when the
  structure doesn't look like a peptide.

---

## 2. Protonation & hydrogens

- **`protonate_phosphate_oxygens(struct)` → (Structure, n_added)** — adds an H
  to each bare, non-bridging phosphate O that needs one.
  - **Idempotent**: running it twice adds no extra H the second time. If no
    protonation is needed it returns the **same `Structure` instance** (`is`
    identity) with `n_added = 0`.
  - **H geometry**: O–H bond `0.96 Å`; P–O–H angle `109.47°` (sp3
    tetrahedral, computed as `sin = √8 / 3`); the O–H points *outward* from the
    centroid of P's other heavy neighbours, falling back to a perpendicular
    axis when that centroid is collinear with P–O.
  - **Edge cases (must not crash)**: empty structure → `(struct, 0)`; a P with
    a single non-bridging O (a lone `P=O`) → no protonation; mixed protonation
    → the alphabetically-first bare O stays as `P=O`, the rest get H.
- **`add_hydrogens(struct)` → Structure** — general H-addition for any built
  structure (correct sp3/sp2/sp geometry). **OpenBabel first**
  (`OBMol.AddHydrogens()`, no ghost-coordinate failure mode), **RDKit fallback**
  (`Chem.AddHs(mol, addCoords=True)`). This is what nucleic/3DNA builds route
  through (see `engines/builders.md`).

---

## 3. Geometry cleanup & properties

| Function | Purpose |
|---|---|
| `min_nonbonded_contact(struct, search_radius=2.5)` | closest approach between atoms in **different** residues — a steric-clash probe; returns `(distance, i, j)`, or `(None, None, None)` when there are no residue labels (intra-residue contacts are bonds, not clashes) |
| `relieve_clashes(struct, steps=1000)` → Structure | nudge atoms apart to remove steric clashes |
| `estimate_partial_charges(struct, total_charge=0.0, *, bond_cutoff=1.95, hx_cutoff=1.30)` | heuristic per-atom partial charges from electronegativity gaps (the cutoffs are keyword-only; same values as § 1) |
| `estimate_dipole_moment_debye(struct, …)` | the molecular dipole moment, in Debye |

---

## 4. Not here — the scientific-correctness machinery

`chemistry.py` also decides whether a *calculation setup* is physically valid.
That is a **science** concern, documented in `science/` (migrating in the
science wave), not the data model:

| Function / type | What it decides | Doc |
|---|---|---|
| `check_spin_charge_parity(struct, charge, spin)` | is `(charge, spin)` electron-count-consistent? | `science/chemistry-correctness.md` |
| `detect_open_shell_metals` / `detect_transition_metals` | which metals need open-shell treatment | `science/chemistry-correctness.md` |
| `explain_metal_spin`, `suggest_spin_total` | plausible spin states for a metal centre | `science/chemistry-correctness.md` |
| `resolve_pyscf_ecp(struct, …)` | which atoms need an effective core potential | `engines/pyscf.md` |
| `analyze_structure(struct)` → `ChemistryAnalysis` + `register_adapter` / `registered_adapters` | the analyzer + per-engine parameter adapters (the validation call graph) | `science/validation.md` |

Keeping the correctness half in `science/` means a reviewer checking whether a
default is scientifically defensible reads it alongside the other validation
contracts, not buried in a data-model helper doc.
