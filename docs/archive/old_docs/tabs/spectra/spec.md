# Spectrum-calculation tab — design spec

Status: **design-locked, not yet implemented**.  Author-approved
2026-05-11.  Renamed Spectra → Spectrum-calculation 2026-06-06
(route was `/spectra`, now `/spectrum-calculation`); `/api/spectra/*`
BACKEND routes kept their name for stability.  This document is
the test contract for the Spectrum-calculation tab and the
underlying `molbuilder.spectra` subpackage.  Sister
files in this folder:

* [`references.bib`](references.bib) — curated bibliography (each
  entry verified by the author before merge).

## 1. Purpose

A post-relaxation analysis tab that runs **harmonic vibrational
analysis** on a relaxed structure and emits a self-contained
script the user runs externally.  The script produces:

* **Normal-mode wavenumbers** ω_i (cm⁻¹) and atomic eigenvectors
  Q_i (which atom moves how) — restricted to user-selected free
  atoms via a partial Hessian.
* **Raman activities** per mode (analytic polarizability
  derivative dα/dR_j projected onto each mode Q_i).
* For a user-selected subset of modes, **per-mode electronic
  structure**: HOMO / LUMO and the MO spectrum within a window
  [HOMO−N, LUMO+M] at displaced geometries (±A·Q_i), enabling
  computation of electron-phonon coupling constants
  g_i = ∂E_HOMO/∂Q_i (and similarly for LUMO) that drive
  inelastic transport features (phonon sidebands in I(V), IETS).

The Raman spectrum + 3D mode animation answer "what does this
molecule do vibrationally?"; the per-mode electronic structure
answers "**how does each vibration modulate the orbitals that
carry current?**" — the load-bearing data product for the
upcoming Transport tab.

## 2. Scope (v1)

In scope (1a + 1b):

* Hessian (analytic or finite-difference fallback) with partial-
  Hessian support when atoms are fixed.
* Raman activities (analytic dα/dR where the engine supports it).
* Per-mode displaced-geometry SCFs producing the electronic-
  structure data above.
* Engine: **PySCF only**.  The engine layer is abstracted (see
  §3.2) so adding SIESTA (finite-difference force constants +
  external Raman post-process) is a future drop-in.

Out of scope (1c, deferred):

* **IR intensities** (dipole derivatives).  The infrastructure
  (engine protocol, JSON schema, UI panel) reserves space —
  `ir_intensity_km_mol` is a nullable field in the mode result
  shape, the toggle exists in the form (disabled in v1), but
  rendering and computation are not implemented.
* **Anharmonic / quasi-RRHO corrections**.  v1 is strictly
  harmonic.
* **SIESTA engine**.  Reserved choice in `SpectraConfig.engine`,
  no implementation.

## 2.5 Layer model — progressive disclosure of computational effort

The tab presents a **four-layer linear chain** that maps directly
onto the natural quantum-chemistry workflow.  Each layer is
opt-in (after L1); each layer's data persists on disk; the user
moves up the chain incrementally as they want more detail.

| Layer | Name | Cost | What you get | Triggered by |
|---|---|---|---|---|
| **L1** | **Setup** | none (config only) | method / basis / dispersion / atom-fixing chosen; structure loaded | the user editing any L1 config field |
| **L2** | **Frequencies** | ~1-N SCF-equivalents (analytic Hessian) | equilibrium MO spectrum; wavenumbers ω_i + eigenvectors for every mode; 3D mode-animation data | always runs after L1 is set; required for any spectrum work |
| **L3** | **Raman activities** | ~5N SCF-equivalents | per-mode Raman activities (analytic dα/dR projected onto each mode) | `compute_raman = True` |
| **L4** | **Per-mode electronic structure** | 2 SCFs × M selected modes | HOMO/LUMO shift, ΔGap, g_HOMO / g_LUMO per selected mode | `es_mode_selection ≠ "skip"` |

### 2.5.1 Dependency graph

The chain looks linear from the user's perspective, but the
actual computational dependencies form a small tree:

```
L1 (config)
   │
   ▼
L2 (equilibrium SCF + Hessian + eigenvectors)
   │
   ├─► L3 (Raman activities)     — needs only L2's converged
   │                               equilibrium wavefunction
   │
   └─► L4 (per-mode displaced SCFs) — needs only L2's eigenvectors
                                      (to know which direction to
                                      displace along)
```

**L3 and L4 are independent of each other.**  L4 doesn't read
anything from L3; L3 doesn't read anything from L4.  This is
load-bearing for the re-run rules below: re-running L3 must NOT
wipe L4, and vice versa.

### 2.5.2 Re-run / wipe rules

* **L1 change** (method / basis / dispersion / atom-fixing /
  density-fit) → wipe L2, L3, L4.  Method changes invalidate the
  Hessian (which depends on the wavefunction at that level of
  theory); atom-fixing changes invalidate the mode count.  All
  downstream data is recomputed.
* **L2 change** (only triggered by an L1 change in practice — L2
  has no knobs of its own beyond the L1 config it inherits) →
  wipe L3 + L4.  Both depend on L2 directly.
* **L3 change** (`compute_raman` toggle) → **no effect on L4**.
  When toggled on, L3 runs; when toggled off, nothing destructive
  happens (the UI just hides the L3 panel, data stays in the file).
* **L4 mode-selection change** (`es_mode_selection`,
  `es_explicit_indices`, `freq_min_cm1`, `freq_max_cm1`) →
  **no effect on L3, additive within L4**.  Existing per-mode ES
  data is preserved; the next run computes ES for newly-selected
  modes that don't already have it.
* **L4 compute-parameter change** (`displacement_amplitude_ang`,
  `es_n_homo_below`, `es_n_lumo_above`) → wipe L4 only (L3
  unaffected).  Existing ES data was computed at a different
  amplitude / window and cannot be mixed.

The user-facing rule, simplified: **changing the molecule or the
level of theory means starting over; changing what you want
computed at the existing level of theory means adding work, not
redoing it.**

### 2.5.3 Soft dependency: top_n / threshold L4 selectors require L3

Two of the five Model-2 selectors (spec § 8) pick modes BY Raman
activity, so they require L3 to be complete first:

* `top_n`, `threshold` — disabled in the UI (with a hint) when L3
  is not complete.
* `none`, `all`, `explicit` — work with just L2 done.

The dependency is enforced both client-side (form
compatibility-engine locks the selector options when L3 is empty)
and server-side (the engine's `preflight()` returns an
error-severity Issue if a top_n / threshold selector is submitted
without L3 data on disk).

### 2.5.4 Frequency-range filter for L4 mode selection

L4 carries an **optional frequency window**
(`freq_min_cm1`, `freq_max_cm1`) that constrains mode selection
to modes whose frequency falls in `[freq_min, freq_max]`.  The
filter composes with the Model-2 selector:

* `all + freq=[500, 2000]` → ES for every mode in [500, 2000] cm⁻¹.
* `top_n=10 + freq=[2800, 3200]` → top-10 Raman-active modes
  within the C-H stretch region.
* `explicit=[7, 12, 14]` → ES for exactly those three modes; the
  frequency filter is **ignored** when an explicit list is given
  (the user said exactly these modes).

Cost-wise this filter pays off only at L4 (L2 + L3 are
fixed-cost — you compute the entire Hessian and the entire
polarizability-derivative tensor regardless of which window the
user cares about; you can't compute "just the C-H stretch part
of the Hessian").  L4 scales linearly with the number of
selected modes, so the filter directly reduces L4 cost.

### 2.5.5 Stepper UI

The Spectra tab presents the layer chain as a four-step stepper
at the top of the page:

```
[L1 ✓ Setup]  →  [L2 ✓ Frequencies]  →  [L3 ○ Raman]  →  [L4 ○ Electronic structure]
B3LYP/def2-SVP    54 modes (412-      not run             not run
Au fixed (32)     3656 cm⁻¹)
```

Each step's icon reflects its `phase_status` (see § 5):

* `○` empty — not yet run
* `◐` running — script is currently working on this phase
  (live-watch mode polling the JSON; § 6.1)
* `✓` complete — phase finished, data on disk

Clicking a step opens the form below configured for that step:

* L1 step open → all L1 config fields editable.  Editing surfaces
  a confirm dialog "this will discard L2 / L3 / L4 data".
* L2 step open → no editable fields (L2 has no knobs of its own);
  shows a summary panel of what L2 produced.
* L3 step open → `compute_raman` toggle + summary panel.
* L4 step open → all L4 fields editable (selector, value field,
  freq range, displacement amplitude, MO window).  Compute-
  parameter changes surface a confirm "this will wipe existing
  L4 data and recompute".  Mode-selection changes are additive
  (no confirm needed).

The four display panes below the stepper (spectrum plot, mode
list, 3D mode animation, ES panel) **auto-disable when their
backing layer is `empty`** — e.g., the spectrum plot greys out
until L2 is complete; the mode list shows but its Raman activity
column is blank until L3 is complete; the ES panel shows a
placeholder until at least one mode has L4 data.

## 3. Architecture

### 3.1 Engine-agnostic L1

```
molbuilder/
├── config/
│   └── spectra.py            # SpectraConfig
├── spectra/
│   ├── __init__.py
│   ├── results.py            # SpectraResults, ModeData (typed dataclasses)
│   ├── engine_base.py        # SpectraEngine Protocol + registry
│   ├── pyscf_engine.py       # PySCFSpectraEngine (v1)
│   ├── selection.py          # mode-selection logic (cfg -> List[int])
│   └── methods.py            # render_methods_md(cfg, results) -> str
└── parsers/
    └── spectra_json.py       # engine-independent JSON -> SpectraResults
```

`SpectraConfig` declares **what** the user wants — universal
vocabulary, no engine-specific names.  `SpectraResults` is the
shape the UI consumes — independent of which engine produced it.
Adding SIESTA later requires zero changes to these L1 types.

### 3.2 Engine-specific L2

```python
# molbuilder/spectra/engine_base.py
class SpectraEngine(Protocol):
    name:  str                              # "pyscf"
    label: str                              # "PySCF (analytic Hessian + dα/dR)"

    @classmethod
    def render_script(cls, struct: Structure, cfg: SpectraConfig) -> str: ...
    @classmethod
    def parse_output(cls, path: str) -> SpectraResults: ...
    @classmethod
    def preflight(cls, struct: Structure, cfg: SpectraConfig) -> List[Issue]: ...
    @classmethod
    def methods_fragment(cls, cfg: SpectraConfig,
                         modes: List[ModeData]) -> str: ...

_ENGINES: Dict[str, Type[SpectraEngine]] = {}

def register_engine(cls): _ENGINES[cls.name] = cls; return cls
def get_engine(name: str) -> Type[SpectraEngine]: ...
```

Adding SIESTA later: one `@register_engine class SiestaSpectraEngine`
plus a tuple-entry in `SpectraConfig.engine`'s `choices`
metadata.  Nothing else changes — the form, the blueprint, the
parser dispatch, and the Methods generator all delegate to the
registered engine.

### 3.3 Data flow

```
SpectraConfig + Structure
        │
        ▼
PySCFSpectraEngine.preflight()  ──►  [Issues]    (range + scientific warns)
        │
        ▼
PySCFSpectraEngine.render_script()  ──►  spectra.py
        │                                  │
        │                          [user runs externally]
        │                                  │
        │                                  ▼
        │                          <job>.spectra.json
        │                                  │
        ▼                                  ▼
PySCFSpectraEngine.parse_output()  ──► SpectraResults
        │
        ▼
methods.py::render_methods_md(cfg, results)  ──► Methods markdown
        │
        ▼
Spectra tab UI (engine-blind: consumes SpectraResults only)
```

External run = the user typing `python spectra.py` on a cluster.
We do not orchestrate the run.

## 4. `SpectraConfig` — every field

All fields carry `dataclasses.field(..., metadata=...)` with the
standard keys (`section`, `label`, `unit`, `range`, `choices`,
`tier`, `help`, optional `id_suffix`, optional `null_label`) so
the schema-driven Build-form pipeline (see
[`../../protocols/web-api.md`](../../protocols/web-api.md)
§ `/api/build/schema/<engine>`) renders the form automatically.

| Field | Type | Default | Section | Notes |
|---|---|---|---|---|
| `engine` | str | `"pyscf"` | System | `choices=("pyscf",)`; SIESTA reserved |
| `job_name` | str | `"spectra"` | System | filesystem-safe basename per [`job-layout.md`](../../protocols/job-layout.md); pattern `[A-Za-z0-9_-]+` |
| `method` | str | `"RKS"` | Method | `choices=("RKS","UKS","RHF","UHF")` |
| `functional` | str | `"B3LYP"` | Method | any libxc string |
| `basis` | str | `"def2-SVP"` | Method | any PySCF-recognised basis |
| `dispersion` | Optional[str] | `"d3bj"` | Method | `choices=("d3","d3bj","d4","none")`; `"none"` -> None |
| `density_fit` | bool | `True` | Method | RI-J / RI-JK |
| `frozen_elements` | List[str] | `[]` | Frozen atoms | union with the index list (see §7) |
| `frozen_residue_names` | List[str] | `[]` | Frozen atoms | union |
| `frozen_indices` | List[int] | `[]` | Frozen atoms | 0-based |
| `compute_raman` | bool | `True` | Spectrum | `False` runs only the Hessian (faster; for diagnostic use) |
| `compute_ir` | bool | `False` | Spectrum | reserved, disabled in v1 UI |
| `displacement_amplitude_ang` | float | `0.02` | Spectrum | range (0.02, 0.30); 0.02 Å keeps the ES probe in the linear-response regime (ΔE_orbital ∝ displacement); see §11.4 |
| `es_mode_selection` | str | `"skip"` | Electronic structure | Model 2 selector: `choices=("skip","all","top_n","threshold","explicit")`; `top_n` + `threshold` soft-require L3 (§ 2.5.3) |
| `es_top_n` | int | `10` | Electronic structure | active when selector=`top_n` |
| `es_threshold` | float | `1.0` | Electronic structure | Å⁴/amu; active when selector=`threshold` |
| `es_explicit_indices` | List[int] | `[]` | Electronic structure | 1-based mode indices |
| `freq_min_cm1` | Optional[float] | `None` | Electronic structure | optional lower bound on mode frequency for L4 selection (§ 2.5.4); ignored when selector=`explicit` |
| `freq_max_cm1` | Optional[float] | `None` | Electronic structure | optional upper bound on mode frequency for L4 selection |
| `es_n_homo_below` | int | `5` | Electronic structure | record MOs from HOMO−N to LUMO+M |
| `es_n_lumo_above` | int | `5` | Electronic structure | same |
| `scf_conv_tol` | float | `1e-9` | SCF | Hartree |
| `scf_max_cycle` | int | `100` | SCF | |
| `grid_level` | int | `4` | SCF | DFT grid; hybrid functionals need ≥4 |
| `max_memory_mb` | int | `4000` | Runtime | |
| `threads` | Optional[int] | `None` | Runtime | None → inherit OMP_NUM_THREADS |
| `verbose` | int | `4` | Runtime | PySCF log level |
| `verbose_comments` | bool | `True` | Runtime | inline citation + tuning hints in emitted script |

**Class-level**:

```python
_form_section_order = (
    "System", "Method", "Frozen atoms",
    "Spectrum", "Electronic structure", "SCF", "Runtime",
)
```

so the schema-driven form renders sections in workflow order.

## 5. `SpectraResults` + `ModeData`

```python
@dataclass
class ModeData:
    index_1based:          int
    frequency_cm1:         float
    raman_activity_a4_amu: Optional[float]            # None if compute_raman=False
    ir_intensity_km_mol:   Optional[float]            # always None in v1 (1c reserved)
    eigenvector_canonical: np.ndarray                 # shape (n_free, 3); canonical mass-weighted (Σ_k m_k |L_k|² = 1)
    eigenvector_display:   np.ndarray                 # shape (n_free, 3); max(|L|)=1 per mode, for UI animation
    has_imag:              bool                       # |ω|^2 < 0  -> True; sign convention: imag freqs reported negative

    # Per-mode electronic structure (None when this mode wasn't selected)
    electronic_structure:  Optional[ModeElectronicStructure] = None

@dataclass
class ModeElectronicStructure:
    amplitude_ang:        float                       # the A used for ±A·Q displacement
    mo_energies_eq_eh:    np.ndarray                  # equilibrium reference (shape (n_window,))
    mo_energies_minus_eh: np.ndarray
    mo_energies_plus_eh:  np.ndarray
    homo_index_in_window: int                         # which row of the windows is the HOMO
    scf_energy_eq_eh:     float
    scf_energy_minus_eh:  float
    scf_energy_plus_eh:   float

@dataclass
class SpectraResults:
    schema_version:   int                             # = 1 for this spec
    engine:           str
    engine_version:   str
    molbuilder_version: str
    timestamp:        str                             # ISO-8601 UTC

    structure_hash:   str                             # SHA-256 of canonical XYZ; provenance
    n_atoms_total:    int
    free_atom_idxs:   List[int]                       # 0-based; complement of frozen set
    frozen_atom_idxs: List[int]                       # 0-based

    equilibrium_scf_eh:         float
    equilibrium_mo_energies_eh: np.ndarray            # all MOs at equilibrium (Hartree)
    equilibrium_homo_idx:       int

    modes:                List[ModeData]              # sorted by frequency ascending
    selected_mode_idxs_1based: List[int]              # which ones got ES treatment

    config:               Dict[str, Any]              # the SpectraConfig as dict (provenance)
    methods_text:         str                         # pre-rendered Methods paragraph
    bibliography_keys:    List[str]                   # keys actually cited in methods_text

    # Per-layer status flags (replaces the older single `complete: bool`).
    # Each one of:
    #   "empty"     -- phase has not been computed (data fields are
    #                  default-empty / None / [])
    #   "running"   -- phase is currently being computed; partial
    #                  data may be present (e.g. some modes have ES
    #                  but not all; L4 only)
    #   "complete"  -- phase finished, data is final for this run.
    phase_frequencies: str        # L2: equilibrium SCF + Hessian + modes
    phase_raman:       str        # L3: Raman activities
    phase_es:          str        # L4: per-mode displaced-geometry SCFs

    # Engine-specific noise kept here so the common schema doesn't bloat.
    engine_metadata:      Dict[str, Any] = field(default_factory=dict)
```

Phase semantics:

* **L1 (Setup)** has no phase_status of its own — the *presence*
  of a valid `SpectraResults` (with a passing `__post_init__`)
  IS the Setup-complete signal.  The `config` field carries the
  L1 parameters that produced everything below.
* `phase_frequencies == "complete"` means `modes` is populated
  with at least one mode, eigenvectors are present, equilibrium
  SCF data is valid.
* `phase_raman == "complete"` means every mode's
  `raman_activity_a4_amu` is a real float (not None).
* `phase_es == "complete"` means every mode listed in
  `selected_mode_idxs_1based` has a populated
  `electronic_structure`.  Modes NOT in that list have
  `electronic_structure = None` — that's correct, not an
  incomplete state.
* The `"running"` states only appear in live-watch mode (when
  the engine's script atomically replaces the JSON during a
  multi-phase run).  Post-completion the file's three phase_*
  fields settle into some combination of `"empty"` and
  `"complete"` reflecting what the user opted into.

The UI consumes only the common surface (`modes`, `equilibrium_*`,
`methods_text`, `bibliography_keys`).  `engine_metadata` is for
debugging / future tools and is **not** rendered by default.

### 5.1 Atom-index contract (validatable)

This module is **independent** of the fused molview+selection module
(`molview-module.md` — spectra's "selection" is a *mode*, not atoms), but
isolation must **not** fork indexing: it shares the system-wide atom-index
contract (`data-vocabulary.md § 3.1 / § 3.2`). Two 0-based index spaces coexist
here — mixing them is the silent-corruption hazard, so state them explicitly.

**Two 0-based index spaces:**
- **Global atom index** — over all `n_atoms_total` atoms, in the structure's
  0-based CARRIED order (`§ 3.2`), pinned by `structure_hash`.
  `equilibrium.elements[i]` / `positions_ang[i]` are **global atom `i`**.
- **Free-atom row** — over the `n_free` free atoms only.
  `ModeData.eigenvector_{display,canonical}` are shape `(n_free, 3)`; **row `k`
  is NOT global atom `k`** — it is global atom `free_atom_idxs[k]`. Frozen atoms
  have no eigenvector row (they don't move).

**Invariants — a conforming `SpectraResults` MUST satisfy (each is a test):**
1. `free_atom_idxs` and `frozen_atom_idxs` are 0-based global indices that
   **partition** `range(n_atoms_total)` (disjoint; union = every atom).
2. `len(eigenvector_display) == len(eigenvector_canonical) == len(free_atom_idxs)`
   for every mode.
3. The animation displaces global atom `free_atom_idxs[k]` by row `k`; every
   atom **not** in `free_atom_idxs` stays at zero. *(Frontend `_startAnimation`
   builds a length-`n_atoms_total` zero-filled displacement array, then scatters
   free rows by `free_atom_idxs` — that scatter IS the check.)*
4. Any atom index shown to the **user** goes through `atomIndexModel.toDisplay`
   (1-based, `§ 3.1`). `ModeData.index_1based` is the **mode** number (already
   1-based), **not** an atom index — don't confuse the two.

**Why it matters:** with frozen atoms present, applying `eigenvector_display[i]`
to global atom `i` would silently displace the **wrong atoms**. Invariant 3 (the
`free_atom_idxs` scatter, not a positional `[i]`) is the guard. A test loads a
`SpectraResults` with a non-trivial frozen set and asserts the built displacement
array is zero on `frozen_atom_idxs` and equals the eigenvector on
`free_atom_idxs`.

## 6. `<job>.spectra.json` — on-disk schema

The script writes exactly one JSON file per run:

```json
{
  "schema_version":     1,
  "engine":             "pyscf",
  "engine_version":     "2.x.y",
  "molbuilder_version": "1.2.0",
  "timestamp":          "2026-05-11T12:34:56Z",
  "structure_hash":     "sha256:...",
  "n_atoms_total":      50,
  "free_atom_idxs":     [3, 4, 5, 6, 7, ...],
  "frozen_atom_idxs":   [0, 1, 2, 8, 9, ...],
  "config":             { /* SpectraConfig as JSON-safe dict */ },

  "equilibrium": {
    "scf_energy_eh":    -123.4567890,
    "mo_energies_eh":   [-19.2, -10.5, ..., 0.21, 0.45, ...],
    "homo_idx":         27
  },

  "modes": [
    {
      "index_1based":         1,
      "frequency_cm1":        412.3,
      "raman_activity_a4_amu": 12.5,
      "ir_intensity_km_mol":  null,
      "has_imag":             false,
      "eigenvector_canonical": [[dx, dy, dz], ...],
      "eigenvector_display":   [[dx, dy, dz], ...],
      "electronic_structure": null
    },
    {
      "index_1based":         7,
      "frequency_cm1":        1023.4,
      "raman_activity_a4_amu": 87.2,
      "ir_intensity_km_mol":  null,
      "has_imag":             false,
      "eigenvector_canonical": [[dx, dy, dz], ...],
      "eigenvector_display":   [[dx, dy, dz], ...],
      "electronic_structure": {
        "amplitude_ang":        0.10,
        "mo_energies_eq_eh":    [...],
        "mo_energies_minus_eh": [...],
        "mo_energies_plus_eh":  [...],
        "homo_index_in_window": 5,
        "scf_energy_eq_eh":     -123.4567,
        "scf_energy_minus_eh":  -123.4521,
        "scf_energy_plus_eh":   -123.4523
      }
    }
  ],

  "selected_mode_idxs_1based": [7, 12, 14, ...],
  "methods_text":              "Harmonic vibrational analysis was ...",
  "bibliography_keys":         ["Sun2020", "Becke1993", "Grimme2011", ...],

  "phase_frequencies": "complete",
  "phase_raman":       "complete",
  "phase_es":          "running"
}
```

**Conventions**:

* All energies in **Hartree** (Eh) at the wire level; the parser
  converts to **eV** when populating the typed `SpectraResults`
  fields that have an `_ev` suffix.
* All frequencies in **cm⁻¹**.  Imaginary frequencies reported as
  negative values; `has_imag: true` is the canonical flag.
* All distances in **Å**.
* Atom indices are **0-based** in JSON to match Python; the UI
  displays them 1-based.
* Mode indices are **1-based** everywhere (JSON + UI) to match
  spectroscopic literature convention.
* Each mode ships **two normal-mode eigenvector arrays**, both shape
  `(n_free, 3)`, both restricted to the free atoms:
    * `eigenvector_canonical` -- Cartesian normal mode L_cart in the
      canonical mass-weighted normalisation `Σ_k m_k |L_k|² = 1`
      (m_k in atomic units of mass).  Use this for Placzek Raman
      activity (the formula `45 a² + 7 γ²` gives Å⁴/amu in these
      units), IR intensity, electron-phonon coupling gradients --
      anything that depends on the physical amplitude of nuclear
      motion.
    * `eigenvector_display` -- same mode rescaled so `max(|L_k|)=1`.
      Dimensionless.  Use this for 3D animation (every mode reaches
      the same peak amplitude on screen) and for the fixed-amplitude
      electron-phonon "probe displacement" in Phase 4.  Do NOT feed
      this into physical-amplitude formulas.
  SCHEMA_VERSION 1 had a single `eigenvector_free` field that
  ambiguously served both roles; `from_dict` continues to accept
  v1 documents and treats `eigenvector_free` as the display form.

## 6.1 Live-watch: atomic-replace JSON checkpointing

The emitted script writes `<job_name>.spectra.json` at the end
of each phase, plus after each per-mode SCF in L4.  Writes are
**atomic** (`tempfile.NamedTemporaryFile` + `os.replace`) so a
concurrent reader never observes a partially-written JSON.

`phase_*` fields transition over the lifetime of a run:

```
start:        phase_frequencies="empty",  phase_raman="empty",  phase_es="empty"
L2 begins:    phase_frequencies="running"
L2 done:      phase_frequencies="complete"
L3 begins:    phase_raman="running"           (only if compute_raman=True)
L3 done:      phase_raman="complete"
L4 begins:    phase_es="running"              (only if es selector ≠ "skip")
L4 in flight: phase_es="running",  per-mode ES populated incrementally
                                   (selected modes get electronic_structure
                                   populated one at a time; partial state
                                   is a valid intermediate JSON)
L4 done:      phase_es="complete"
```

The Watch-style polling endpoint (`/api/spectra/data`, § 10)
returns the current state of the file each poll; the UI's
stepper renders `running` icons for phases in flight and updates
the four display panes as their backing layers become
`"complete"`.

A user re-running the script with new L4 mode selections starts
from `phase_es="empty"` (only L4 wiped, per § 2.5.2's compute-
parameter-change rule) or with `phase_es` preserved at its
existing state (mode-selection change is additive; new modes are
added to the existing L4 data).  The engine's preflight
determines which path applies and emits the appropriate script
header.

## 7. Atom-fixing semantics

The free-atom set is computed once at script start:

```
frozen = ∪ {i : element[i] ∈ cfg.frozen_elements}
       ∪ ∪ {i : residue_name[i] ∈ cfg.frozen_residue_names}
       ∪ set(cfg.frozen_indices)
free   = {0, 1, ..., N-1} ∖ frozen
```

Union semantics: an atom is frozen if it matches **any** of the
three filters.  Empty filters do nothing.  A user freezing all Au
atoms (e.g. for a metal–molecule–metal junction) sets
`frozen_elements=["Au"]`; everything else stays free.

* **Edge cases**:
  * `frozen = ∅`: all atoms move (default).
  * `frozen = all atoms`: error-severity issue surfaced in
    preflight ("no free atoms to vibrate").
  * `n_free = 1`: degrees of freedom too few for a Hessian; warn.
  * Residue-name filter requires the input structure to carry
    residue metadata; loaded `.xyz` files don't (the `.xyz` format
    has no residue column).  The form disables the residue-name
    multi-select when the loaded structure has no residue info.

> **Sidecar-driven defaults.**  When the user picks an XYZ that
> has a `.molstruct.json` sidecar, the form's `frozen_indices`
> field is pre-filled with `sidecar.frozen_atoms` so the boundary
> condition is **visible** before Generate.  The form is
> authoritative (the user can edit, including clearing the
> pre-fill).  Divergence between sidecar and form, and any
> sidecar labels the engine doesn't consume (e.g. `regions`), are
> surfaced as preflight WARN-severity Issues.  Full contract:
> `docs/design.md` §"Sidecar-driven boundary conditions — the
> three-stage contract".

Effect on the Hessian: PySCF's `mol.set_geom_(...).build()` with
the `atmlst=free_atom_idxs` kwarg on `Hessian.kernel()` computes
a **partial Hessian** — only the (3·n_free × 3·n_free) block is
filled.  Vibrational modes are then mass-weighted and diagonalised
over the free subspace; frozen atoms contribute zero to the
eigenvectors.

## 8. Mode-selection semantics (Model 2)

After the Hessian + Raman activities are in hand, the script
applies `cfg.es_mode_selection` to pick which modes get the
displaced-geometry SCFs:

| Selector | Selected mode set | Notes |
|---|---|---|
| `none` | `{}` | No displaced SCFs.  Output has `electronic_structure: null` on every mode.  Cost: 0 extra SCFs. |
| `all` | every vibrational mode | 2 SCFs per mode (±A).  Cost: 2 × (3·N_free − 6) SCFs (5 for linear molecules — translation/rotation degrees of freedom removed). |
| `top_n` | `n` modes with the highest `raman_activity_a4_amu` | Ties broken by lower mode index.  Cost: 2·n SCFs. |
| `threshold` | modes with `raman_activity_a4_amu > cfg.es_threshold` | Cost: variable. |
| `explicit` | `cfg.es_explicit_indices` (1-based) | Validated: indices out of range raise a preflight error.  Cost: 2·len(list) SCFs. |

**Two-stage workflow** (cheapest scientifically): first run with
`es_mode_selection="skip"` → see the spectrum + animations →
identify modes of interest → re-run with `es_mode_selection="explicit"`
and the chosen indices.  This is a natural usage pattern of the
single-script design; no special UI flow is needed.

### 8.1 Frequency-range filter (composes with the selector)

L4 carries an optional frequency window
(`freq_min_cm1`, `freq_max_cm1`) that **restricts the selector's
output** to modes whose frequency lies in `[freq_min, freq_max]`:

| Selector | Effect of `freq_min` / `freq_max` |
|---|---|
| `none` | ignored (no L4 work) |
| `all` | "all modes within the window" |
| `top_n` | "top n by Raman activity, AMONG modes within the window" |
| `threshold` | "modes with activity > threshold AND within the window" |
| `explicit` | **ignored** — the user named specific modes, the window doesn't override |

Either bound `None` removes that side of the window
(`freq_min=None` → no lower bound).  Both `None` → no filter
(default).

Why the filter exists (cost-wise): L4 is the only phase whose
cost scales with the number of selected modes.  L2 (Hessian)
and L3 (Raman activities) are fixed-cost — you compute the whole
matrix and the whole polarizability-derivative tensor regardless
of the user's frequency window.  See § 2.5.4 for the cost-table.

Scientific caveat (rendered as a yellow hint in the UI next to
the freq fields):

> Filtering by frequency range skips modes whose strong
> electron-phonon coupling may lie outside that window.  The L4
> ES data only reflects the selected modes; transport
> interpretation should account for the omitted ranges.
> [Galperin2007]

## 9. UI contract

### 9.1 Form (schema-driven)

The Spectra tab's form is generated by the existing form-schema
pipeline (see [`../../protocols/web-api.md`](../../protocols/web-api.md)
§ `/api/build/schema/<engine>`) from `SpectraConfig` field
metadata, with one new schema endpoint:

* `GET /api/build/schema/spectra` — returns the schema for
  `SpectraConfig`, parallel to existing `siesta` / `pyscf`
  endpoints.

The form has a load-file row at the top (Option A: structure
comes from disk only; see §1 design fork resolved 2026-05-11).
A "Send to Spectra" handoff from Build / Modify is **not**
shipped in v1 — the user saves a relaxed structure to disk and
loads it here.

**The form panel below the stepper changes based on which step
is open** (§ 2.5.5):

* L1 step open → all L1 config fields editable (System / Method /
  Frozen atoms / SCF / Runtime sections of the schema).
* L2 step open → no editable fields; summary panel showing what
  L2 produced (number of modes, frequency range, equilibrium SCF
  energy, HOMO/LUMO gap at rest).
* L3 step open → just the `compute_raman` toggle + a summary
  panel (max Raman activity, brightest mode index).
* L4 step open → all L4 config fields (selector, value field,
  freq_min / freq_max, displacement_amplitude_ang, MO window).

The schema-driven form's compatibility-engine handles the
section-level locking — when L1 is finalised (its phase is
"complete" in the loaded JSON), the L1 fields lock until the
user explicitly clicks the L1 step to edit them (which raises
the discard-downstream confirm).

### 9.2 Display — four panes

```
┌────────────────────────────────────────────────────────────────┐
│  9.2.1 Spectrum plot                                           │
│  - x: wavenumber (cm⁻¹), 0 .. 4000                             │
│  - y: Raman activity (Å⁴/amu)                                  │
│  - sticks + Lorentzian-broadened envelope (FWHM user-adjustable)│
│  - imaginary modes flagged red, at negative ω                  │
│  - click stick -> select mode (synchronises with §9.2.2)       │
├────────────────────────────────────────────────────────────────┤
│  9.2.2 Mode list (full tabular view of all per-mode data)      │
│  See § 9.2.2.* below for the column set.                       │
├────────────────────────────────────────────────────────────────┤
│  9.2.3 3D viewer with mode animation                           │
│  - structure rendered via shared 3Dmol style (see Build)       │
│  - selected mode: free atoms animate sinusoidally along Q_i    │
│  - amplitude slider (default 0.5 Å peak-to-peak)               │
│  - speed slider; pause toggle                                  │
│  - fixed atoms greyed out, no animation                        │
├────────────────────────────────────────────────────────────────┤
│  9.2.4 Electronic-structure panel (for selected mode)          │
│  - bar diagram of MO energies at -A / 0 / +A                   │
│  - HOMO highlighted (filled), LUMO highlighted (open)          │
│  - gap drift annotated: ΔGap(±A) = Gap(±A) − Gap(0)            │
│  - electron-phonon coupling readout:                           │
│      g_HOMO = (E_HOMO(+A) − E_HOMO(−A)) / (2·A·√(ℏ/(2·m·ω)))   │
│      g_LUMO (similar)                                          │
│  - placeholder + hint when selected mode has ES = null         │
└────────────────────────────────────────────────────────────────┘
```

#### 9.2.2.1 Mode list — columns

The mode list is the **tabular twin of the spectrum plot**.  It
shows every mode with all available per-mode data so a user who
wants numbers (rather than a chart) gets them without leaving
the tab.  Every row corresponds to one stick in §9.2.1; clicking
either drives the same selection state.

| Column | Source | Units / format | Notes |
|---|---|---|---|
| `#` | `mode.index_1based` | 1-based int | sortable; ascending by default |
| `ω` | `mode.frequency_cm1` | cm⁻¹, 1 dp | imaginary modes shown red with negative sign |
| Raman activity | `mode.raman_activity_a4_amu` | Å⁴/amu, 3 sig fig | sortable; blank if `compute_raman=False` |
| IR intensity | `mode.ir_intensity_km_mol` | km/mol | blank in v1 (1c reserved column) |
| ES? | derived | "✓" or "—" | "✓" if `mode.electronic_structure` is populated |
| HOMO (eq) | `ModeElectronicStructure.mo_energies_eq_eh[homo_index_in_window]` | eV (converted from Eh) | only when ES present |
| LUMO (eq) | adjacent row in same array | eV | only when ES present |
| Gap (eq) | computed | eV | LUMO − HOMO at equilibrium |
| ΔHOMO(+A) | `mo_energies_plus_eh[homo] − mo_energies_eq_eh[homo]` | meV | shift of HOMO under +A displacement |
| ΔHOMO(−A) | analogous | meV | |
| ΔLUMO(+A) | analogous | meV | |
| ΔLUMO(−A) | analogous | meV | |
| ΔGap (max) | `max(|ΔGap(+A)|, |ΔGap(−A)|)` | meV | quick-scan column for transport relevance |
| g_HOMO | computed | meV/(amu·Å)¹ᐟ² | electron-phonon coupling magnitude (see §9.2.4 formula) |
| g_LUMO | analogous | meV/(amu·Å)¹ᐟ² | |

Defaults visible: `#`, `ω`, `Raman act`, `ES?`, `HOMO`, `LUMO`,
`Gap`, `ΔGap (max)`.  The remaining columns are toggled on via a
"Columns…" menu (default-off because they bloat the table when
ES is computed for all modes).  Column visibility persists in
sessionStorage.

#### 9.2.2.2 Mode list — interactions

* **Sort**: click any column header.  Click again to reverse.
  Default sort is `#` ascending (frequency ordering).
* **Filter**: a single text input above the table filters by any
  visible column's stringified value (case-insensitive substring
  match).
* **Selection**: clicking a row sets the active mode for the 3D
  animation (§9.2.3) and the electronic-structure panel (§9.2.4).
  Active row gets a highlighted background.  Clicking a stick in
  the spectrum plot (§9.2.1) also sets the selection; the two
  views are synchronised through a single shared state.
* **Export**: a small "Export CSV" button on the table header
  downloads the currently-filtered, currently-sorted view as a
  CSV with the visible columns.  Useful for plotting elsewhere
  or pasting into a manuscript table.
* **Empty-row hint**: when ES isn't populated for any mode (the
  user picked `es_mode_selection="skip"`), the ES-derived
  columns are hidden entirely (not just blanked).  When ES is
  populated for *some* modes, those columns are shown and
  un-selected modes display "—".

#### 9.2.2.3 Mode list — accessibility

* The table is a real `<table>` with `<thead>` + `<tbody>` + a
  visible `<caption>`, not a div-stack; screen readers announce
  it as a data table.
* Each row has `aria-selected` synced to the active selection.
* Column headers carry `aria-sort` reflecting the current sort.
* The filter input has an `aria-controls` pointer to the table
  `id`.

### 9.3 Cost preview

A live readout near the bottom of the form:

```
This configuration will run:
  1 equilibrium SCF
  1 analytic Hessian (~ N_free²·k_basis cost)
  1 polarizability derivative (~ 5 SCF-equivalents for analytic dα/dR)
  2 × M displaced SCFs       (M = N selected modes by current selector)
  ─────────────────────────
  ≈ K SCF-equivalents total

  Estimated wall time on 8 cores: T hours
  (heuristic; actual depends on basis size and SCF convergence)
```

`K` and `T` are computed client-side from `cfg` + the loaded
structure (`N_atoms`, element distribution).  The estimate is
explicitly labelled as a heuristic with a 2-3× error bar.

### 9.4 Methods-preview button

Button: **Show methods text**.  Opens a modal showing the
Markdown that would land in the script's header comment.  Updates
live as form values change so the user can see how the prose
shifts.  Copy button so it can be pasted into a manuscript
draft.

The Markdown is computed by `molbuilder/spectra/methods.py::render_methods_md(cfg, struct)`
(structure metadata informs phrasings like "5 fixed Au atoms" vs
"all-atom analysis").  Run-time numbers that depend on the actual
calculation result (final wavenumbers, gap drifts, etc.) are
placeholders pre-run; the post-run version uses the parsed
`SpectraResults` and substitutes real numbers.

### 9.5 Explainer panel + tooltips + scientific caveats

A collapsible panel at the top of the tab (open by default on
first visit, dismissable; state in sessionStorage):

> **What this tab does** — One paragraph of plain-English
> explanation per the §1 purpose statement.  Walks the user from
> "you have a relaxed structure" through "you'll get a Raman
> spectrum + electronic-structure data per mode".  No jargon
> without a tooltip-target.

Per-field tooltips include both **what** the knob does and a
**when-to-change** hint with a citation key, e.g.:

```
displacement_amplitude_ang:
  ±A along each mode eigenvector for the per-mode electronic-
  structure SCFs.  Typical 0.05–0.15 Å — small enough that
  anharmonic-cubic mixing < 1% [Mills1972 §2.4]; large enough
  that finite-difference noise on ΔE_HOMO is suppressed.
```

**Scientific-caveat banners** (rendered as yellow `hint`
sections in the form):

* Next to `top_n` / `threshold` selectors:
  > Pruning by Raman activity may miss modes with weak Raman
  > activity but strong electron-phonon coupling.  Use
  > `explicit` (after inspecting the spectrum) or `all` when
  > the goal is transport-coupling input data.
  > [Galperin2007]

* Next to `displacement_amplitude_ang`:
  > Larger displacements probe anharmonic curvature; smaller
  > displacements increase finite-difference noise.  Default 0.10 Å
  > is a defensible production value.

* Top of the page (always visible) when `n_free > 30`:
  > This is a large free-atom set ({n_free} atoms, {3·n_free − 6}
  > vibrational modes).  Estimated time at default convergence is
  > {T} hours.  Consider fixing the metal slab via the "Frozen
  > atoms — element" multi-select if you're only interested in the
  > molecule's modes.

## 10. API endpoints

The Spectra tab is served by a new blueprint
`molbuilder/web/blueprints/spectra.py`, mounted on the same Flask
app as the Build/Modify/Watch blueprints.

| route | method | body | response | status |
|---|---|---|---|---|
| `/spectra`                  | GET  | — | `spectra.html` | 200 |
| `/api/build/schema/spectra` | GET  | — | `{ok, schema}` (extends the existing schema endpoint family) | 200 / 404 |
| `/api/spectra/render`       | POST | `{xyz, params}` | `{ok, script, methods_md, issues}` | 400 bad input · 500 render fail |
| `/api/spectra/load`         | POST | JSON `{path}` OR multipart `file=` | `{ok, results}` (the parsed `SpectraResults` as JSON) | 400 bad input · 404 missing file · 500 parse fail |

**Conventions match the Build blueprint** (see
[`../../protocols/web-api.md`](../../protocols/web-api.md)):

* Error shape: `{"ok": false, "error": "<msg>"}` for HTTP 4xx /
  5xx.
* `/api/spectra/render` returns `issues` as the same structured
  list the Build endpoints emit; the JS issues panel handles them
  identically.

The `/api/build/schema/spectra` route is added to the existing
schema dispatch (currently covers `siesta` and `pyscf`); the
dispatch dict gets one entry.

## 11. Publication-quality requirements

Per the design discussion (this conversation, 2026-05-11), the
**generated files are the single source of truth** for what was
run — a user reading them should be able to distil a Methods
paragraph without consulting external docs.

### 11.1 Inline citation keys in emitted script

Every numerical parameter in the script header + body carries a
trailing comment of the form:

```python
displacement_amplitude_ang = 0.02   # ±A·Q_i along each mode
                                    # eigenvector; 0.02 Å sits in
                                    # the linear-response regime
                                    # so ΔE_orbital/ΔA is the
                                    # physically meaningful slope.
                                    # 0.05–0.15 Å is also defensible
                                    # if you need a larger signal-
                                    # to-noise margin; anharmonic-
                                    # cubic mixing stays < 1% up to
                                    # ~0.10 Å per [Mills1972 §2.4].
```

Citation keys resolve against
[`references.bib`](references.bib) in this folder.

### 11.2 Methods-paragraph header in script

The first ~60 lines of the emitted Python script are a docstring
block containing:

* A 2-paragraph Methods-section draft with the actual parameter
  values inlined ("…analytic Hessian using `pyscf.hessian.rks`
  [Sun2020]…with B3LYP/def2-SVP [Becke1993, Grimme2011]…").
* A "selected modes" line: which modes were chosen for ES
  treatment, by what criterion.
* A bibliography listing the citation keys used in the prose +
  inline comments, in BibTeX-key form.

The same text is also served by the Methods-preview button in
the UI (§9.4) — exactly the same prose, so the user sees
identical content in both places.

### 11.3 Bibliography at `tabs/spectra/references.bib`

A BibTeX file colocated with this spec.  Each entry **verified by
the author** before the entry ships in a release (`@verified`
comment marker on each entry that has been checked).  Adding a
new citation to the script body requires adding a verified entry
to the .bib first.

### 11.4 Pre-flight scientific warnings

`SpectraEngine.preflight()` returns warn-severity `Issue`s for
scientifically dubious configurations:

* `grid_level < 4` with a hybrid functional (B3LYP / PBE0 / M06-2X
  / wB97X-D): "Hybrid functionals need DFT grid ≥ 4 for reliable
  Hessian." [Sun2020]
* `displacement_amplitude_ang > 0.20`: "Large displacement may
  mix anharmonic contributions." [Mills1972]
* `displacement_amplitude_ang < 0.04`: "Small displacement
  amplifies finite-difference SCF noise; ΔE_HOMO may be
  numerically unstable."
* `es_mode_selection ∈ {top_n, threshold}`: surface the
  Raman-bright ≠ EPC-strong caveat from §9.5.
* `n_free > 50` AND `es_mode_selection == "all"`: estimate hours
  on 8 cores, suggest atom-fixing or switching to `explicit`.

These run client-side (in the live preflight loop) AND
server-side (when `/api/spectra/render` is called) — same pattern
as Build's preflight in the existing schema-driven form.

## 12. Test contracts

Tests must be derivable from this spec without reading the
implementation (per [`../../README.md`](../../README.md)).

### 12.1 Unit tests (no PySCF runtime needed)

* `SpectraConfig` field metadata is complete (every field has
  `section`, `label`, `help`; numeric fields have `range`).
* `dataclass_to_form_schema(SpectraConfig, "sp")` produces the
  expected schema (pin section names + per-section field counts,
  same pattern as the existing SIESTA / PySCF schema pin tests in
  `tests/test_web.py`).
* Atom-fixing semantics: given a structure with a known
  `(elements, residue_names)`, `compute_free_atoms(struct, cfg)`
  returns the expected free-atom list for each filter combination
  (empty / element-only / residue-only / index-only / union of
  two / union of all three).
* Mode-selection: given a list of `(idx_1based, raman_activity)`
  tuples, the selector returns the expected subset for each of
  `skip` / `all` / `top_n` / `threshold` / `explicit`.
* `SpectraResults` JSON round-trip: parser reads back a
  hand-written `.spectra.json` byte-identical to what the parser
  produced.

### 12.2 Engine-shim tests (PySCF runtime needed; marked `@pytest.mark.smoke`)

* H₂O at HF/STO-3G: full Raman analysis (no atom-fixing,
  selector=`all`).  Three vibrational modes; bend ~1500–1700 cm⁻¹;
  sym + asym stretches ~3500–4500 cm⁻¹ (HF/STO-3G overestimates
  by ~10–15%).  Each mode has non-zero Raman activity; HOMO−LUMO
  gap shifts by < 1 eV under default A.  Just verifies the
  end-to-end pipeline.
* H₂O with `frozen_elements=["O"]`: partial Hessian on 2 atoms;
  one mode survives (the symmetric H–H pseudo-stretch).
* Small Au cluster + dithiol with `frozen_elements=["Au"]`:
  confirms slab-fixed transport-prep workflow runs end-to-end.

### 12.3 Web / E2E tests (Playwright)

* `/spectra` page loads with zero JS errors.
* Form renders all sections in `_form_section_order`.
* Sections lock/unlock via compatibility rules (e.g.
  `es_top_n` / `es_threshold` / `es_explicit_indices` inputs lock
  when the matching selector value isn't active).
* Spectrum plot renders for a fixture `.spectra.json`.
* Mode-click in the spectrum highlights the matching row in the
  mode list and triggers the 3D animation.
* "Show methods text" modal opens and shows the expected Markdown
  for a known config.
* Pre-flight cost preview updates live when form values change.

## 12.1 End-to-end numerical validation (water, B3LYP/def2-SVP)

A reference run validates the full pipeline against an independent
hand-written raw-PySCF script (no molbuilder code):

* **Starting point**: same MMFF water from
  ``molbuilder.build_from_smiles("O")`` -- O at
  ``(0.001, 0.398, 0)``, H at ``(±0.763, -0.199, 0)``.
* **Same operations**: RKS / B3LYP / def2-SVP / DF, geomeTRIC
  optimization, analytic Hessian, central-difference
  ``dα/dR`` via pyscf-properties polarizability, Placzek
  ``45 a² + 7 γ²`` * ``BOHR_TO_ANG⁶``.

| Quantity                | molbuilder            | raw-PySCF             | max Δ              |
|-------------------------|-----------------------|-----------------------|--------------------|
| Relaxed positions       | ``-76.35832575`` Ha   | ``-76.35832575`` Ha   | 1.1×10⁻⁷ Å         |
| ν₁ bend                 | 1638.77 cm⁻¹ / 6.816 Å⁴/amu | 1638.77 cm⁻¹ / 6.816 Å⁴/amu | < 10⁻³ cm⁻¹, < 10⁻⁶ Å⁴/amu |
| ν₂ sym stretch          | 3791.22 cm⁻¹ / 76.905 Å⁴/amu | 3791.22 cm⁻¹ / 76.905 Å⁴/amu | < 10⁻⁵ cm⁻¹, < 10⁻⁶ Å⁴/amu |
| ν₃ asym stretch         | 3886.54 cm⁻¹ / 36.818 Å⁴/amu | 3886.54 cm⁻¹ / 36.818 Å⁴/amu | < 10⁻³ cm⁻¹, < 10⁻⁶ Å⁴/amu |

Frequencies match literature B3LYP/def2-SVP water (~1638 / ~3791 /
~3887 cm⁻¹); Raman activities fall in the standard Å⁴/amu range
expected at this level of theory, confirming the ``BOHR_TO_ANG⁶``
units conversion is correct.

This is documented here -- not in code -- because the comparison
script is one-shot validation (not a unit test).  Adding it as a
unit test would require pyscf + pyscf-properties in the test env,
which we've kept out of the host env on purpose.

## 13. Future extensions

### 13.1 IR add-on — scaffold landed; absolute magnitudes NOT YET VALIDATED

**Status (2026-05-15):** the IR pipeline is wired end-to-end.  Setting
`compute_ir=True` (which requires `compute_raman=True` in v1) populates
`ir_intensity_km_mol` on every mode with values in **km/mol**.

**How the math works.**  PySCF's `mf.dip_moment(unit='Debye')` is
essentially free after an already-converged SCF, so each of the
6·N_FREE displaced SCFs that Raman is already running for `dα/dR`
also yields `μ(±δ)`.  The script captures both in the same loop,
forms `dμ/dR_kα = (μ(+δ) - μ(-δ)) / (2δ)`, projects onto each
canonical mass-weighted normal mode `L_canonical`:

```
dμ/dQ_n  =  Σ_{k,α} (dμ/dR_{k,α}) · L_canonical_{k,α,n}     (3-vector)
```

and applies the standard textbook IR-intensity formula:

```
I_n  =  K · |dμ/dQ_n|²
K   =  N_A · π / (3 · c²)  ·  (D/Å)² / amu  →  km/mol
    =  42.2561 km/mol per (D/Å)²/amu     (Gaussian / ORCA / psi4 value)
```

The 42.2561 constant is the same one quoted in the Gaussian
whitepaper on IR intensities and the ORCA manual.  Code reference:
`molbuilder/spectra/pyscf_script.py::_emit_ir_projection`.

**Cost.**  Zero extra SCFs.  IR is computed inside the existing
Raman FD loop; the only added work per displacement is one
dipole-moment integral on the already-converged `mf` (microseconds).

**What's NOT validated yet.**  The Raman path was cross-checked by
running molbuilder's full pipeline AND a hand-written raw-PySCF
script (see § 12.1) and showing bit-for-bit agreement plus
literature-ballpark match on the absolute scale.  **The IR path has
not yet been through that exercise.**  The projection math + the
prefactor are textbook-correct; what's unverified is whether
PySCF's `dip_moment()` returns the dipole in the convention this
formula assumes (origin choice, sign, electron vs nuclear convention)
and whether the 42.2561 constant matches PySCF's specific Debye
convention to better than ~1%.

**v1 constraint.**  `compute_ir=True` raises a clear `ValueError`
at script-render time if `compute_raman=False`, on the grounds that
a standalone IR path would either duplicate the displaced-SCF cost
or change the script's structural shape — neither earns its
complexity until a user actually needs IR without Raman.  Same
unit-conversion machinery + the same Hessian → canonical
mass-weighted modes are reused; nothing IR-specific touches the
science layer.

**To finish the IR add-on:**

1. Run the equivalent of § 12.1 with `compute_ir=True`: a water
   B3LYP/def2-SVP comparison against an independent hand-written
   IR script (raw-pyscf dipole derivatives + same prefactor) AND
   against a published Gaussian/ORCA IR table for the same level
   of theory.  Acceptance bar: < 2% on absolute IR intensities,
   identical relative pattern.
2. If the absolute match fails: trace whether (a) PySCF's dipole
   convention differs from the assumed one, (b) the prefactor needs
   a different precision, (c) there's an origin / units bug in the
   projection.
3. Add a `tests/spectra/test_smoke.py` IR smoke check (gated on
   `compute_ir=True` config + pyscf available) that the field is
   non-None, finite, and > 0 for at least one mode.
4. Drop the "NOT YET VALIDATED" banner from the script header +
   the help text on `compute_ir`; move the discussion above into
   a § 12.2 IR validation entry mirroring § 12.1.

UI work for the spectrum plot toggle ("Raman / IR / Both") can
proceed against the current scaffold data (the JSON has the field
populated; values are usable for relative-intensity comparisons and
qualitative analysis right now, just don't quote absolute km/mol in
a publication until step 1 above is done).

### 13.2 SIESTA engine

`molbuilder/spectra/siesta_engine.py` implements `SpectraEngine`
for the SIESTA force-constant workflow (`MD.TypeOfRun = FC`,
finite-difference dipoles for IR, external tool — likely
`vibrana` — for Raman activities).  Significantly more work
because SIESTA's vibrational path is FD-based and Raman activities
require post-processing.  Reserved choice in `SpectraConfig.engine`
gets the new value; everything else is automatic.

### 13.3 Methods extractor CLI

`molbuilder spectra methods <output_dir>` — reads the script +
the `.spectra.json` from a completed run and emits a Markdown
file with the Methods paragraph + bibliography (BibTeX, only the
entries actually cited).  Pairs with the equivalent Transport
extractor sketched in the Transport-tab spec (yet to be written).
Useful when the user has run several configurations and wants
machine-curated Methods drafts.

## 14. References

The bibliography file [`references.bib`](references.bib)
contains the cited works.  Candidate entries to seed (each
verified by the author before the v1.2 release tag — see § 11.3):

* Wilson, Decius, Cross 1955 — *Molecular Vibrations* (canonical
  text)
* Sun et al. 2020, J. Chem. Phys. 153, 024109 — PySCF capabilities
* Komornicki & Fitzgerald 1993, JCP 99, 1398 — analytic dα/dR
* Mills 1972 — anharmonicity bounds on displacement
* Galperin, Ratner, Nitzan 2007, JPCM 19, 103201 — vibrational
  effects on molecular conductance
* Frederiksen et al. 2007, PRB 75, 205413 — electron-phonon
  coupling from DFT for molecular junctions
* Becke 1993, JCP 98, 5648 — B3LYP
* Grimme et al. 2011, JCC 32, 1456 — D3BJ

Adding a new citation to the spec or to the emitted script
requires adding the verified BibTeX entry to `references.bib`
first.
