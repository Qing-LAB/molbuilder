# F12 review, round 1 — slice F: the vibration calculation (both engines)

Checkout: `the checkout` at f578482db (2.77.2). Read only; nothing in the checkout was written.

**Count: 2 defects, 3 untrue sentences.** All of them are low severity. On the roads a person uses (the PySCF deck, the SIESTA run of points with its plan, record and finish-run steps, the reader, the views, the export, the sweep, the frame generator), the code does what the chapters say. None of the findings changes a frequency, an intensity, a Raman strength or a thermochemistry number.

## What was read

- **Chapters, end to end:** `engines/vibration.md` (all 3917 lines), `engines/vibration-postprocessing.md`, `science/normal-modes.md`, `science/vibrational-averaging.md`, `science/vibrational-optical-response.md`.
- **Code, every function:**
  - `spectra/`: `activity`, `datafile`, `derived`, `displacement_sweep`, `export`, `infrared`, `__init__`, `methods`, `mode_frames`, `normal_modes`, `optical_data`, `pyscf_vibration`, `raman`, `results`, `selection`, `siesta_rung`, `siesta_vibration`, `vibrational_analysis`, `vibration_points`, `vibration_sum`.
  - `pyscf/vibration_deck.py` and `pyscf/vibration_emitters.py`.
  - `frame_average.py` and `frameset.py`.
  - `sidecars/__init__.py`, `sidecars/spectra.py` and `sidecars/molstruct.py`. In `molstruct.py`, `apply_to_structure` and `to_dict` were read in full; the rest was read for what the slice's chapters claim.
- **Callees followed only to trace a claim:** `parse/engines/siesta_fc.py`, `atom_permutation.Permutation` / `map_rows_to_user_order` / `AtomIndexMap`, `runpoints.rows_of`, `siesta/vibration_phases.Call` / `point_entry`, `structure.ATOM_METADATA_FIELDS` / `apply_atom_metadata`, `script_emit.emit_atom_metadata`, `pyscf/input._sidecar_for`, `deck_record.extract_vibration_record`, `parse/fdf.read_deck`, `runs.run_of` / `on_label_at`, `persist.check_keys` / `json_text`, and `postprocess`'s signatures.
- **Read-only checks run in Python:**
  - The rank rule's table rows: CO₂ free 5, CO₂ with both O held 0, water with O held 3, water with O and one H held 1, a lone atom 3, a wire 4, a slab 3, H₂ with one atom held 2.
  - The Gauss–Hermite ordering and weights.
  - The constants: 42.2561; 4.8032; 5140.487; 16.858; Q_zp at 1000 cm⁻¹ = 0.130.
  - The engine→user index mapping on a held-first CO₂ permutation.
  - The remedy text `_stationarity` writes when no relaxation record is passed (D1).

---

## Defects

### D1 — The by-hand `finish` verb misstates a run that relaxed, and gives the wrong remedy

**Where:**
- `molbuilder/spectra/siesta_vibration.py:572-580` (`_finished`), with the verb's printout at `:1324-1337`.
- `molbuilder/spectra/vibrational_analysis.py:218-219`, `:352-358` and `:289-294`.

**What the chapter says:** `engines/vibration.md` § 5.5, *When the finish fails*, names this verb as the recovery road: "The analysis can be run by hand in the force-constant call's folder … `mb_vibration.pyz finish <deck> <output>` — which writes the spectrum beside that call's deck in seconds". The same section states the remedy rule (line 1671 on): "when this stage's run relaxed it, give it more relaxation steps (`relax_steps`) … ; when the person stated it relaxed, relax first".

**What the code does:** `_finished` calls `res = result_of(run)` with no `relaxation`. The deck's `vibration` block always states `"relaxation": None` (`vibration_record`, `:202`). So `vibrational_analysis` receives `ladder_relaxation=None`, and:

```python
phase_relaxation=(PHASE_COMPLETE if ladder_relaxation is not None
                  else PHASE_NOT_REQUESTED),                        # :218
...
nonstationary_remedy(
    relaxation_stage if isinstance(ladder_relaxation, Mapping) else None, ...)   # :352
return {"enabled": ladder is not None, ...                          # :357
```

`nonstationary_remedy(None)` returns "Relax first -- … then untick `already_relaxed` (Task setup, or task.json) so the calculation relaxes first …" (`:289-294`). The verb then prints that remedy as a WARN (`:1336-1337`). `_with_raman_phase` is also never called on this road, so a run that asked for Raman is written as `phase_raman: not requested`, `raman_route: none`.

**What goes wrong:** Take a SIESTA vibration with the box unticked. Its run relaxed (its `relax` call ran), and its record step failed afterwards, for example on the job env. The person follows § 5.5 and runs `finish` by hand:
- The spectrum written beside the deck says `phase_relaxation: "not requested"` and `relaxation.enabled: false`.
- If the reference forces are above `relax_force_tol` (for example, the relaxation hit its step cap), the session log and the file's `relaxation.warning` tell the person to untick `already_relaxed`, which is already unticked. The right remedy, per § 5.5, is more `relax_steps`.
- If the run asked for Raman, the file states that Raman was not requested.

Measured: `_stationarity(forces, 2, [0], 0.01, 'note', already_relaxed=False, ladder=None, None, 'freq')` returns `enabled False` and the "untick `already_relaxed`" text.

**Confidence:** Read and traced. It only reaches the by-hand recovery road; the job's own `record` and `finish-run` steps pass the `relax` call's record from the index (`_force_constants`, `:1152-1157`).

### D2 — A PySCF frame set never reads a stated-relaxed structure's `info.calculation`

**Where:** `molbuilder/spectra/mode_frames.py:102-110` (`_relaxation_record`).

**What the chapter says:** `engines/vibration.md` § 5.10 ③, the *What it reads* table, row for frame 0's `info.calculation`: "where the vibration's own run relaxed it — its `relax` call — that run's settings …; **where the structure was stated already relaxed (§ 2.2), the record that structure states — its `info.calculation` … — read from the calculation's own structure beside it.**" `science/normal-modes.md` § 4b.8 says the frame set is "built for transport, on both engines".

**What the code does:**

```python
record = (extract_vibration_record(read_deck(deck)) if deck is not None else None)
if not (record or {}).get("already_relaxed"):
    own = (view_of(run).info or {}).get("calculation")
    return own, ("" if own is not None else
                 f"{run.label}'s deck records no settings (a PySCF deck states none), ...")
```

"Stated relaxed" is decided only from the SIESTA force-constant deck's `vibration` block. A PySCF deck is a Python script and has no such block (`extract_vibration_record` returns `None` for "every deck but a SIESTA force-constant one", `deck_record.py:191-199`). So every PySCF run takes the "own run" branch, whatever its `already_relaxed` says, and the structure beside the calculation is never read.

**What goes wrong:** A PySCF vibration with `already_relaxed` ticked, on a structure that carries `info.calculation` (for example, one exported from a SIESTA relaxation). *Write frame set…*, or `molbuilder frameset from-mode`, writes a set with no `info.calculation`. It says "…'s deck records no settings (a PySCF deck states none), so this set states no info.calculation and is no transport citation" instead of carrying the structure's record, as the rule asks.

**Confidence:** Read and traced. It is an unusual road (PySCF modes feeding a transport citation).

---

## Untrue sentences

### U1 — "`info` — as the run was given it" is false for a SIESTA result

**Chapter:** `engines/vibration.md` § 6.10.1, table *The structure, whole*, line 3410: "| `info` | as the run was given it: **the input's own run records stay behind** — its `relaxation` and `calculation` … (`pyscf/input.py::_sidecar_for`) |".

**What the code does:** The SIESTA finish builds the result's structure from the deck's blocks:

```python
struct = Structure(elements=list(run.elements), positions=..., title=run.title,
                   **{k: ... for k, v in run.identity.items()})              # siesta_vibration.py:382-387
struct.apply_metadata_dict({**struct.metadata_to_dict(), **box})
struct.apply_atom_metadata(run.atom_metadata or {})                          # :389-390
```

The ATOM-METADATA block carries `ATOM_METADATA_FIELDS = FILE_FIELDS − BOX_FIELDS` = regions, annotations, customized and atom_order — no `info` (`structure.py:156-178`, `script_emit.emit_atom_metadata:333-337`). The `vibration` block adds only the title and the identity columns (§ 6.10.1, *How each writer builds it*, confirms this). So on SIESTA the result's `structure.info` is always empty. Every `info` key the input carried is dropped, not only its run records. Only the PySCF route (`_sidecar_for`) keeps the input's `info` minus `relaxation` and `calculation`.

**Effect:** A person who saves a SIESTA vibration's equilibrium structure from the Results tab loses their own `info` notes. The chapter says they are kept. Either the row needs "on PySCF; on SIESTA none", or the SIESTA deck needs to carry `info`. This is a design choice for the user.

**Confidence:** Read and traced.

### U2 — The vibrational-only note's "why" does not name SIESTA's reason

**Chapter:** `engines/vibration.md` § 4.7, lines 1115-1118: "The file's `note` says this beside the numbers, on both engines and in both regimes: … and, for a vibrational-only answer, why it is one — atoms held, or a free molecule whose full RRHO PySCF could not give."

**What the code does:** The SIESTA route (and any route through `vibrational_analysis`) always writes:

```python
why="this route hands in the force constants alone and carries no electronic energy"
```

(`vibrational_analysis.py:175-179`). This holds whether atoms are held or not. Only the PySCF deck writes "atoms are held…" or "PySCF thermo could not give the full RRHO…" (`vibration_deck.py:436-482`).

**Effect:** A SIESTA result with atoms held never says "atoms held". The chapter's list of reasons is PySCF's only. The code's reason is accurate; the sentence is what is incomplete.

**Confidence:** Read and traced.

### U3 — `relaxation.record` is listed as written by "both"; PySCF never writes it

**Chapter:** `engines/vibration.md` § 6.2, line 3106: "| `relaxation.{enabled, already_relaxed, n_steps, record, max_force_eh_bohr, …}` | both | …".

**What the code does:** The PySCF deck's `relaxation` block is `{'enabled', 'already_relaxed', 'n_steps', 'max_force_eh_bohr', 'max_force_all_atoms_eh_bohr', 'converged', 'warning'}` (`vibration_emitters.py:586-590`). No `record` key is ever added; it is absent, not `null`. Only `vibrational_analysis._stationarity` (`:363`), the SIESTA route, writes `record`.

**Effect:** A person reading a PySCF file, or a script doing `rx["record"]`, finds no such key.

**Confidence:** Read and traced. Minor.

---

## Checked against the code and found consistent (no finding)

- **The one harmonic path** (`normal_modes`):
  - The rank rule, a cell required for non-isolated axes, the 10⁻³ Å tolerance and the geometric centre over all atoms.
  - Projection in the mass-weighted metric before `eigh`, 3·N_free − n_rigid modes, and the canonical Σ m|L|² = 1.
  - The display form, the 5140.487 conversion, the thermo sums (ZPE at T ≤ 0), the grid holding the headline T, `pressure_atm: null` for vibrational-only, `largest_move` / `q_for_largest_move` / `displaced`.
- **The PySCF deck:**
  - Mass check against `atom_mass_list(isotope_avg=True)`; relaxation with `$freeze` and per-step writes; `converged` judged on the free-atom component against `geom_gmax`; the gradient check under `already_relaxed` (free and all-atom figures).
  - The equilibrium SCF halting on non-convergence; the reduced Hessian (`hess_elec + hess_nuc + get_dispersion()[F,F]`, placed by `np.ix_`, a non-DF rebuild, `hessian_density_fit`).
  - The analytic IR rule (IR only, nothing held), the injected `mf.Hessian()` plus dispersion, D/Å conversion = e; Raman finite differences, non-DF and on the CPU, with (Bohr/Å)⁶; the IR projection `einsum('kai,ka->i')` and the vector stored (v12).
  - RRHO with `_OMEGA_AU` in PySCF's own frequency unit; the ES loop on the display form, 2 SCFs per mode, windows padded and written null, dropped explicit numbers said; phase flags and `not requested` from the first write; the Methods fragment.
- **The SIESTA job steps (`siesta_vibration`):**
  - The trailing-range check, the deck and output atom agreement, `.FC` row and range checks (`siesta_fc`), FC step 0 as the reference.
  - Masses, temperature and `raman` refusals naming *prepare again*; the held-set double statement check.
  - The user order through the deck's record (eigenvectors, patterns, charges, optical frames); `partial_charges` steps in SIESTA's order.
  - `_plan_optical` (δQ = δ / max|L|, frames R₀ ± δQ·L in deck order, entries and `.continued-from`), `_with_raman_phase` states, `_optical_payload` frame states by the one done rule, finish-run's rewrite.
  - The criterion taken over from the first stage's record for a sweep's later stage.
- **The reader (`results`):** every refusal listed in § 6.7, including the derived-value check (rtol 10⁻⁹) and the partition without building `range(n)`.
- **The data files and views:**
  - `datafile` key gates and round trip; `optical_data` grid and spin refusals.
  - `infrared` (centre of mass of the free atoms, central differences, `3K(ê·v)²` held, `K|v|²` free, rank against real modes above the floor, `asked` / `no_charges_reason`).
  - `raman` (convolution width √(σ² − σ_run²), the principal-value split with pole subtraction, the static integrand from e > 0, complex α, refusals and side states, the `raman.nc` table).
  - `export` refusals.
- **The sweep and frame sets:**
  - `displacement_sweep`: matching by shape, geometry within 10⁻⁶ Å with the two remedies, H(δ) in eV/Å² over the nudged block, pending versus failed.
  - `mode_frames`: Gauss–Hermite nodes, positive side first, σ = Q_zp√(2n+1), the outermost move, the rows' vocabulary, `info.vibration`, an existing pair refused.
  - `frame_average.motion_weights` / `reaveraged` (moments, the swing reach refused, a negative weight said).
  - `frameset` weights.
- **vibration-postprocessing.md:** every call and key in its examples exists with the stated shape.
