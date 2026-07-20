# Data vocabulary & persisted formats — the system's shared language

> **Concentrated, authoritative definition** of the names and JSON formats
> molbuilder subsystems use to **exchange key information**. The rule: if
> two files name the same concept, they use the name defined here. Every
> persisted artifact follows the schema convention here. This doc is the
> place to look up "what is this field called, system-wide?" — the other
> docs reference it rather than re-defining names.
>
> Maintained because the names DID drift (e.g. a job-set field once read
> `omp`/`walltime` while every other exchange file said `cpus_per_task`/
> `time`, 2026-06-30). One language prevents that.

---

## § 1 Persisted artifacts (the files that carry key info)

| Artifact | File | Schema string | Authoritative doc | Key top-level fields |
|---|---|---|---|---|
| User config | `molbuilder.json` / `.molbuilder.json` | *(validated, no `@N`)* | `config.md` | `scheduler{directives,gpu,defaults,routing}`, `execution`, `script_generation`, `envs` |
| Detected environment | `environment.json` | `molbuilder/environment@1` | `benchmark-workflow.md`, `bench/environment.py` | `scheduler`, `topology`, `site` |
| Benchmark manifest | `bench-manifest.json` | `molbuilder/bench-manifest@2` | `benchmark-workflow.md`, `bench/generate.py` | `points.{cpu,gpu}` |
| Benchmark result | `bench-result.json` | `molbuilder/bench-result@1` | `benchmark-workflow.md`, `bench/result.py` | `points`, `choice`, `recommend` |
| **Job-set plan** | `job-set.json` | `molbuilder/job-set@1` | `staged-execution.md` | `name`, `engine`, `kind`, `shared`, `jobs[]` |
| Decoded run | `decoded.json` | `schema_version: <int>` *(predates the `@major` convention)* | `job-decoder.md` | `schema_version`, decoded plots, job-type, triggers |
| Workflow handoff | `<stem>.xyz` + `<stem>.molstruct.json` | *(sidecar pair)* | `bundle-contract.md`, `bundle_writer.py` | geometry; `regions`/`frozen_atoms`/`structure_hash` |
| Checkpoint binary archive | `.binsnapshots/<sha>/MANIFEST` | *(3-col `<sha256> <bytes> <name>`)* | `run-checkpoints.md` § 10 | — |
| Checkpoint config | `.mbcheckpoint.json` | `molbuilder/checkpoint-config@1` | `run-checkpoints.md` § 9 | `engine`, `archive_globs` (the engine-specific, user-editable big-binary classification) |

**Schema-string convention.** `molbuilder/<name>@<major>`. Readers check the
**major only** (tolerate same-major minor bumps, reject a different major) —
enforced by the single shared helper **`molbuilder/persist.py::check_schema_major`**
(with `schema_major` + `read_json`/`write_json`), adopted by
`bench/environment.py`, `bench/result.py`, and `jobset/model.py` (was
hand-rolled 3× with a subtle missing-`@` inconsistency). New persisted
artifacts MUST use it. The one exception is `decoded.json`, which predates
the convention and carries a bare integer `schema_version` (`job-decoder.md`);
not worth a breaking change, but don't copy that pattern for anything new.

---

## § 2 The canonical parameter vocabulary

There are **two layers** with a deliberate, documented translation between
them. Within a layer, **one concept = one name**:

- **config layer** — the scientific dataclasses the user sets
  (`SiestaConfig` / `PySCFConfig`). Vocabulary tuned for the scientist.
- **exchange/scheduler layer** — the persisted artifacts (manifests,
  `job-set.json`) + the SLURM flags they become. Vocabulary tuned for the
  scheduler. Persisted files and `jobset.Resources` use THIS column.

| Concept | config-layer name | exchange / SLURM name | producer translates at |
|---|---|---|---|
| MPI ranks | `mpi_np` | `mpi_np` → `-n` | *(same name)* |
| OMP cores per rank | `omp_threads` | **`cpus_per_task`** → `-c` | `stages_to_jobset` (jobset) / CLI+`build` callers (single-job) — see note |
| Walltime | *(none; `defaults.time`)* | **`time`** → `-t` | — |
| Memory | `max_memory_mb` (cap) / `defaults.mem` | `mem` → `--mem` | `render_sbatch` (estimate) |
| Whole-node | `gpu.exclusive` | `exclusive` → `--exclusive` | — |
| Partition | `directives.partition` | `partition` → `-p` | resolved from `domain` |
| QoS | `directives.qos` | `qos` → `-q` | resolved from `domain` |
| Routing domain (named menu pick) | `scheduler.routing[].name`, `execution.domain` | `domain` (in `jobset.Resources`) | `--domain` → `-p`/`-q` |
| GPU request | `enable_gpu` + `diag_algorithm` | `gres` → `--gres` | derived from `.fdf` + GPU type |
| Eigensolver | `diag_algorithm` (`ScaLAPACK`/`ELPA-1STAGE`/`ELPA-2STAGE`) | `.fdf`: `Diag.Algorithm` | `render_fdf` |
| Non-convergence policy | `on_nonconvergence` (`proceed`/`continue`/`halt`) | `dep_kind` (`afterok`/`afterany`) | `stages_to_jobset` (§ 8) |

**The translation rule:** persisted/exchange files use the exchange-layer
name; a *producer* maps config→exchange at its boundary (e.g.
`stages_to_jobset` maps `SiestaConfig.omp_threads` → `cpus_per_task`). Never
mix the two within one file. `render_sbatch` is a *consumer* — it receives
`cpus_per_task` already translated; it does not re-derive it from
`omp_threads`. (In `runwrap` these are two distinct knobs that coincide on
SLURM, where `-c` sets `SLURM_CPUS_PER_TASK`, the wrapper's OMP default; the
"one concept" framing is the SLURM mapping, not a Python rename.)

---

## § 3 Identifier & path conventions

| Convention | Form | Used for |
|---|---|---|
| **Project ID** | SIESTA `SystemLabel` / PySCF `JOB = "…"` | keys warm-restart files `<ID>.<ext>` (`script-execution.md`) AND the SLURM `-J` job name |
| **Warm-restart files** | `<ID>.XV` / `.DM` / `.CG` (SIESTA); `<ID>.chk` / `<ID>_optimized.xyz` (PySCF) | engine-native resume (`script-execution.md`) |
| **Per-job directory** | `point-<name>/` | benchmark `point-G<g>K<k>C<c>/`; stage ladder `point-stage<N>/` (`staged-execution.md`) |
| **SLURM job name** | `-J <ID>` (single) / `-J <ID>-G<g>K<k>C<c>` or `-J job-stage<N>` (per-job) | `squeue` differentiation (`slurm-integration.md` § 4.4) |
| **Dependency kind** | `afterok` / `afterany` | stage chaining (`staged-execution.md` § 8) |

### § 3.1 Atom index base (0-based internal, 1-based user-facing)

Atom indices use **two bases with a single explicit conversion boundary**. This
is deliberate: arrays/JSON are 0-based by nature; scientists count atoms 1-based
(SIESTA `.fdf`, PDB serials, counting `.xyz` lines). Mixing them silently is the
classic off-by-one hazard.

| Layer | Base | Examples |
|---|---|---|
| **Internal / machine** | **0-based** | Python `Structure` (`regions`, `frozen_atoms`, positions), the `.molstruct.json` sidecar + `.fdf`/`.py` ATOM-METADATA JSON, `/api/selection/*` rules (`by_index_range`, `by_region`, …), the JS selection store `atom.index`, all wiring (`data-atom-index`, pick indices) |
| **User-facing / scientific** | **1-based** | everything a user *reads or types*: the selector atom-list index column, the 3D viewer's atom-index labels (auto + picked), measurement chips, tooltips, the "By atom index" filter input |
| **Engine input files** | **1-based** | SIESTA `.fdf` `AtomicCoordinates` order + `%block Geometry.Constraints` (native SIESTA) |

**The conversion boundary — the ONE rule:** convert only at the user-facing edge.
`display = internal + 1`; parse user input with `input − 1`. Never let a 1-based
value into internal state, and never show a 0-based value to a user. The JS
helper `lib/workspace/_atom-index.js` (`toDisplay` / `fromDisplay` /
`shiftExpression`) is the single implementation of this rule for the web UI;
standalone embeds (`mol-viewer-embed.js`) inline `+ 1` at the label with a
reference back to this section.

### § 3.2 Atom index provenance & translation boundaries

The atom index is a **safety-critical identity**: the atom a user acts on in
the UI must be the same physical atom (element + position) in the generated
engine input. This maps the whole chain — where the identity is *defined*, how
it's *carried*, and the **only** points where it is *translated*.

**DEFINED (the fact).** The canonical identity is the **0-based index into
`Structure`** (`elements[i]` / `positions[i]`), fixed by the **atom order in the
source file** when parsed (`.xyz`/`.pdb`/… → `Structure`). Nothing invents an
index; that order *is* the identity.

**CARRIED (0-based, unchanged).** The identity travels 0-based and untranslated
through: the JS selection store (`atom.index`), `/api/selection/*` rules, the
`.molstruct.json` sidecar + the `.fdf`/`.py` ATOM-METADATA block, and all
metadata (`frozen_atoms`, `regions`, `annotations`). These indices are only
valid against the structure they were computed on — **pinned by
`structure_hash`**; a mismatch must refuse, not mis-apply.

**TRANSLATED (the only three boundaries).**

| Boundary | Direction | Single API |
|---|---|---|
| internal → **display** | 0-based → 1-based | `_atom-index.js` `toDisplay` (frontend) |
| user input → internal | 1-based → 0-based | `_atom-index.js` `fromDisplay` / `shiftExpression` (frontend) |
| internal → **engine** | 0-based → engine convention | **`engine_atom_index.py`** (backend, engine-facing layer) |

`engine_atom_index.py` is the *sole* place a 0-based identity becomes an engine
atom number, per-engine: `siesta_atom_index` (SIESTA `.fdf`, **1-based**),
`geometric_atom_index` (geomeTRIC `$freeze`, **1-based**), `pyscf_atom_index`
(PySCF `mol.atom`, **0-based**). No other code applies a bare `i+1`.

**Load-bearing invariant.** Engine coordinate blocks emit atoms in **internal
`Structure` order** (no reordering), so engine atom `siesta_atom_index(i)` is
the coordinate line for internal atom `i`. The display convention is chosen so
`toDisplay(i)` **equals** the engine atom number the user reads in the file
(SIESTA `.fdf`, geomeTRIC `$freeze`) — bound by
`test_engine_atom_index::test_frontend_display_matches_engine_atom_number`.
End-to-end element+position tests bind the full user→engine round-trip.

---

## § 4 How to use this doc

- Adding a field to a persisted artifact → use the **exchange-layer** name
  from § 2 (or add a row if it's a new concept). Don't invent a synonym.
- Adding a new persisted artifact → follow the § 1 schema convention and
  add a row.
- A producer that reads config and writes an exchange file → it is the
  translation point; cite this doc at that boundary.


## § 5 The structure-metadata contract — ONE get/set, no drift

The **structure metadata** (periodicity + selection tags + per-atom annotations) that rides in
the `.molstruct.json` sidecar and reaches the MolView viewer has exactly **one** serialization
authority: **`molbuilder.Structure` itself**. This section is the rule every caller MUST comply
with when adding or removing a metadata key.

### § 5.1 The two methods that ARE the contract

```
Structure.metadata_to_dict()  -> dict     # struct  -> JSON metadata dict   (THE writer)
Structure.apply_metadata_dict(dict) -> None  # JSON metadata dict -> struct (THE reader)
```

- **Scope** = the dataclass's own fields: `regions`, `frozen_atoms`, `cell`, `cell_origin`,
  `pbc`, `axis_kind`, `vacuum`, `annotations`.
- **Strict type** = the dict is **JSON** (lists / dicts / bools / floats). `annotations` are JSON
  channel dicts, **never `AtomChannel` objects** — `AtomChannel` lives only in-memory on the
  struct; serialize a live map with `annotations_to_json(...)` before it crosses the boundary.
- `apply_metadata_dict` is **full-replace**: an absent key resets that field to its default
  (absent `cell` → non-periodic; absent `regions` → none). It re-runs `Structure.__post_init__`,
  so **all field validation lives in one place** (the dataclass invariants) — there is no second
  validator to drift from.
- **NOT in scope** (they sit *around* the contract, not inside it): `selection_rules` (a
  sidecar-only pass-through, not a Structure field) and the sidecar **envelope**
  (`schema_version` / `n_atoms_total` / `structure_hash` / `created_by` / `created_at`).

### § 5.2 The data flow (one authority, every direction)

```
                         Structure  (in-memory SSOT; AtomChannel objects live here)
                            │  ▲
        metadata_to_dict()  │  │  apply_metadata_dict(dict)   ← the ONLY struct⇄dict crossing
            (JSON out)      ▼  │      (JSON in, __post_init__ validates)
                    ┌───────────────────┐
                    │  JSON metadata dict│  (regions, frozen_atoms, cell, cell_origin,
                    │   = §5.1 field set │   pbc, axis_kind, vacuum, annotations)
                    └───────────────────┘
                    ▲     ▲            │ ▲
   sidecars.to_dict │     │ workingcopy│ │ parse ._normalised_dict / apply_to_structure
   (+envelope,      │     │ _sidecar_  │ │  (both route through structure_fields_via_dataclass
    selection_rules)│     │ dict       │ │   → a scratch Structure = the schema)
                    │     │            ▼ │
          .molstruct.json on disk  ◄────┘         web structure_to_dict → periodicity block
                    │                                          │
                    └──────────────────────────────────────────▼
                                            molview.data  →  the 3-D viewer + panel
```

**The invariant:** the field list is enumerated **once** (the two methods above). `to_dict` and
the read-side `_normalised_dict` both validate by *round-tripping through a scratch Structure*
(`structure_fields_via_dataclass`), so a field cannot exist on the write side but not the read
side. (This is the exact bug that silently dropped `cell_origin` on reload before the contract:
the write path knew the field, the read path didn't.)

### § 5.3 How to ADD a metadata key

1. Add the field to the `Structure` dataclass (`structure.py`) **with its `__post_init__`
   validation + reconciliation** — this is now the *only* place the field is validated.
2. Add it to **`metadata_to_dict()`** (serialize → JSON) and **`apply_metadata_dict()`**
   (deserialize + assign). Nowhere else.
3. If it must survive the sidecar, bump `SCHEMA_VERSION` and add the new version to the read
   module's `_READABLE_SCHEMA_VERSIONS`.
4. If MolView must display/edit it, surface it in the web `structure_to_dict` periodicity block
   and read it in `molview.data`.
5. Add a **save → load → apply round-trip test** (see `test_cell_origin_survives_the_disk_roundtrip`).

You do **not** touch `to_dict`, `_normalised_dict`, `apply_to_structure`, or `_sidecar_dict` —
they read the field set from the two methods, so they pick it up for free.

### § 5.4 How to REMOVE a metadata key

Delete it from the dataclass + `metadata_to_dict` + `apply_metadata_dict`. Old sidecars that
still carry it load fine — `apply_metadata_dict` ignores unknown keys. (Do NOT leave a "read it
but never write it" half-migration; that is what the contract exists to prevent.)

### § 5.5 Worked example (`cell_origin`)

```python
# WRITE: a live struct -> sidecar dict (annotations already JSON via metadata_to_dict)
payload = molstruct.to_dict(struct.metadata_to_dict(),
                            n_atoms_total=struct.n_atoms, structure_hash=h)
molstruct.save(path, payload)

# READ: sidecar -> struct (one crossing; __post_init__ validates)
loaded = molstruct.load(path)              # -> normalised JSON dict, cell_origin preserved
molstruct.apply_to_structure(back, loaded) # -> back.cell_origin == the saved value
```

Never hand-build the field kwargs; never pass an `AtomChannel` into `to_dict`/`apply_*`.
