# Generated-Script Contract

**Status:** v1, 2026-06-16
**Scope:** the structural format that molbuilder-generated `.fdf`,
`.py`, and `.run.sh` files must honor so downstream tooling (bench,
re-render, TranSIESTA, results-tab parsers) can rely on them.

## 1. Purpose

molbuilder generates engine input scripts that get copied from the
edit directory to project execution directories and travel onward.
Today, label metadata (region tags, frozen atoms, selection
provenance) and version provenance (which molbuilder produced the
file, when, with what auto-resolved defaults) is lost at that copy
step: the `.molstruct.json` sidecar stays behind, and the `.fdf` /
`.py` carry only what the engine itself needs.

This contract solves that by reserving comment-block regions of the
generated file for molbuilder's own use, plus a clearly marked
user-custom zone the user owns. The result:

- `head -50 file.fdf` answers "which molbuilder version made this,
  with what defaults".
- A `.fdf` carries the same label metadata as the `.molstruct.json`
  that produced it. No sidecar coordination required.
- Tools (bench, TranSIESTA) read from a contract surface, not by
  scraping the engine body.
- User edits in the user-custom zone survive regeneration.

## 2. Files in scope

| File type | Engines        | Comment prefix |
|-----------|----------------|----------------|
| `.fdf`    | siesta, transport (transiesta) | `#` |
| `.py`     | pyscf          | `#` |
| `.run.sh` | all            | `#` |

The same comment-block grammar applies in every file — they differ
only in what blocks are populated (e.g., `.run.sh` does not carry
an atom-metadata block; only `.fdf` and `.py` do).

## 3. File structure

Reserved blocks appear in this order, top to bottom. **Every
reserved block is optional;** a file without any of them is still
a valid engine input. ENGINE BODY is not a "reserved block" — it
is the actual engine content, and is always present (a file
without it would not be a script at all).

Tools that need a specific reserved block refuse cleanly when it
is missing rather than guessing.

```
1. HEADER          (reserved, optional)
2. PROVENANCE      (reserved, optional)
3. BENCH-MARKS     (reserved, optional;  .fdf and .py only)
4. ATOM-METADATA   (reserved, optional;  .fdf and .py only)
5. ENGINE BODY     (always present — the actual engine content)
6. USER-CUSTOM     (reserved, optional)
```

### Block markers

Every reserved block uses these literal lines:

```
# === molbuilder <block-name> BEGIN ===
... block content (comment-prefixed) ...
# === molbuilder <block-name> END ===
```

Parsers find blocks by scanning for these marker lines. Anything
outside any reserved block (and before USER-CUSTOM BEGIN) is engine
body.

## 4. Block specifications

### 4.1 HEADER

Human-readable run instructions. Free-form within the block; not
parsed by any tool. Format example:

```
# === molbuilder header BEGIN ===
# === Run with (job-layout v1) ===
# Run from this directory -- all outputs share the SystemLabel below.
#     mpirun -np 4 siesta < siesta-stage3.fdf > siesta-stage3.out
# Stage 3 of a staged relaxation; SIESTA reads .XV / .DM from
# the previous stage (same SystemLabel, same directory).
# === molbuilder header END ===
```

### 4.2 PROVENANCE

Static snapshot of the generator state at the moment of generation.
Always parseable as key/value pairs of the form `key value-or-list`.

```
# === molbuilder provenance BEGIN ===
#   generator-version    git e8a4f81
#   generated-at         2026-06-16T17:30:00-07:00
#   form-config-hash     sha256:7c4d...           # optional
#   resolved-defaults:
#     mpi_np            auto -> 4 (gpu+mps policy)
#     omp_threads       auto -> 2 (cores_per_socket // mpi_np)
#     BlockSize         auto -> 256 (10 * 212 atoms / mpi_np, capped pow2)
#     enable_gpu        true
#     kgrid             1x1x1 (auto-from-cell-vacuum)
# === molbuilder provenance END ===
```

**Notes:**
- `generator-version` is the molbuilder git SHA at generation time
  (short form). It identifies WHICH version of molbuilder produced
  this file. `git log <sha>` in the molbuilder repo recovers the
  full state.
- `generated-at` is ISO-8601 with timezone.
- `resolved-defaults` shows fields the user left "auto" in the form
  alongside what the auto-policy resolved them to and why. Fields
  the user set explicitly are not listed here (they live in the
  engine body where the engine reads them).
- For `.run.sh`: `resolved-defaults` carries form-state at generation
  time only. Runtime-resolved values (the actual `mpi_np` chosen
  after hardware probe + env overrides) belong to the runtime
  banner, not the provenance block.

### 4.3 BENCH-MARKS (`.fdf` and `.py` only)

Machine-readable surface that bench tooling reads to discover which
fields in the engine body are safe to override, within what limits.

```
# === molbuilder bench-marks BEGIN ===
#   version v1
#   n_atoms             212
#   n_orbitals_est      2700      # 10 * n_atoms, rough DZP heuristic
#   gpu_mode            true
#   numa_pin            socket-0
#
#   field BlockSize        anchor=BlockSize        type=pow2  range=[16,256]  default=256
#   field MaxSCFIterations anchor=MaxSCFIterations type=int   default=500
#   field MD.NumCGsteps    anchor=MD.NumCGsteps    type=int   default=200
#   field MeshCutoff       anchor=MeshCutoff       type=float unit=Ry  default=400.0
# === molbuilder bench-marks END ===
```

**Parser rules:**
- `version v1` is the block format version. Higher versions may
  add fields; older parsers refuse with a clear error rather than
  guessing.
- Top-level keys (`n_atoms`, `gpu_mode`, etc.) are informational.
- `field <name> ...` lines declare overridable parameters. Tools
  override only `field`-declared parameters; everything else stays
  as the generator emitted it.
- `anchor=<text>` is the literal token a parser greps for at the
  start of a code line (after leading whitespace) in ENGINE BODY
  to locate the override site. Anchor-based, not line-number-
  based, so the reference survives any layout drift in the
  reserved blocks above. For SIESTA `.fdf` the anchor is the
  keyword name (e.g. `BlockSize`); for PySCF `.py` it is the
  Python identifier (e.g. `max_memory_mb`).
- `type=` ∈ `{int, float, str, pow2}`. `pow2` means "power of 2".
- `range=[a,b]` and `unit=...` are advisory bounds the tool uses
  to validate user-requested overrides.

### 4.4 ATOM-METADATA (`.fdf` and `.py` only)

Embeds the label/region metadata that `.molstruct.json` carries
next to `.xyz` files, so a generated script that gets copied to an
execution directory does not strand it. The block's JSON payload
follows the **`.molstruct.json` schema** — the canonical definition
lives in `molbuilder/parsers/molstruct_json.py` (current schema is
v3). This document does NOT duplicate the schema; it cites it.

```
# === molbuilder atom-metadata BEGIN ===
# format: molstruct-json/v3
# {
#   "schema_version": 3,
#   "n_atoms_total":  212,
#   "regions": {
#     "L-electrode": [11, 12, 13, ...],
#     "R-electrode": [200, 201, ...],
#     "bridge":      [60, 61, ...]
#   },
#   "frozen_atoms": [88, 89, ..., 211],
#   "selection_rules": { ... },        # optional
#   "created_by":    "molbuilder modify",
#   "created_at":    "2026-05-20T14:23:00Z"
# }
# === molbuilder atom-metadata END ===
```

**Conventions:**
- The JSON body uses **0-based** atom indices throughout, matching
  the `.molstruct.json` schema and `Structure.regions` /
  `Structure.frozen_atoms` in Python. Note: SIESTA's
  `Geometry.Constraints` block in the engine body is 1-based by
  SIESTA convention. The two index conventions coexist in one
  file deliberately — atom-metadata round-trips with the Python
  model and `.molstruct.json`; engine body matches what the engine
  reads. Tools must not assume the same indexing.
- The `structure_hash` field from `.molstruct.json` is **NOT
  emitted** in the in-body block. Rationale: the metadata and the
  coordinates are written by the same generator pass, so they
  cannot drift apart by construction; a hash would be tautological.
- **Emission rule:** the generator emits this block ONLY when
  `regions` or `frozen_atoms` is non-empty. A file with no label
  metadata at generation time has no atom-metadata block at all
  (not a block with empty arrays).  Rationale: an empty in-body
  block would suppress a later `.molstruct.json` sidecar via the
  in-body-wins rule below, even though the user had no labels
  when they generated and only added them after.  Absence is the
  honest signal that this generation had no labels.
- `regions` and `frozen_atoms` may individually be empty when the
  block is present (e.g., the user assigned regions but no frozen
  atoms). At least one must be non-empty for the block to exist.
- When present, downstream code (TranSIESTA generator, re-render,
  etc.) reads from this block and ignores any `.molstruct.json`
  sidecar that may also be present in the directory (in-body wins;
  sidecar is the fallback for plain `.xyz` loads and for `.fdf` /
  `.py` files generated before this contract existed).
- The **load-side contract** that materialises the in-body block
  back into a `Structure` (so the next workflow stage can consume
  it) lives in [`bundle-contract.md`](bundle-contract.md) — see its
  §§ 4–5 for the run-bundle assembler and § 7 for the
  projects-sidebar storage integration.

### 4.5 ENGINE BODY

Everything between the last reserved-block END and the USER-CUSTOM
BEGIN. The generator owns this region; users should not edit it
(edits are lost on regenerate). It is not bracketed because it is
the bulk of the file.

The `anchor=<text>` references in BENCH-MARKS point INTO this
region. Parsers locate an override site by greping the engine body
for `^\s*<anchor>\b`. Anchors are stable; line numbers are not.

### 4.6 USER-CUSTOM

User-owned territory. molbuilder reads it during regeneration so
it knows where the user-custom zone lives, then preserves the
content byte-for-byte in the new output.

```
# === molbuilder user-custom BEGIN ===
# Your own additions go here.  molbuilder will preserve this section
# verbatim across regenerations.
# === molbuilder user-custom END ===
```

**Promises:**
- Regenerating with the same form-config gives byte-identical
  reserved sections AND a byte-identical user-custom block (modulo
  generator-version / generated-at fields in PROVENANCE).
- The user-custom block may be missing entirely; on regenerate the
  generator emits an empty one.
- The contents of user-custom are not validated by molbuilder. If
  the user writes engine-invalid content there, the engine will
  reject it; molbuilder will not.

## 5. Versioning

Structured blocks carry their own version tag (`version v1` in
BENCH-MARKS, `format: molstruct-json/v3` in ATOM-METADATA).
PROVENANCE has structured key/value content but no version tag —
its keys are additive and forward-compatible (new keys may be
added; old parsers ignore unknown keys). HEADER is genuinely
free-form prose and not parsed.

**Rules:**
- Generator emits the current version of each block.
- Parsers read the version tag and either handle that version or
  refuse with a clear error pointing at "regenerate this file with
  the current molbuilder".
- No autodetection of older formats. No silent upgrade. Older files
  must be regenerated.
- Block versions evolve independently: bumping BENCH-MARKS to v2
  does not affect ATOM-METADATA.

## 6. What tools can assume

Given a file conforming to this contract:

- Provenance answers "which molbuilder produced this and with what
  defaults".
- Bench-marks list which engine-body fields a tool may override and
  within what bounds.
- Atom-metadata is round-trippable: the dict can be passed straight
  to `apply_to_structure` (the same entry point `.molstruct.json`
  uses), and writing the file back produces an identical block.
- User-custom is preserved across regeneration.
- `head -50 file.fdf` shows provenance + bench-marks without
  scrolling.

## 7. What this contract does NOT cover

- **Form-config sidecar persistence.** This contract preserves
  user-custom content and label metadata, but does NOT persist the
  full form state that produced a `.fdf` / `.py`. Full
  re-rendering from scratch (versus regenerating with the user's
  current form) is a separate, larger piece of work.
- **PySCF-specific bench fields** beyond what's in `.fdf`. When
  PySCF bench lands, this doc will list the relevant `field`
  declarations for `.py`.
- **Wrapper bench-marks.** `.run.sh` does not carry a bench-marks
  block; only `.fdf` / `.py` do. Wrapper-side parameters (np, omp,
  MPS) are overridden via existing wrapper env vars, not via
  in-file marks.
- **Auto-upgrading older files** to this contract. Pre-contract
  files are valid engine inputs; they will not gain reserved
  blocks without explicit regeneration by the user.

## 8. Pinned references

- `.molstruct.json` schema (v3): `molbuilder/parsers/molstruct_json.py`
- `Structure.regions` / `Structure.frozen_atoms`: `molbuilder/structure.py`
- Sidecar load/apply: `molbuilder/web/blueprints/_shared.py::apply_sidecar_if_possible`
- Generator entry points: `molbuilder/siesta/input.py::render_fdf`,
  `molbuilder/pyscf/input.py::render_script`,
  `molbuilder/runwrap.py::render_run_wrapper`
- Bundle / load-side contract: [`bundle-contract.md`](bundle-contract.md);
  extract primitives `molbuilder/script_contract.py::extract_script_source`
  + assembler `molbuilder/script_bundle.py::assemble_from_run_dir`.
