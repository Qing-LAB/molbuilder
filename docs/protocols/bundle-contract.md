# Run-Bundle Contract

**Status:** v1, 2026-06-17
**Scope:** the workflow-handoff object that lets one finished run
(SIESTA or PySCF) flow into the next stage (Transport, continuation
optimization, spectrum, etc.) without the user re-supplying the
labels and annotations they already configured upstream.

> **This document is the sole source of truth for the run-bundle
> handoff.**  Code that assembles, materializes, or consumes a
> bundle MUST satisfy this contract.  Pointer in `design.md` § 0
> (Protocols).

## 1. Purpose

molbuilder generates engine input scripts (`.fdf`, `.py`) into a
project directory; the run produces outputs (`.XV` for SIESTA, an
optimized-geometry file or log for PySCF).  Historically the
originating `.xyz` + `.molstruct.json` are NOT copied into the run
dir.  Consequently, when the user wants to:

- continue with Transport (needs L-/R-electrode + bridge regions),
- restart from converged coords with adjusted parameters,
- spectrum-calculate at the optimized geometry,
- assemble a junction from a previously-relaxed fragment,

…the next stage has no clean source for the label metadata that
defined the run.  The user must dig back to the original design
directory — if it still exists, if they remember the path, if the
labels still match the atom ordering used by the run.

The run-bundle is the portable object that closes this gap.  It
fuses:

1. the **final structure** (coords + elements) read from the
   converged engine output, with
2. the **labels** (regions, frozen atoms) extracted from the
   originating `.fdf` / `.py` in-body `ATOM-METADATA` block, plus
3. **user-custom annotations** preserved verbatim from the script,
4. **provenance** from the script (which molbuilder produced it,
   when, with what defaults).

A bundle materialized to a destination directory writes a `.xyz` +
`.molstruct.json` pair the next tab's existing load path already
understands.  No new load primitive in downstream tabs.

## 2. Scope

**In scope:**
- Reading `.fdf` / `.py` for ATOM-METADATA, USER-CUSTOM, PROVENANCE.
- Reading final structure from a run dir's converged-coords source.
- Assembling a typed bundle object in memory.
- Materializing the bundle as `.xyz` + `.molstruct.json` at a target.

**Out of scope:**
- Execution state (SCF iterations, walltime, NSCF logs, error tails).
- Config replay: bundles do NOT carry the SIESTA/PySCF `Config`
  that produced the run.  Downstream tabs build their own config
  via their own form; the bundle supplies only structure + labels.
- Pseudopotentials, basis-set files, KMESH overrides.
- Multi-fragment composition (electrode + scattering region merge):
  a higher-level operation that consumes multiple bundles.

## 3. Data model

```python
@dataclass(frozen=True)
class RunBundle:
    structure:         Structure                       # final coords + elements
    regions:           Dict[str, List[int]]            # may be {}
    frozen_atoms:      List[int]                       # may be []
    user_custom_lines: List[str]                       # may be []
    provenance:        Dict[str, str]                  # k/v from PROVENANCE
    source_script:     Path                            # which .fdf / .py
    source_engine:     Literal["siesta", "pyscf"]
    final_coords_from: Literal["xv", "fdf-initial",
                               "py-opt", "py-initial"]
    notes:             List[str]                       # diagnostics
```

**Field semantics:**

- `structure.n_atoms` MUST equal the source script's
  `atom-metadata.n_atoms_total`.  Mismatch is an error, not a
  silent reconciliation (Section 8).
- `regions` and `frozen_atoms` indices are 0-based, matching
  `molstruct_json` schema v3 and `Structure.regions`.
- `user_custom_lines` carries the inner lines of the USER-CUSTOM
  block verbatim (no comment prefix stripping).
- `provenance` keys are whatever the originating PROVENANCE block
  declared.  Forward-compatible: unknown keys flow through.
- `source_script` is the absolute path that fed extraction.
- `final_coords_from` is load-bearing: tools and audit logs need
  to know whether the bundle reflects a converged optimization
  (`"xv"`, `"py-opt"`) or fell back to initial coords because the
  optimization output was missing (`"fdf-initial"`, `"py-initial"`).
- `notes` carries non-fatal diagnostics: schema-version mismatch,
  fallback-to-initial-coords reason, missing-PROVENANCE.  Never
  `None`; the field is always a (possibly empty) list.

## 4. Source priority

### 4.1 Final coordinates

Per engine, in priority order — first hit wins:

**SIESTA:**

| Source | Mark | When chosen |
|---|---|---|
| `<stem>.XV`              | `"xv"`           | converged geometry-opt or any run that wrote `.XV` |
| `.fdf` initial coords    | `"fdf-initial"`  | `.XV` missing — bundle still emits, but `notes` records the fallback |

> **Deferred.** A `<stem>.out` stdout-parsed final-coords source is a
> possible future addition (mark would be `"stdout"`) for runs that
> died before writing `.XV` but printed final coords to stdout.  Not
> in PR-B; the `.XV` write is robust enough that this is a rare edge
> case.

**PySCF:**

| Source | Mark | When chosen |
|---|---|---|
| `<JOB>_optimized.xyz`     | `"py-opt"`        | molbuilder-pyscf optimizer writes this on geom-opt success.  `JOB` literal extracted from `JOB = "..."` line of the `.py`.  When `JOB` extraction fails AND exactly one `*_optimized.xyz` exists, the glob match is used. |
| `.py` `mol = gto.M(atom = '''…''')`| `"py-initial"`    | `<JOB>_optimized.xyz` missing — bundle still emits, but `notes` records the fallback.  Only the molbuilder generator's whitespace-delimited atom-block format is recognised; hand-written PySCF scripts using list-of-tuple format must be re-rendered through molbuilder first. |

> **Deferred.** A pyscf-log stdout-parse final-coords source is a
> possible future addition for runs that died after geom-opt
> convergence but before the `_optimized.xyz` write.  Not in PR-C;
> the `_save_xyz` call in the generated script is reliable enough
> that this is a rare edge case.

### 4.2 Labels

In-body ATOM-METADATA in the source script is the authoritative
label source.  Where bundle assembly is initiated from a `.xyz`
load path with a sibling generated script AND a `.molstruct.json`
sidecar, **in-body wins over sidecar** (Section 6).

### 4.3 Conflict policy

- If multiple `.fdf` or `.py` files exist in the run dir, the
  bundle picks the largest-by-atom-count.  On tie, lexicographic
  by basename.  A `note` records which was chosen and why.
- If the source script's `atom-metadata.n_atoms_total` does NOT
  equal the final-structure's atom count, raise `BundleError`.
  No reconciliation.
- If both `.fdf` and `.py` are present, raise `BundleError`: a
  single run dir SHOULD NOT contain both engines' input.  The
  user MUST clean up or split before bundling.

## 5. APIs

Two-layer module split:

- **`molbuilder.script_contract`** (existing) — file-format
  extractors.  Pure, no I/O on bundle assembly path beyond the text
  passed in.
- **`molbuilder.script_bundle`** (new) — workflow assembly.  Reads
  the run dir, fuses primitives, returns a `RunBundle`.

### 5.1 `script_contract` extract primitives

```python
@dataclass(frozen=True)
class ScriptSource:
    """Everything extractable from one .fdf/.py text body."""
    regions:           Optional[Dict[str, List[int]]]
    frozen_atoms:      Optional[List[int]]
    user_custom_lines: Optional[List[str]]
    provenance:        Optional[Dict[str, str]]
    schema_version:    Optional[int]
    notes:             List[str]


def extract_script_source(text: str) -> ScriptSource: ...
def extract_provenance_dict(text: str) -> Optional[Dict[str, str]]: ...
# Plus the already-existing:
#   extract_atom_metadata_dict, apply_inbody_atom_metadata,
#   extract_user_custom_inner
```

`None` on a `ScriptSource` field = "block absent or unparseable".
Empty `[]` / `{}` = "block present, deliberately empty".  Distinct
states, distinct downstream handling.

### 5.2 `script_bundle` assembly

```python
class BundleError(Exception): ...


def assemble_from_run_dir(run_dir: Path) -> RunBundle:
    """Walk run_dir, pick engine + final-coords source per Section
    4, fuse with ScriptSource.  Raises BundleError on irrecoverable
    state (no script, atom-count mismatch, both-engines-present)."""


def write_bundle_as_handoff(bundle: RunBundle, target_dir: Path,
                            *, stem: str,
                            overwrite: bool = False
                            ) -> Tuple[Path, Path]:
    """Materialize as <target>/<stem>.xyz + <target>/<stem>.molstruct.json.
    Atomic via molstruct_json.save.  overwrite=False raises on
    existing destination files."""
```

### 5.3 `_shared.apply_companion_labels_if_present`

```python
def apply_companion_labels_if_present(struct, structure_path
                                      ) -> Optional[str]:
    """Same-stem .fdf or .py next to .xyz/.pdb.  Apply ATOM-METADATA
    if the companion carries it.  Returns "applied:fdf" / "applied:py"
    / None."""
```

`apply_sidecar_if_possible` calls it first; falls through to the
`.molstruct.json` sidecar branch only on `None`.  Codifies the
"in-body wins" rule end-to-end without needing the full bundle for
the .xyz load case.

## 6. Wire-in points

| Consumer | Wired? | Slice |
|---|---|---|
| `.xyz` load picks up sibling `.fdf`/`.py` labels (companion-wins-over-sidecar) | yes | PR-A |
| Results panel "Bundle for next stage →" button | no | PR-E |
| Transport tab consumes the bundle for electrode/buffer assignment | no | #487 |
| Continuation Optimization tab consumes the bundle as starting structure | no | future |

PR-A defines the contract surface; PR-B/C/D implement the
extractors; PR-E adds the UI button.

## 7. Versioning

- `ScriptSource.schema_version` reflects the ATOM-METADATA block's
  declared version.  `extract_script_source` accepts only the
  current generator schema (v3) without coercion; forward-compatible
  files (future v4 with additive keys) load with a `notes` warning;
  backward-incompatible (v2 with renamed keys) raise `BundleError`.
- `RunBundle` itself is not versioned: it is an in-memory dataclass
  reconstructed from disk on every call.  Persisted form is the
  emitted `.xyz` + `.molstruct.json` pair, both of which carry
  their own schema versions.

## 8. Error model

| State | Outcome |
|---|---|
| No `.fdf` and no `.py` in `run_dir` | `BundleError("no engine script in <dir>")` |
| Both engines present | `BundleError("dir contains both .fdf and .py; ambiguous")` |
| Source script has ATOM-METADATA with `n_atoms_total != structure.n_atoms` | `BundleError("atom-count mismatch ...")` |
| Source script has no ATOM-METADATA | bundle assembles with empty `regions`+`frozen_atoms`; `notes` records "no atom-metadata block in source script — bundle carries unlabeled structure" |
| ATOM-METADATA `schema_version > 3` and only-additive keys | bundle assembles; `notes` records "atom-metadata schema_version <N>; molbuilder expects 3" |
| ATOM-METADATA `schema_version < 3` | `BundleError("atom-metadata schema_version <N> is older than v3; re-render the source script with current molbuilder")` |
| Final-coords source missing all branches | bundle assembles from `.fdf`/`.py` initial coords; `notes` records the fallback |
| `write_bundle_as_handoff` target exists, `overwrite=False` | `BundleError("target exists")` |

## 9. Tests

Test pyramid placement:

- **L1** (`test_script_contract.py`):  `extract_script_source`
  round-trip; empty / partial blocks; version mismatch; provenance
  k/v parse.
- **L1** (`test_script_bundle.py`):  `RunBundle` field validation;
  `BundleError` paths that don't require I/O.
- **L2** (`test_script_bundle.py`):  `assemble_from_run_dir` against
  canned run-dir fixtures (SIESTA + PySCF, with and without `.XV`
  / `.opt.xyz`).
- **L3** (`test_script_bundle.py`):  full round-trip — emit script
  → assemble → write_bundle_as_handoff → re-load via existing .xyz
  load path → labels recovered.
- **L2** (`test_web.py`):  companion `.fdf` next to `.xyz` wins
  over `.molstruct.json` sidecar in the build endpoint flow.

## 10. Out of scope (forward references)

- **PySCF optimized-geometry parser.**  PR-C lands the reader; this
  contract pins where it plugs in but not how it parses.
- **Config replay.**  Bundles do not carry SIESTA/PySCF `Config`.
  A separate config-handoff doc would govern that if we add it.
- **Multi-fragment composition** (electrode-pair + center).  Higher
  level than a single bundle; not in this contract.

## 11. Pinned references

- Script format: [`script-contract.md`](script-contract.md) (§ 4.4 atom-metadata, § 4.6 user-custom).
- Sidecar format: `molbuilder/parsers/molstruct_json.py` (schema v3).
- Sidecar contract: [`sidecar-contract.md`](sidecar-contract.md).
- `Structure` model: `molbuilder/structure.py`.
- SIESTA `.XV` reader: `molbuilder/parsers/siesta.py` (TBD — added in PR-B).
- Sidecar-apply current entry: `molbuilder/web/blueprints/_shared.py::apply_sidecar_if_possible`.

## 12. Process

Updates to this contract require:
1. Code change matched to the doc change in the same commit.
2. Test pinning the new invariant.
3. Pointer update in `design.md` § 0 if the bundle's wire-in
   surface changes.
