# Backend architecture — the four concerns (data · construction · validation · execution)

> **What this is.** A map of the Python backend organised by **functional
> concern** — how the code separates *managing data*, *constructing structures*,
> *scientifically validating* a job, and *running the workflow*. It is the
> concern-lens complement to two existing docs that use different lenses:
> [`architecture.md`](../architecture.md) indexes subsystems by **layer**
> (L1/L2/L3) and points at each authoritative doc; [`web-module-map.md`](web-module-map.md)
> maps the **front-end + web-API** modules. This file answers a question those
> two don't ask head-on: *are the four backend concerns cleanly separated, and
> where do they leak into each other?*
>
> **The short answer:** yes, the separation is real and mostly clean — it rides
> on the load-bearing L1/L2/L3 import-direction rule (enforced by
> `tests/test_layering.py`) plus the four core types as a lingua franca. Data
> management and structure construction are well-isolated. The one concern with
> genuine gaps is **scientific validation**, which is centralised only for the
> SIESTA/PySCF Build engines and scattered for Spectra/Transport. A handful of
> smaller execution↔engine couplings are catalogued in § 6.

---

## 0. Two axes: layers × concerns

The backend is organised on **two orthogonal axes**, and both matter:

- **Layer** (`design.md` Architecture, enforced): *import direction.* L1 core
  types import nothing above them; L2 domain verbs may import L1; L3 surfaces
  (cli/web) import both. This is the invariant that stops the registry/abstraction
  tangle from growing back. `tests/test_layering.py` fails any violation.
- **Concern** (this doc): *functional responsibility.* Every L2 verb belongs to
  exactly one of four concerns. The concern is not enforced by a test — it is a
  design intent this document records so new code lands in the right module.

The concern axis cuts *across* layers. "Data management" owns L1 types **and**
L2 codecs; "scientific validation" owns L1 primitives **and** the L2 validation
pass. So a module's layer tells you *who it may import*; its concern tells you
*what job it does*. Keep both straight.

```
            DATA            CONSTRUCTION       VALIDATION        EXECUTION
          management                          (science)         (workflow)
  ┌───────────────────────────────────────────────────────────────────────┐
L3│  web/blueprints/*   ·   cli.py            (surfaces: deserialize → verb → serialize)
  ├───────────────────────────────────────────────────────────────────────┤
L2│  parse/  sidecars/  │ peptide nucleic   │ validation/     │ jobset/  bench/
  │  workingcopy_       │ smiles pubchem    │ (pass +         │ runwrap  envs/
  │  structure          │ modify            │  engine         │ checkpoint
  │  script_emit        │ builders/backends │  adapters)      │ transport/orchestrate
  ├───────────────────────────────────────────────────────────────────────┤
L1│  structure  frame   │ (none —           │ chemistry       │ (none — execution
  │  selection  config  │  construction is  │ pseudos         │  is all verbs;
  │  trajectory_log     │  all L2 verbs)    │ residues        │  runtime_config is L2)
  │  persist  issues    │                   │                 │
  └───────────────────────────────────────────────────────────────────────┘
```

The four core types (`Structure`, `Frame`, `Config`, `Issue`) are the wire
between concerns: **construction** emits a `Structure`; **validation** reads a
`Structure`+`Config` and returns `List[Issue]`; **execution** runs a job whose
inputs the engine emitters rendered from `Structure`+`Config`; **data
management** owns the round-trip of all of them to and from disk.

---

## 1. Data management

**Job:** own every representation of structure/trajectory/config data and its
serialization — in memory (the dataclasses), on the wire (`to_wire`), and on
disk (the codecs). No domain logic; a data module never validates chemistry,
builds geometry, or runs a job.

| Module | L | Role | Public entry points |
|---|---|---|---|
| `structure.py` | L1 | **the ONE structure codec + key-namer** — coords + metadata | `Structure`, `to_dict`/`from_dict`/`to_wire`, `metadata_to_dict`/`apply_metadata_dict`, `from_xyz`/`to_xyz`/`from_pdb`, `resolve_cell` |
| `frame.py` | L1 | `Frame`/`Trajectory` per-step physics | `Frame`, `Trajectory` |
| `selection.py` | L1 | atom-selection rule algebra (duck-typed on `struct`) | `Rule`, `evaluate`, `to_json`/`from_json` |
| `config/` | L1 | the engine knob **dataclasses** (the lingua franca) + field metadata | `SiestaConfig`, `PySCFConfig`, `SpectraConfig`, `TransportConfig` |
| `trajectory_log/` | L1 | molwatch progress-log format + writer | `format`, `emitter` |
| `persist.py` | L1 | versioned-doc schema check + atomic JSON IO | `check_schema_major`, `read_json`, `write_json` |
| `parse/` | L2 | the **single read-side stack**: File/Text/Dir → typed `ParseResult` | `parse.registry.{parse,parse_dir,parse_text}`, `parse.dirs.job.decode_run_dir` |
| `sidecars/` | L2 | write-side `.molstruct`/`.spectra`/`.transport` JSON | `to_dict`, `save`/`load`, `apply_to_structure` |
| `workingcopy_structure.py` | L2 | the paired `.xyz`+`.molstruct.json` file door | `StructureCodec.{read,write,from_scratch}` |
| `script_emit.py` | L2 | write-side of the generated-script reserved blocks | `emit_header`/`emit_provenance`/`emit_bench_marks`/`emit_atom_metadata` |
| `projects.py` | L2 | filesystem layout / naming authority | `validate_name`, `project_dir`, `list_structures` |

**Assessment: clean.** No data module imports execution, validation, or
construction code; every dependency points *down* to the L1 `Structure`/
`selection`/`frame` authorities or *sideways* within the read/write codec pair
(`sidecars/` write ↔ `parse/sidecars/` read, resolved with lazy imports). The
structure-authority consolidation (one codec: `Structure.to_dict`/`from_dict`/
`to_wire`; sidecar `to_dict` spreads `**fields` with no hand-listing) is
effectively complete. See [`structure-authority.md`](structure-authority.md)
and [`data-vocabulary.md`](data-vocabulary.md).

Two residual data-consolidation gaps (both tracked): the vestigial
`web/blueprints/_shared.py::structure_to_dict` wrapper (gutted to call
`to_wire()` but not deleted), and the CLI load/save path not yet routed through
`StructureCodec` (**task #73**).

---

## 2. Structure construction

**Job:** synthesize or edit geometry — `sequence/SMILES/name` in, `Structure`
out; or `Structure` in, edited `Structure` out. A builder never persists to a
project file (the codec does that) and never runs the formal validation pass
(that gate is at generation time).

| Module | L | Role | Yields |
|---|---|---|---|
| `peptide.py` | L2 | peptide builder | `Structure` |
| `nucleic.py` | L2 | DNA/RNA builders (dispatch to backends) | `Structure` |
| `builders/backends/` | L2 | per-tool backends: `_rdkit` / `_amber` / `_threedna` | `Structure` (or `BackendUnavailable`) |
| `smiles.py` | L2 | SMILES → `Structure` (RDKit + OpenBabel fallback) | `Structure` |
| `pubchem.py` | L2 | name → SMILES → `Structure` | `Structure` |
| `modify.py` | L2 | atom-level edits (delete/add/orient/electrode/calibrate) | `Structure` |

**Assessment: clean, with three consistency nits.** Every builder returns
`molbuilder.Structure`; none persists to the workspace (backends only write
scratch files into a `TemporaryDirectory` to feed an external tool). Build-time
chemistry (H-addition, phosphate protonation, backbone-connectivity self-check,
duplex clash warnings) lives inline **by design** — it is structure *cleanup*,
not the scientific *gate*, and `modify.py:194` documents the boundary (it defers
zero-offset enforcement to `validate_geometry`). The nits, all minor:

1. **Duplicate PDB codec.** `builders/backends/_common.py::parse_pdb_to_structure`
   is a second PDB reader independent of the unified `parse/` stack. Small codec
   logic embedded in the construction layer.
2. **Non-uniform external-tool gating.** Four tools, four idioms: `_threedna` (a
   bespoke filesystem detection chain), `_amber` (defers to `diagnostics` +
   `envs.run_tool`), `_rdkit` (`importlib.util.find_spec`), OpenBabel-in-`smiles`
   (inline `try/import`). Only `_amber` reaches into the execution/env layer, and
   only for external-process routing — not `jobset`/`runwrap`.
3. **One return-type wart.** `smiles.build_from_smiles(..., return_backend=True)`
   returns an ad-hoc `(Structure, str)` tuple instead of a bare `Structure`.

---

## 3. Scientific validation — **the concern with real gaps**

**Job:** answer "is this job scientifically defensible before we emit it?" —
chemistry (spin/charge parity, open-shell/metal), pseudopotential coverage,
engine-keyword sanity (k-grid, cutoff, basis, boundary conditions). The intended
contract ([`science.md`](../science.md), [`scientific-validation.md`](scientific-validation.md)):
**one shared chemistry analyzer**, and **one `validate()` pass per engine** that
a surface calls once.

### What is unified (was already good, still good)

The chemistry `(charge, spin, treatment)` invariant — the place silent errors
hide — is genuinely single-sourced. `chemistry.analyze_structure` (L2, engine-
agnostic) + the shared `check_open_shell_metal` helper are consumed by
`validation/`, both engine emitters, the spectra preflight, and the transport
preflight. `pseudos.check_coverage` is the single pseudopotential owner (C1–C6).
No engine reimplements the chemistry.

### What was scattered — Tier 1 FIXED (2026-07)

The Tier-1 gaps below were the review's "do we need more separation? **yes**"
answer — not a new module, but *finishing the one that exists*. All fixed:

| # | Finding (was) | Fix shipped |
|---|---|---|
| V1 | **No single pass for Spectra/Transport** — each surface ran `validate(...) + engine.preflight(...)`; forget the second and the science silently skipped. | `SpectraConfig` + `TransportConfig` are now registered in `_ENGINE_VALIDATORS`; the blueprints call `validate(struct, cfg, prior=prior)` **once**. `validate()` threads `prior` to the engine validator. Pinned by `test_all_four_engine_configs_are_registered`. |
| V2 | **Transport science lived outside the registry** (`transport/transiesta.py::preflight`). | The registered `_validate_transport` dispatches to the engine's `preflight` via `get_engine(cfg.engine)` — region / electrode-ordering / bias checks now run through `validate()`. |
| V3 | **A third type system** — the device↔electrode cross-run validator used `Check`/`PreflightReport`, not `Issue`. | Kept as a distinct **checklist** type (it reports passing "ok" gates `Issue` can't model — a legitimately different role) but **bridged**: `PreflightReport.to_issues()` / `Check.to_issue()` map the problem-severity checks onto `Issue`, so it is no longer a *disconnected* third system. |
| V4 | **A duplicated rule** — hybrid `grid_level < 4` in both `validation/pyscf.py` and `spectra/pyscf_engine.py` with different detectors. | ONE body: `validation.pyscf.is_hybrid_functional` + `check_hybrid_grid_level(cfg, context=…)`, called by both sites; the message is context-selected (opt forces vs Hessian frequencies). |
| V5 | **Doc drift** — `chemistry-correctness.md` pointed at a non-existent `molbuilder/analyzer.py` / `analysis_notes`. | Repointed to `chemistry.py::analyze_structure` / field `warnings` / the real deterministic-analyzer test. |

**The render-gate vs preflight-UX split (V1/V2 subtlety).** Registering the
spectra preflight surfaced that it mixed two concerns: render-gate **science**
(grid / amplitude / parity / method / open-shell) and a preflight-only
**selector-availability** check (`top_n`/`threshold` want prior Raman data). Only
the science gates render — a `top_n` script is valid to *emit* (it resolves at
run time). So the spectra engine split into `render_checks` (registered as the
`SpectraConfig` validator; run by both the render path and `validate()`) and
`selector_checks` (preflight-only UX the /spectra endpoint adds on top).
`preflight()` composes both for back-compat.

---

## 4. Workflow / execution

**Job:** run a set of related jobs on any target — stage laddering, scheduling,
env dispatch, launcher emission, checkpointing, benchmarking, run-bundle handoff.
The design intent ([`staged-execution.md`](staged-execution.md),
[`jobset-infrastructure.md`](../jobset-infrastructure.md) §3c): **everything after
the *producer* is engine-agnostic** — the core never parses a `.fdf`, sees only
opaque script filenames.

| Module | L | Role | Engine knowledge? |
|---|---|---|---|
| `jobset/` (`model`/`materialize`/`prep`/`plan`/`submit`/`_cli`) | L2 | engine-agnostic staged-execution core | **none** (correct) |
| `jobset/runstatus.py` | L2 | read-only per-stage status + warm-file inventory | **leaks** (see W2) |
| `bench/` | L2 | benchmark sweep; a **JobSet producer** | siesta (sanctioned — producer) |
| `runwrap.py` | L2 | the `.run.sh`/`.sbatch` launcher emitter | **deep** (see W1) |
| `runtime_config.py` | L2 | `molbuilder.json` reader (scheduler/routing/exec) | — (see W3) |
| `diagnostics.py` | L2 | host capability snapshot + env-for-category routing | env-category tables (config, fine) |
| `envs/` | L2 | conda-env dispatch + doctor/validate/install toolkit | — (clean) |
| `checkpoint.py` | L1* | git-backed run-dir snapshot/restore + binary archiving | engine glob tables, *parameterized* (fine) |
| `monitor.py` | L2 | shipped standalone job-status tailer | siesta `.out` markers (by design — bare-env) |
| `transport/orchestrate.py` | L2 | 3-run TranSIESTA driver | fused (see W4) |

**Assessment: the core is exemplary; the edges couple to a specific engine.**
The `jobset/` core is a model of the separation — pure engine-agnostic verbs,
no data/science/construction reach-ins, all engine knowledge held by producers.
The coupling smells live at the edges:

| # | Finding | Evidence |
|---|---|---|
| W1 | **`runwrap.py` is the coupling hotspot.** Engine-aware by necessity (restart semantics), but it also reaches into SIESTA *science* (`siesta.memory.estimate_siesta_memory` for the `--mem` audit) and the `.fdf` *input schema* (`_fdf_requests_gpu/_elpa`, `_parse_fdf_n_atoms`). Execution is fused to one engine's input+memory model rather than sitting behind an interface. | `runwrap.py:2838,3154` (memory); `:1438,1473,1497` (fdf) |
| W2 | **The one leak inside the "agnostic" core.** `jobset/runstatus.py` hardcodes `siesta`→`.XV/.DM/.CG`, `pyscf`→`.chk` warm-file extensions — the only post-producer module that knows engines, contradicting §3c. Extract to a producer-supplied inventory or an engine registry. | `jobset/runstatus.py:31-34` |
| W3 | **`runtime_config` leaks scheduler schema as untyped dicts** into `jobset/submit.py` and `runwrap.py` (`d["partition"]`, `d["qos"]`, …), and the module also bundles web-auth/TLS config with scheduler config — two concerns in one reader. | `runtime_config.py:201-237,504-548`; consumers `submit.py:86`, `runwrap.py:2836` |
| W4 | **Transport bypasses the framework.** `transport/orchestrate.py` hand-rolls a bash driver and fuses orchestration with `.fdf` science + structure construction (`electrode_wizard`) in one file, instead of producing a `JobSet`. This is the documented future migration (jobset-infrastructure §7 Use-case C), blocked on `depends_on` becoming multi-parent (`List[str]`) for the device diamond. | `transport/orchestrate.py:45-54,173-244` |
| W5 | **Two modules misfiled under "workflow."** `bundle_writer.py` and `script_emit.py` are really **data** serializers (they reach into `parse.types`, `sidecars`, `structure`), not execution verbs. (`script_bundle.py` no longer exists — it was split into these two.) | `bundle_writer.py:34,116`; `script_emit.py:242,318` |

The scheduler vocabulary in `jobset/model.py::Resources` (`mpi_np`, `time`,
`mem`, `gres`, `domain`) is SLURM-shaped — a *scheduler* coupling in the neutral
core, not an *engine* one. Acceptable today (SLURM is the only real backend), but
worth naming so a future PBS/LSF backend knows where the seam is.

---

## 5. The cross-concern pipeline

The concerns compose into one flow; the arrows are the only sanctioned
inter-concern calls, and each carries a core type:

```
 CONSTRUCTION            VALIDATION              (engine emit)          EXECUTION            DATA (read-back)
 build_peptide/dna/   →  validate(struct,cfg) →  render_fdf /        →  jobset prep/submit → parse.decode_run_dir
 smiles/name/modify      → List[Issue]           render_script          runwrap/envs         → Frame / Trajectory
      │                        │                  (string)               (runs the job)          │
      └── Structure ───────────┴──────────────────────┘                                          │
                                                                                                 ▼
                         DATA owns the round-trip of every arrow to/from disk:  StructureCodec (.xyz+.json),
                         sidecars (.molstruct/.spectra/.transport), script_emit (reserved blocks), persist (@major).
```

Read it as: **construction** produces a `Structure`; **validation** gates it
against a `Config`; the engine emitter (a construction-adjacent verb) renders the
input text; **execution** runs it; **data management** parses the results back
into `Frame`/`Trajectory` and owns persistence at every step. The `Issue` list is
advisory while editing and enforcing at generation (`report()` is the only hard
gate).

---

## 6. Verdict + ranked plan

**Do we need a new logical separation?** No — the four concerns already have
distinct homes, enforced downward by the L1/L2/L3 rule and wired by the four core
types. Data management and structure construction are clean. What's needed is
**completing** the separation in two concerns, not inventing a new one:

**Tier 1 — the real separation gap (scientific validation). ✅ SHIPPED 2026-07.**
See § 3's V1–V5 table for the as-built detail.
1. ✅ `SpectraConfig` + `TransportConfig` registered in `_ENGINE_VALIDATORS`;
   `validate(struct, cfg, prior=…)` is the single per-engine gate (V1/V2).
2. ✅ The duplicated hybrid `grid_level < 4` rule is one shared body in
   `validation.pyscf` (V4).
3. ✅ `transport/preflight`'s checklist bridges to `Issue` via
   `to_issues()`/`to_issue()` — kept as a distinct type (it reports passing
   "ok" gates `Issue` can't model) but no longer disconnected (V3).
4. ✅ The `analyzer.py`/`analysis_notes` doc drift in
   `chemistry-correctness.md` is fixed (V5).

**Tier 2 — execution↔engine decoupling (workflow).**
5. Give the `.fdf`/memory reach-ins in `runwrap.py` an engine-interface seam so
   execution isn't hard-fused to SIESTA (W1).
6. Move `jobset/runstatus.py`'s warm-file table to a producer-supplied inventory
   (W2) — restores the engine-agnostic contract of the core.
7. Resolve `runtime_config` scheduler dicts behind a typed API and split the
   web-auth config out of the scheduler reader (W3).
8. Migrate `transport/orchestrate.py` onto `jobset` once `depends_on` is
   multi-parent (W4) — the documented future work.

**Tier 3 — hygiene.**
9. Re-file `bundle_writer.py` + `script_emit.py` under the data concern (W5);
   collapse the duplicate PDB codec in `builders/backends/_common.py` onto
   `parse/` (§2.1); normalise external-tool gating (§2.2); drop the `smiles`
   `return_backend` tuple (§2.3).

None of these are load-bearing bugs — the backend works and the layering holds.
They are the difference between "separated in practice" and "separated by
contract, uniformly."

---

## 7. Keeping the separation (the boundary contract)

When adding backend code, place it by concern **and** layer:

- **A new builder** returns a `Structure` and nothing else. It may clean up
  chemistry inline; it must not persist, and must not run `validate()`.
- **A new scientific check** goes in `validation/` (or `chemistry.py` if it is an
  irreducible per-structure primitive), registered so a single `validate(cfg)`
  reaches it. Never inline a new `preflight` outside the registry.
- **A new execution verb** stays engine-agnostic if it is post-producer; engine
  knowledge belongs in a *producer* (like `bench/to_jobset`). Never parse a
  `.fdf` in the jobset core.
- **A new data codec** routes through the `Structure` authority
  (`to_dict`/`from_dict`/`to_wire`) and `persist.check_schema_major`; never
  hand-list metadata fields or hand-roll a schema check.

The enforcement floor stays `tests/test_layering.py` (import direction). This
document is the concern-level intent above that floor; keep it in sync when a
concern boundary moves.
