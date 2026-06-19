# Job decoder protocol

**Status:** v1 draft (2026-06-19) — SIESTA-optimization only.
Spectrum and transport extensions follow the same envelope with
engine-specific plot keys.

**Modules** (to be added):
[`molbuilder/jobs/decoder.py`](../../molbuilder/jobs/decoder.py),
[`molbuilder/jobs/monitor.py`](../../molbuilder/jobs/monitor.py)
&nbsp;·&nbsp; **Tests** (to be added):
[`tests/jobs/test_decoder.py`](../../tests/jobs/test_decoder.py),
[`tests/jobs/test_decoder_multistage.py`](../../tests/jobs/test_decoder_multistage.py)

## 1. Position vs the existing parser stack

This protocol defines a **directory-level decoder** that sits on top
of the existing per-file infrastructure. It introduces NO new parsing
code for engine output — it composes:

* [`docs/types/parsers.md`](../types/parsers.md) — file-level
  `TrajectoryParser` registry, `Trajectory`/`Frame` dataclasses,
  `trajectory_to_legacy_dict`. The decoder calls `detect_parser` +
  `.parse` on each `.out` file in the run dir; never opens a SIESTA
  output file directly.
* [`script-contract.md`](script-contract.md) — the 6 reserved blocks
  in `.fdf` and `.py` (HEADER / PROVENANCE / BENCH-MARKS /
  ATOM-METADATA / ENGINE BODY / USER-CUSTOM).  The decoder reads via
  `molbuilder.script_contract.extract_script_source(fdf_text)` —
  it does NOT re-grep for atom-metadata / user-custom / provenance
  text itself.
* [`bundle-contract.md`](bundle-contract.md) — the
  `assemble_from_run_dir` API that picks the source `.fdf` (largest
  by atom-count if multi-stage), reads `.XV` / `.STRUCT_OUT` for
  final coords, and validates atom-count consistency.

What the decoder adds that NONE of these provide:

| Capability | Source |
|---|---|
| **Multi-stage consolidation** — one document per project dir, with stage1/2/3 `.out` traces stitched into per-source plot buckets | new |
| **`job_type` classification** — labels the document as `optimization` / `spectrum` / `transport` so downstream consumers pick the right plots/fields | new |
| **Status state machine** — `running` / `finished` / `failed` / `stale`, detected from `.out` tail markers + filesystem mtimes | new |
| **Progress + ETA** — `current_cg_step` / `target_cg_steps` + an estimate based on observed per-step wall time | new |
| **Engine-input snapshot per stage** — captures the full configuration the engine actually saw (HEADER / PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM verbatim + a curated `engine_body_summary`) so Results can show the user what their hand-edits resolved to | new (composes script_contract) |
| **Trigger/event semantics** — when to fire webhooks (job finished, job failed, every N CG steps) | new |

Consumers (all post-Phase-1):

* `JobMonitor` (background thread) — calls `decode_run_dir(path)` on
  every tick; compares vs. cached snapshot; fires triggers on change.
* `/api/jobs/{id}/decoded` — returns the latest decoded.json for any
  watched job; instant from cache.
* `/api/jobs/decode-once?path=...` — one-shot decode for a dir not
  being watched (Results-tab consumer path).
* Webhook delivery — POSTs decoded.json + a rendered PNG to the
  configured receiver on trigger.

## 2. The decoded.json envelope

The wire format every consumer reads. Schema version is independent
of `script-contract` versions; consumers gate on
`schema_version`.

```json
{
  "schema_version": 1,
  "decoded_at":     "2026-06-19T15:42:01.234Z",
  "decoder_version": "git <sha>",

  "job_type": "optimization",
  "engine":   "siesta",
  "system_label": "siesta-BDT-Au111-TJ",
  "run_dir":   "/abs/path/to/project/dir",

  "status": {
    "state":       "running",
    "detail":      "stage 2 of 3, CG step 145/600",
    "last_change_at": "2026-06-19T15:41:58.012Z",
    "active_source": "siesta-BDT-Au111-TJ-stage2.out"
  },

  "progress": {
    "current_cg_step":          145,
    "target_cg_steps":          600,
    "current_scf_iter_global":  4823,
    "last_iter_wall_s":         43.2,
    "mean_iter_wall_s":         43.5,
    "estimated_remaining_s":    19792,
    "stages_completed":         1,
    "stages_total_known":       3
  },

  "geometry": {
    "n_atoms":     444,
    "xyz":         [["Au", 23.45, 16.78, 15.00], ...],
    "cell":        [[46.17, 0, 0], [0, 44.43, 0], [0, 0, 64.41]],
    "regions":     { "L-electrode": [...], "bridge": [...], "R-electrode": [...] },
    "frozen_atoms": [120, 121, ..., 443],
    "coords_source": "siesta-BDT-Au111-TJ.XV",
    "coords_state":  "converged"
  },

  "plots": {
    "etot_per_cg": {
      "siesta-BDT-Au111-TJ-stage1.out": [[1, -12345.67], [2, -12345.91], ...],
      "siesta-BDT-Au111-TJ-stage2.out": [[145, -12346.34], ...]
    },
    "fmax_per_cg": {
      "siesta-BDT-Au111-TJ-stage1.out": [[1, 0.42], [2, 0.31], ...],
      "siesta-BDT-Au111-TJ-stage2.out": [[145, 0.018], ...]
    },
    "scf_residual": {
      "siesta-BDT-Au111-TJ-stage1.out": [[1, 0.5e-1], [2, 1.2e-2], ...],
      "siesta-BDT-Au111-TJ-stage2.out": [[4567, 0.8e-3], ...]
    },
    "scf_etot": {
      "siesta-BDT-Au111-TJ-stage1.out": [[1, -12345.0], ...],
      "siesta-BDT-Au111-TJ-stage2.out": [[4567, -12346.2], ...]
    }
  },

  "source_files": [
    {
      "path":      "siesta-BDT-Au111-TJ-stage1.fdf",
      "kind":      "fdf",
      "stage":     1,
      "mtime":     "2026-06-19T07:52:55.123Z",
      "size_bytes": 38421
    },
    {
      "path":      "siesta-BDT-Au111-TJ-stage1.out",
      "kind":      "out",
      "stage":     1,
      "mtime":     "2026-06-19T13:14:22.301Z",
      "size_bytes": 12483920,
      "cg_step_range": [1, 144],
      "scf_iter_range_global": [1, 4566],
      "run_state": "finished"
    },
    {
      "path":      "siesta-BDT-Au111-TJ-stage2.out",
      "kind":      "out",
      "stage":     2,
      "mtime":     "2026-06-19T15:41:58.012Z",
      "size_bytes": 8421120,
      "cg_step_range": [145, 145],
      "scf_iter_range_global": [4567, 4823],
      "run_state": "in_progress"
    },
    {
      "path":      "siesta-BDT-Au111-TJ.XV",
      "kind":      "xv",
      "stage":     null,
      "mtime":     "2026-06-19T15:41:58.012Z",
      "size_bytes": 31448
    }
  ],

  "engine_input_by_stage": {
    "siesta-BDT-Au111-TJ-stage1.fdf": { ... engine_input envelope ... },
    "siesta-BDT-Au111-TJ-stage2.fdf": { ... },
    "siesta-BDT-Au111-TJ-stage3.fdf": { ... }
  },

  "parse_warnings": [
    {"source": "siesta-BDT-Au111-TJ-stage1.out", "line_no": 4521, "category": "scf-format", "error": "...", "snippet": "..."}
  ],

  "diagnostics": {
    "active_decoder_path":  "...",
    "tick_count":           42,
    "last_tick_wall_ms":    187
  }
}
```

### 2.1 The `engine_input` envelope (per-stage value)

```json
{
  "schema_version":  1,
  "engine":          "siesta",
  "source_fdf":      "siesta-BDT-Au111-TJ-stage1.fdf",

  "header":          "...verbatim free-form prose from HEADER block, or '' if absent...",

  "provenance": {
    "present":             true,
    "generator_version":  "git 96d6fd9",
    "generated_at":       "2026-06-19T07:52:55-07:00",
    "form_config_hash":   "sha256:7c4d...",
    "resolved_defaults":  { "mpi_np": 20, "omp_threads": 1, "BlockSize": 128, "enable_gpu": true }
  },

  "bench_marks": {
    "present":         true,
    "version":         "v1",
    "n_atoms":         444,
    "n_orbitals_est":  4440,
    "gpu_mode":        true,
    "fields": [
      {"name": "BlockSize",        "anchor": "BlockSize",        "type": "pow2",  "range": [16, 256], "default": 128},
      {"name": "MeshCutoff",       "anchor": "MeshCutoff",       "type": "float", "unit": "Ry",       "default": 200.0},
      {"name": "MaxSCFIterations", "anchor": "MaxSCFIterations", "type": "int",   "default": 500}
    ]
  },

  "atom_metadata": {
    "present":         true,
    "schema_version":  3,
    "n_atoms_total":   444,
    "regions":         { "L-electrode": [...], "bridge": [...], "R-electrode": [...], "BDT": [...] },
    "frozen_atoms":    [120, 121, ..., 443],
    "created_by":      "molbuilder modify",
    "created_at":      "2026-06-19T07:52:55-07:00"
  },

  "user_custom_verbatim":  "...verbatim text of the user-custom block, or '' if absent or empty...",

  "engine_body_summary": {
    "SystemLabel":          "siesta-BDT-Au111-TJ",
    "SystemName":           "siesta-BDT-Au111-TJ",
    "NumberOfAtoms":        "444",
    "NumberOfSpecies":      "4",

    "MeshCutoff":           "200.0 Ry",
    "PAO.BasisSize":        "DZP",
    "PAO.EnergyShift":      "0.02 Ry",
    "DM.Tolerance":         "1e-03",
    "DM.MixingWeight":      "0.02",
    "DM.NumberPulay":       "3",
    "MaxSCFIterations":     "500",
    "ElectronicTemperature": "300.0 K",

    "XC.functional":        "GGA",
    "XC.authors":           "PBE",
    "SpinPolarized":        null,

    "SolutionMethod":       "diagon",
    "BlockSize":            "64",
    "Diag.Algorithm":       "ELPA-2STAGE",
    "Diag.ELPA.GPU":        ".false.",
    "Diag.ParallelOverK":   ".false.",

    "MD.TypeOfRun":          "CG",
    "MD.NumCGsteps":         "600",
    "MD.MaxForceTol":        "0.04 eV/Ang",
    "MD.MaxCGDispl":         "0.2 Ang",

    "kgrid_Monkhorst_Pack":  "1x1x1"
  }
}
```

#### 2.1.1 `engine_body_summary` curated key list

The decoder extracts a **fixed** set of SIESTA directives. The set is
codified here; adding a key requires a doc update + a test. Keys
absent from the .fdf are emitted as `null` (not omitted), so the
shape is stable across runs:

| Group | Keys |
|---|---|
| **System** | `SystemLabel`, `SystemName`, `NumberOfAtoms`, `NumberOfSpecies` |
| **SCF** | `MeshCutoff`, `PAO.BasisSize`, `PAO.EnergyShift`, `DM.Tolerance`, `DM.MixingWeight`, `DM.NumberPulay`, `MaxSCFIterations`, `ElectronicTemperature` |
| **XC** | `XC.functional`, `XC.authors`, `SpinPolarized` |
| **Solver** | `SolutionMethod`, `BlockSize`, `Diag.Algorithm`, `Diag.ELPA.GPU`, `Diag.ParallelOverK` |
| **MD / Relax** | `MD.TypeOfRun`, `MD.NumCGsteps`, `MD.MaxForceTol`, `MD.MaxCGDispl` |
| **k-mesh** | `kgrid_Monkhorst_Pack` (extracted as a compact `"NxNxN"` string from the `%block kgrid_Monkhorst_Pack` block) |

**Rule:** values are emitted as RAW strings, including units
(`"200.0 Ry"`, `".false."`, `"1e-03"`). The decoder does NOT
interpret, normalise, or convert values. Type-aware reading is the
engine's job; rendering is the Results-tab's job. This keeps the
decoder dumb and the Results-side render-friendly.

## 3. Multi-source consolidation

One decoded.json per project directory, regardless of how many
`.out` files the dir contains. Per-source plot buckets keep the
stage boundary visible:

```json
"plots": {
  "etot_per_cg": {
    "siesta-BDT-Au111-TJ-stage1.out": [[1, -12345.67], ...],
    "siesta-BDT-Au111-TJ-stage2.out": [[145, -12346.34], ...],
    "siesta-BDT-Au111-TJ-stage3.out": [[290, -12347.02], ...]
  }
}
```

### 3.1 Stage identification

* If `.out` filename matches the pattern `*-stage<N>.out` for integer
  `N`, that's the stage number.
* Otherwise `stage` is `null`; the `.out` is treated as an
  unstagged single-source run. Multi-stage consolidation collapses
  to a single bucket keyed by the `.out` filename.

### 3.2 Plot data invariants

* **Per-source data is append-only.** Stage 1's bucket never gains
  new points after stage 2 starts. A continuation that re-extends
  stage 2 (rare; e.g., a crashed run that warm-started from .DM)
  appends to the same bucket.
* **CG-step numbers are global.** Stage 2 starts at the CG step after
  stage 1's last (e.g. stage1 = 1..144, stage2 = 145..289). This
  matches what users see in `siesta-BDT-Au111-TJ.MDE`.
* **SCF iter numbers are global.** Same rule; computed from cumulative
  per-stage counts.

### 3.3 Engine-input is per-stage

`engine_input_by_stage` is keyed by `.fdf` filename. A multi-stage
project that tightens `DM.Tolerance` from `1e-03` (stage 1) to
`1e-05` (stage 3) shows three snapshots, each with the tolerance
value the engine actually saw at that stage. Results can show diffs
("MeshCutoff 200 → 350 between stage 1 and 2").

## 4. `job_type` classification

Two-step decision, with explicit precedence:

### Step 1 — script-contract preferred

If the active `.fdf` carries a `BENCH-MARKS` block with a
`job_type` declaration (forward-compatible field, may not exist in
current files), use it verbatim.

If the `.fdf` has any of the script-contract blocks but no explicit
`job_type`, infer from script-contract content:

* `BENCH-MARKS` field list contains `MD.NumCGsteps` →  `optimization`
* No `MD.NumCGsteps` but `%block ProjectedDensityOfStates` in engine body → `spectrum`
* `%block TS.Elec.<name>` in engine body → `transport`

### Step 2 — sniff fallback (no script-contract present)

For `.fdf` files generated before 2026-06-16 (no script-contract
blocks at all):

* Engine body contains `MD.TypeOfRun` (any value) and
  `MD.NumCGsteps` > 0 → `optimization`
* `%block ProjectedDensityOfStates` → `spectrum`
* `%block TS.Elec.<anything>` → `transport`
* Multiple matches → ambiguous; raise `JobTypeAmbiguousError` with
  the list of matches. This mirrors the
  `assemble_from_run_dir` "both .fdf and .py" precedent.

### Step 3 — `.py` (PySCF) classification

Same logic, with Python-side anchors:
* `tools.geomopt.optimize` or `as_pyscf_method` calls → `optimization`
* TDDFT / IR / Raman objects → `spectrum`
* No TranSIESTA-equivalent today; `transport` not supported via PySCF.

## 5. Status state machine

```
        ┌──────────────────────────────┐
        │                              │
        ▼                              │
   ┌─────────┐ tick:write             │
   │ running │────────────────────────┤
   └────┬────┘                         │
        │                              │
   tick: stale write (>60s no growth)  │
        │                              │
        ▼                              │
   ┌────────┐  tick: file grew         │
   │  stale │──────────────────────────┘
   └────┬───┘
        │
   tick: end-marker
        │
        ▼
   ┌──────────┐       tick: error-marker
   │ finished │           or rc != 0
   └──────────┘    ┌──────────┐
                   │  failed  │
                   └──────────┘
```

State definitions:

| State | Detection |
|---|---|
| **running** | The most recent `.out` has grown in size since the last tick, OR was last-modified within the past 30s. |
| **stale** | The most recent `.out` has not grown for >60s but does not yet carry the end-of-run marker. The job may have stalled (hung SCF), been killed, or just be in a long propagator step. Tick continues; status flips back to `running` if growth resumes. |
| **finished** | The most recent `.out` contains the SIESTA end-of-run marker `Job completed` AND a `Total CPU time` block. Tick cadence drops to manual-only after 5 minutes in this state. |
| **failed** | The most recent `.out` contains any of the failure markers in the curated list (`siesta: ERROR`, `propor: ERROR`, MPI abort, etc.). `last_iter_wall_s` may be `null`. |

The end-of-run and failure marker lists are codified in
`molbuilder/jobs/decoder.py::END_MARKERS` and `FAILURE_MARKERS`.
Adding markers requires a doc update + test.

## 6. Progress + ETA

| Field | Computation |
|---|---|
| `current_cg_step` | Max CG step across all per-stage plot buckets. |
| `target_cg_steps` | `engine_body_summary["MD.NumCGsteps"]` of the active stage's .fdf. Null if unparseable. |
| `current_scf_iter_global` | Sum of per-stage SCF iter counts. |
| `last_iter_wall_s` | Time delta between the last two SCF lines in the active `.out`. |
| `mean_iter_wall_s` | Mean over the last 50 SCF iter wall times in the active `.out` (rolling window; falls back to all-available if fewer). |
| `estimated_remaining_s` | `mean_iter_wall_s * (target_cg_steps - current_cg_step) * <expected_scf_per_cg>`. `<expected_scf_per_cg>` defaults to 30 (typical SCF-converges-per-CG-step for Au junctions); may become a per-job override in v2. |
| `stages_completed` | Count of `.out` files with `run_state == "finished"`. |
| `stages_total_known` | Number of distinct `*-stage<N>.fdf` files in the dir. Null if no stage suffixes. |

`estimated_remaining_s` is best-effort; the doc explicitly states "the
ETA is not load-bearing for any downstream consumer; Results renders
it as advisory text only." This keeps us honest about the limits of
the estimate.

## 7. Trigger / event model

Per `job_type`. Events fire when the decoder's tick observes the
trigger condition AND the condition has not fired before for this
state value (idempotent).

### 7.1 SIESTA-optimization (v1)

| Event | Condition |
|---|---|
| `job_finished` | `status.state` transitions to `finished`. |
| `job_failed` | `status.state` transitions to `failed`. |
| `cg_step_milestone` | `progress.current_cg_step % 50 == 0` AND not previously fired for this step value. (Threshold configurable per-job; default 50.) |
| `stage_advanced` | A new stage's `.out` appears in `source_files`. |

**Not** triggered automatically:
- Per-SCF-iter (too frequent; user explicitly opt-in if needed).
- Per-CG-step (too frequent for typical 500-step relaxation; the 50-step grouping matches the user's stated cadence).

### 7.2 Event payload

The decoded.json AT THE TIME OF THE EVENT, plus:

```json
{
  "event":         "cg_step_milestone" | "stage_advanced" | "job_finished" | "job_failed",
  "event_at":      "2026-06-19T15:42:01.234Z",
  "event_value":   50,
  "decoded":       { ...full decoded.json envelope... },
  "summary_text":  "TJ-BDT-Au111: CG step 50/600, fmax = 0.18 eV/Å"
}
```

### 7.3 Trigger budget

Each event-type per job has a per-tick rate limit. If the decoder
tick somehow advances past multiple thresholds in one tick (e.g.,
came back from a long sleep), only the LATEST one fires. The skipped
ones are recorded in `webhook_log` with `status: "coalesced"`.

## 8. Webhook delivery log

Persisted per-job at `~/.molbuilder/jobs/{id}/webhook_log.jsonl`.
Append-only; latest N (default 200) visible on the Jobs tab. The
log lives outside the run dir intentionally — webhook delivery is a
JobMonitor concern, not a per-project artifact.

```json
{
  "at":          "2026-06-19T15:42:01.234Z",
  "event":       "cg_step_milestone",
  "event_value": 50,
  "status":      "delivered" | "failed" | "coalesced" | "skipped",
  "http_status": 200,
  "duration_ms": 245,
  "attempt":     1,
  "retry_scheduled_at": "2026-06-19T15:43:01.234Z",
  "error":       null,
  "url_redacted": "https://hooks.slack.com/services/T**********/B****..."
}
```

Status semantics:
* `delivered`: 2xx response, body OK.
* `failed`: 4xx (no retry), 5xx (retry until budget exhausted), or
  transport error. `error` field carries a short message.
* `coalesced`: event was skipped because a later event of the same
  type took priority within one tick.
* `skipped`: webhook is disabled for this job OR the event
  threshold was previously fired and not re-armed.

URL is redacted (path component hidden) in the log so a stolen log
doesn't expose the secret-in-the-URL pattern Slack uses.

## 9. Forbidden patterns

These are the things tools must NOT do; violations create silent
divergence between the decoder and other consumers:

1. **Never re-parse what `extract_script_source` already extracts.**
   Atom-metadata, user-custom, provenance come from there
   exclusively. The decoder's own grep is forbidden for these
   blocks. If `extract_script_source` is missing a field, fix it
   there — don't side-grep in the decoder.
2. **Never re-parse engine-output (the `.out` file) outside the
   `TrajectoryParser` registry.** Plot data, geometry, run_state,
   parse_warnings all come from `detect_parser(out).parse(out)`.
   The decoder may inspect filesystem mtimes + sizes (for status
   detection), but not file CONTENTS.
3. **Never interpret `engine_body_summary` values.** They ship as
   raw strings, units included. Results consumers may render them
   with formatting; they may NOT recompute on them (e.g., comparing
   `"200.0 Ry"` to `"350 Ry"` should be a string compare or a
   downstream-formatted compare, not a numeric one in the decoder).
4. **Never fire a webhook on every tick.** The 50-CG-step threshold
   is the default; smaller thresholds are configurable per job but
   must respect the once-per-state-value rule (Section 7).
5. **Never bypass the schema-version gate.** Consumers that see
   `schema_version` higher than they support must refuse with a
   clear error message ("decoded.json schema v<N>; this consumer
   knows v<M>"). They must NOT attempt to render unknown shapes.
6. **Never persist the webhook URL in the run dir.** It belongs to
   the JobMonitor's `~/.molbuilder/jobs/*` state, not the project.
   This keeps secrets out of the project tree so a project sync to
   shared storage does not leak them.

## 10. Test coverage list

Required tests for the v1 ship. Live under `tests/jobs/`:

| Test | Level | What it pins |
|---|---|---|
| `test_decode_run_dir_smoke` | L3 | Decode a known good fixture; assert top-level schema_version + required keys present. |
| `test_decode_multistage` | L3 | Multi-stage `BDT-withAuJunction` fixture: per-source plot buckets keyed by `.out` filename; CG-step ranges contiguous across stages. |
| `test_engine_input_envelope` | L2 | Per-stage `engine_input` carries header / provenance / bench_marks / atom_metadata / user_custom_verbatim / engine_body_summary; missing-block fields are explicit (`present: false` or `null`). |
| `test_engine_body_summary_curated_keys` | L2 | Exact key list present (with `null` for missing keys); no extra keys; raw string values. Catches accidental schema drift. |
| `test_job_type_classification_script_contract` | L2 | `TJ-BDT-Au111` fixture (has script-contract block) classifies as `optimization`. |
| `test_job_type_classification_sniff_fallback` | L2 | `BDT-withAuJunction` fixture (no script-contract block) classifies as `optimization` via sniff. |
| `test_job_type_classification_ambiguous` | L2 | A constructed `.fdf` with both `MD.NumCGsteps > 0` AND a `%block TS.Elec.*` raises `JobTypeAmbiguousError`. |
| `test_status_running_to_finished` | L3 | Status transitions correctly when `Job completed` marker appears mid-tick. |
| `test_status_failed_on_marker` | L3 | Status flips to `failed` when failure marker present. |
| `test_progress_eta_monotonic` | L2 | `estimated_remaining_s` decreases tick-over-tick when CG step grows. |
| `test_forbidden_no_direct_out_grep` | L2 / lint | grep-lint that the decoder source contains no `re.search`/`re.match`/`grep` calls against `.out` content (only filesystem-level checks). |
| `test_trigger_idempotent_per_state_value` | L2 | Firing `cg_step_milestone` at step=50 once does not re-fire at the next tick when current_cg_step is still 50. |
| `test_webhook_url_redacted_in_log` | L2 | Persisted log carries redacted URL form. |

## 11. Engine extensions (future)

The envelope's `job_type` field is the extension hook. Adding
`spectrum` and `transport` later requires:

* Defining the per-engine plot keys under `plots.*` (e.g.
  `transmission_E` for transport, `peaks_table` + `dipole_strength`
  for spectrum). The per-source bucket structure stays identical.
* Defining the engine-specific `engine_body_summary` curated key
  list (transport adds `TS.Voltage`, `TS.Elec.<name>` block summary,
  etc.; spectrum adds `%block ProjectedDensityOfStates` summary).
* Defining the engine-specific trigger thresholds.

Each engine extension lands as its own incremental section in this
doc, with its own L2 test set. The v1 shape stays stable.

## 12. Pinned references

* [`parsers.md`](../types/parsers.md) — file-level `TrajectoryParser`
  registry + `Trajectory`/`Frame` dataclasses. The decoder calls
  `detect_parser` + `.parse`; never opens engine output files
  directly.
* [`script-contract.md`](script-contract.md) — the 6 reserved
  blocks. `extract_script_source` returns
  atom-metadata + user-custom + provenance. The decoder's
  `engine_input.engine_body_summary` is a NEW layer for engine-body
  directives only.
* [`bundle-contract.md`](bundle-contract.md) — `assemble_from_run_dir`
  picks the source `.fdf` + reads `.XV` for final coords. The
  decoder reuses this for the `geometry` field.
* [`results-state-contract.md`](results-state-contract.md) — Results
  tab state shape. Phase 5 of the JobMonitor refactor switches
  Results to consume decoded.json instead of parsing on its own;
  results-state-contract.md will gain a section pointing at
  `decoded.engine_input` etc. as the new source of truth.
* [`web-api.md`](web-api.md) — the
  `/api/jobs/{id}/decoded` + `/api/jobs/decode-once` endpoints
  added in Phase 2 are documented there.

## 13. Decisions log

| Date | Decision |
|---|---|
| 2026-06-19 | Initial draft. SIESTA-optimization only. `engine_input_by_stage` per-stage (not collapsed). Forbidden patterns codified. Curated `engine_body_summary` key list pinned. |
