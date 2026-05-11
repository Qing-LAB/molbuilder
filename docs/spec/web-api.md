# Spec — Web UI + Build API

**Module**: `molbuilder/web/app.py` (registers the three blueprints)
&nbsp;·&nbsp; **Templates**:
`molbuilder/web/templates/{index,modify,watch}.html`
&nbsp;·&nbsp; **Static**:
`molbuilder/web/static/{viewer,modify/viewer,watch/viewer}.js` +
`static/lib/{tabs,tokens}.css`, `static/lib/mol-{style,format,pick}.js`
&nbsp;·&nbsp; **Tests**:
`tests/test_web.py`, `tests/watch/`, `tests/test_modify_e2e.py`,
`tests/test_review_fixes.py` (S6)

The web UI is a single Flask app serving **three tabs** that share
the 3Dmol.js viewer + style helpers and tab-bar partial:

| Path | Tab | Spec | Blueprint |
|---|---|---|---|
| `/`         | Build  | this file       | `web/blueprints/build.py` |
| `/modify`   | Modify | `modify-tab.md` | `web/blueprints/modify.py` |
| `/watch`    | Watch  | `watch-ui.md`, `watch-api.md` | `web/blueprints/watch.py` |

The Modify and Watch endpoint contracts live in their own specs.
This document covers (a) shared infrastructure and (b) the
`/api/build/*` surface.

---

## Shared endpoints

| route | method | body | response | status |
|---|---|---|---|---|
| `/api/health`   | GET | — | `{ok: true, version: "x.y.z"}` | — |
| `/api/backends` | GET | — | `{ok, available: {rdkit, amber, threedna}: bool}` | — |

`/api/backends` is consumed by Build's backend picker and by Modify
for cross-check.  The `threedna` entry is `true` only when the
detection chain (in-tree → `$X3DNA` → `fiber` on PATH; see
`docs/design.md` § "3DNA") finds a complete install.

---

## `/api/build/*` — Build blueprint (mounted in `web/blueprints/build.py`)

| route | method | body | response | status |
|---|---|---|---|---|
| `/api/build/molecule`         | POST | `{kind, input, ...}` | structure JSON | 400 bad input, 500 missing dep |
| `/api/build/load`             | POST | JSON `{text, format, filename}` OR multipart `file=` | structure JSON | 400 empty, 413 too big |
| `/api/build/fdf`              | POST | `{xyz, params}` | `{ok, fdf, system_label, issues}` | 400 bad params, 500 render |
| `/api/build/pyscf`            | POST | `{xyz, params}` | `{ok, script, job_name, issues}` | 400 bad params, 500 render |
| `/api/build/preflight`        | POST | `{xyz, engine, params}` | `{ok, issues}` | 200 always (issues carry errors) |
| `/api/build/schema/<engine>`  | GET  | — (engine ∈ {`siesta`, `pyscf`}) | `{ok, schema}` | 404 unknown engine |

The structure JSON shape on success (returned by `/molecule` and
`/load`):

```json
{
  "ok":            true,
  "xyz":           "<xyz text>",
  "pdb":           "<pdb text>",
  "n_atoms":       int,
  "n_residues":    int,
  "summary":       "<Structure 'title': N atoms, R residues, formula CxHyOz>",
  "title":         str,
  "elements":      ["C", "H", ...],
  "atom_names":    [...],
  "residue_ids":   [...],
  "residue_names": [...],
  "chain_ids":     [...],
  "source_format": "xyz" | "pdb"          // /load only
}
```

The error JSON shape is identical across all endpoints:

```json
{ "ok": false, "error": "human-readable message" }
```

### `/api/build/molecule` payload

```json
{
  "kind":     "peptide" | "dna" | "rna" | "smiles" | "name",
  "input":    str,
  // DNA / RNA only:
  "backend":   "auto" | "rdkit" | "amber" | "threedna",
  "form":      "B" | "A" | "Z",
  "terminal":  "OH" | "5P" | "3P" | "PP",
  "add_hydrogens":         bool,
  "protonate_phosphates":  bool
}
```

Unknown `kind` → 400 with a list of valid values.  Empty `input` →
400.  Missing optional dep (PeptideBuilder for `peptide`, PubChemPy
for `name`, OpenBabel + tleap for `amber`, 3DNA for `threedna`) →
500 with install hint.

### `/api/build/load` payload variants

JSON body:
```json
{
  "text":     "<xyz or pdb text>",
  "format":   "xyz" | "pdb" | "auto",
  "filename": str       // helps format auto-detect, optional
}
```

Multipart form-data: `file=<uploaded file>`.

Format detection precedence: explicit `format` → filename extension
→ content sniff (digits-only first line → xyz, else pdb).

### `/api/build/fdf` and `/api/build/pyscf` payload

```json
{
  "xyz":    "<xyz text from a previous build>",
  "params": { /* SiestaConfig or PySCFConfig fields */ }
}
```

Server-side:
* Parses `xyz` via `Structure.from_xyz` (canonical parser).
* Filters `params` against `fields(SiestaConfig|PySCFConfig)` so
  unknown keys are silently dropped.
* Type-coerces each value via `_shared.coerce_to_field_type` (handles
  `Optional[X]`, kgrid tuple, sequence-of-strings, etc.).
* Per-field validators (`metadata["range"]`, `metadata["validate"]`)
  run via `validation.validate(struct, cfg)` BEFORE emission; an
  error-severity `Issue` raises `ValidationError` and the route
  returns 400 with the issues array.
* Special-case `kgrid`: incoming `[a, b, c]` list converted to
  `(int, int, int)` tuple.
* Special-case `net_charge`: empty string or null → `None`
  (auto-detect from phosphate protonation).
* Generators emit a "Run with:" verbose-comment block referencing
  the protocol basename (`SystemLabel` / `job_name`) per the
  job-layout v1 protocol; see [`job-layout.md`](job-layout.md).

### `/api/build/preflight` payload

Validation-only sibling of `/fdf` and `/pyscf`.  Same body shape
plus an `engine: "siesta" | "pyscf"` discriminator; returns just
`{ok, issues}` so the UI's issues panel can update live without
generating the file body.

### `/api/build/schema/<engine>` (form-rendering schema)

Read-only.  Returns the JSON-friendly schema produced by
`_shared.py::dataclass_to_form_schema(cls, id_prefix)` for the
SIESTA / PySCF Build panels — the Build tab's JS renderer
(`web/static/lib/form-schema.js`) consumes this directly so the
dataclass is the only place form-field declarations live.

Response shape:

```json
{
  "ok": true,
  "schema": {
    "config":    "SiestaConfig" | "PySCFConfig",
    "id_prefix": "p" | "py",
    "sections": [
      {
        "name":   "<section legend>",
        "fields": [<field_schema>, ...]
      },
      ...
    ]
  }
}
```

Per-field schema (only the keys relevant to the inferred `kind`
are populated):

```json
{
  "name":     "<dataclass field name>",
  "id":       "<id_prefix>-<id_suffix>",     // HTML id; renderer/compat-engine contract
  "label":    "<human label>",
  "help":     "<tooltip text>",
  "default":  <JSON-serialisable default>,
  "optional": bool,
  "tier":     "basic" | "advanced",
  "kind":     "checkbox" | "int" | "number" | "text"
              | "select" | "tri-select" | "int-triple",
  // number / int:
  "min":   ..., "max": ..., "step": ...,
  // select / tri-select:
  "choices":     [...],
  "null_option": true,
  "null_label":  "(default)" | "(auto)" | ...,
  // int-triple (kgrid):
  "labels":  ["x", "y", "z"],
  // display:
  "unit":    "Å" | "Ry" | "Hartree" | ...,
  "pattern": "<HTML5 pattern attr>"
}
```

Contract:

* **Opt-in**: only dataclass fields whose metadata declares a
  `"section"` key are exposed.  Unsectioned fields (path-typed
  knobs, always-on flags, MD-only state) stay on the dataclass
  for the Python API + CLI but stay off the form.
* **ID stability**: by default `id = f"{id_prefix}-{field_name.replace('_', '-')}"`.
  Fields with legacy short IDs (e.g. `p-temperature` for
  `electronic_temperature`, `p-block-size` for
  `parallel_block_size`) declare `metadata["id_suffix"]` so the
  compatibility engine + sessionStorage list stay
  backwards-compatible.
* **Section ordering**: the dataclass can declare a class-level
  `_form_section_order` tuple to pin section order; otherwise
  sections appear in the order the first field declaring each
  section is declared.
* **Unknown engine** → 404 with `{ok: false, error: "..."}`.
* **No POST equivalent**: the schema is the contract, not a
  validator hook.  The existing `/api/build/{fdf,pyscf,preflight}`
  routes consume the JSON dict the JS collector produces from
  the rendered DOM.

Pin-tests in `tests/test_web.py::test_siesta_form_schema_matches_documented_layout`
and the PySCF counterpart lock the section names + per-section
field counts so a stray field-reorder doesn't silently rearrange
the UI.

---

## Request-size cap

`MAX_CONTENT_LENGTH = 50 MB` on the Flask app.  Watch uploads (large
trajectory logs) need the headroom; Build's typical PDB / XYZ
uploads are < 1 MB.  Oversized bodies → 413 with the standard
`{ok: false, error: "..."}` JSON shape (a Flask error handler
converts Werkzeug's default HTML 413 page into JSON so the JS
uploaders' `r.json()` doesn't crash).

---

## Front-end contract

All three tabs:
* Load 3Dmol.js from `cdnjs/3Dmol/2.1.0/3Dmol-min.js` (pinned).
* Share `static/lib/tabs.css` (the top-of-page Build / Modify / Watch
  nav) and `static/lib/tokens.css` (CSS custom properties for
  colours / radii / spacing).
* Share `static/lib/mol-style.js` (3Dmol style-spec builder), `mol-
  format.js` (chemical-formula renderer), `mol-pick.js` (selection
  halo helper used by Modify and Watch).
* Theme: dark.  CSS variables in `:root` for every colour.  No
  hardcoded `#fff` / `#000` in selectors.
* Every dynamic insertion uses `textContent` (not `innerHTML`).

The Build page (`index.html`) specifically:
* Layout: header, 12-col grid main, footer.
* Left column (controls): card "1. Build / Load", card "2. Generate
  input" (with two tabs SIESTA `.fdf` | PySCF script).
* Right column (viewer): card "Inspect" with a resizable 3Dmol
  viewer (CSS `resize: both` on `.viewer-wrap`).
* A `ResizeObserver` on `.viewer-wrap` calls `viewer.resize() +
  render()` on dimension change.
* Every successful build / load resets `state.fdf` / `state.pyscf`
  to null and disables the download buttons so the user can't
  accidentally download text from the previous structure.
* `sessionStorage["builder-structure"]` carries the Modify→Build
  handoff (M5); `sessionStorage["builder-form"]` survives form
  values across navigation.

### Form-side compatibility rules

`viewer.js::applyCompatibility()` locks parameter combinations that
would produce an invalid or wrong-physics config.  Runs on page load
and on `change` of any trigger input.  Each locked field gets
`disabled` + a `.lock-reason` hint span.

PySCF tab:
| trigger | dependent | lock |
|---|---|---|
| `method ∈ {RKS, RHF}`            | `spin`              | force `spin = 0` |
| `optimize = false`               | optimizer, geom_*, preopt | lock entire Optimization + Pre-opt sections |
| `optimize = true` AND `preopt = false` | preopt_*    | lock with "Pre-opt is disabled" |
| `solvent = ""`                   | solvent_method      | lock with "No solvent selected (gas phase)" |

SIESTA tab:
| trigger | dependent | lock |
|---|---|---|
| `spin_polarized = false` | spin_total                        | SpinTotal meaningless without polarisation |
| `relax_type = "none"`    | relax_steps, force_tol, max_displ | no MD block emitted |

### Defence in depth

The server does NOT trust the UI.  Even if a malicious or buggy
client submits an invalid combination, the same validators
(`validation.py:validate(struct, cfg)`) run server-side via field
metadata.  The UI rules give the user fast feedback; the server
rules protect the data.

---

## Forbidden patterns

The Flask app must NOT:

1. Run with `debug=True` by default — Flask's debugger allows
   arbitrary code execution.  Enable only via explicit `--debug`
   CLI flag.
2. Bind to `0.0.0.0` by default — that exposes `/api/watch/load`
   (reads any local file the server can access) to the network.
   Default `127.0.0.1`; print a loud warning when the user opts in
   to a non-loopback host (`warn_if_remote` in
   `web/blueprints/watch.py`).
3. Echo unsanitised user input as HTML.
4. Trust the UI's compatibility-locking to validate inputs — the
   server-side validation pass is the source of truth.

---

## Test reference

* `tests/test_web.py` — every Build + Modify endpoint × every documented payload variant.
* `tests/watch/test_api_load.py` — every Watch endpoint variant including directory-mode discovery and multi-stage merge.
* `tests/test_modify_e2e.py` — Playwright + live Flask end-to-end.
* `tests/test_review_fixes.py::test_s6_web_app_caps_upload_size` — confirms the upload cap fires (HTTP 413, JSON body shape).
