# Spec — Flask web UI + API

**Module**: `molbuilder/web/app.py` &nbsp;·&nbsp; **Templates**:
`molbuilder/web/templates/index.html` &nbsp;·&nbsp; **Static**:
`molbuilder/web/static/{viewer.js,style.css}` &nbsp;·&nbsp; **Tests**:
`tests/test_web.py`, `tests/test_review_fixes.py` (S6)

The web UI is a single HTML page that talks to a Flask backend.  It
should mirror what the CLI does, with a 3D viewer (3Dmol.js) and a
form-driven interface.

## Endpoints

| route | method | body | response | error code |
| --- | --- | --- | --- | --- |
| `/`           | GET  | — | HTML page | — |
| `/api/health` | GET  | — | `{ok: true, version: "x.y.z"}` | — |
| `/api/backends` | GET | — | `{ok, available: {rdkit: bool, amber: bool}}` | — |
| `/api/build`  | POST | `{kind, input, ...}` | structure JSON (see below) | 400 bad input, 500 missing dep |
| `/api/load`   | POST | json `{text, format, filename}` OR multipart `file=` | structure JSON | 400 empty, 413 too big |
| `/api/fdf`    | POST | `{xyz, params}` | `{ok, fdf, system_label}` | 400 bad params, 500 render |
| `/api/pyscf`  | POST | `{xyz, params}` | `{ok, script, job_name}` | 400 bad params, 500 render |

### Request size cap

`MAX_CONTENT_LENGTH = 10 MiB` on the Flask app.  This applies to
`/api/load` (real PDB / XYZ uploads) but caps every endpoint.  Larger
bodies → 413 (Payload Too Large) automatically.

### Structure JSON shape

`/api/build` and `/api/load` both return the same shape on success:

```json
{
  "ok":          true,
  "xyz":         "<xyz text>",
  "pdb":         "<pdb text>",
  "n_atoms":     int,
  "n_residues":  int,
  "summary":     "<Structure 'title': N atoms, R residues, formula CxHyOz>",
  "title":       str,
  "elements":    ["C", "H", ...],
  "source_format": "xyz" | "pdb"          // /api/load only
}
```

### Error JSON shape

```json
{ "ok": false, "error": "human-readable message" }
```

## `/api/build` payload

```json
{
  "kind":     "peptide" | "dna" | "rna" | "smiles" | "name",
  "input":    str,
  // DNA / RNA only:
  "backend":   "auto" | "rdkit" | "amber",
  "form":      "B" | "A" | "Z",
  "terminal":  "OH" | "5P" | "3P" | "PP",
  "protonate_phosphates": bool
}
```

Unknown `kind` → 400 with a list of valid values.  Empty `input` →
400.  Missing optional dep (e.g. PeptideBuilder for `peptide`,
PubChemPy for `name`) → 500 with install hint.

## `/api/load` payload variants

JSON body:
```json
{
  "text":     "<xyz or pdb text>",
  "format":   "xyz" | "pdb" | "auto",
  "filename": str       // helps format auto-detect, optional
}
```

Multipart form-data:
```
file=<uploaded file>
```

Format detection precedence:
1. Explicit `format` field (if not `"auto"`).
2. Filename extension (`.xyz` / `.pdb`).
3. Content sniff: first line is digits → `xyz`, else `pdb`.

## `/api/fdf` payload (SIESTA)

```json
{
  "xyz":    "<xyz text from a previous build>",
  "params": { /* SiestaConfig fields */ }
}
```

Server-side:
* Parses `xyz` via `Structure.from_xyz` (the canonical parser; the
  legacy `_xyz_to_structure` wrapper delegates to it — T5 fix).
* Filters `params` against `fields(SiestaConfig)` so unknown keys
  are silently dropped.
* Special-case `kgrid`: incoming `[a, b, c]` list converted to
  `(int, int, int)` tuple.
* Special-case `net_charge`: empty string or null → `None`
  (auto-detect).

## `/api/pyscf` payload

Same shape as `/api/fdf` but mapped to `PySCFConfig`.  Special-cases:
* `dispersion = "none"` → `None` (no dispersion).
* `solvent = ""` → `None` (gas phase).
* `auxbasis = ""` → `None` (let `density_fit()` pick).

## Front-end contract

The HTML page:

* Loads 3Dmol.js from `cdnjs/3Dmol/2.1.0/3Dmol-min.js` (pinned).
* Has a top-level layout: header, 12-col grid main, footer.
  * Left column (controls): card "1. Build / Load", card "2. Generate
    input" (with two tabs).
  * Right column (viewer): card "Inspect" containing the resizable
    3Dmol viewer (CSS `resize: both` on `.viewer-wrap`).
* Tabs in card 2: SIESTA `.fdf` | PySCF script.  Switching preserves
  each tab's last-generated output.
* Theme: dark.  CSS variables in `:root` for every colour.  No
  hardcoded `#fff` / `#000` in selectors.
* Resize-aware: a `ResizeObserver` on `.viewer-wrap` calls
  `viewer.resize() + render()` on dimension change, so the WebGL
  canvas tracks the user's drag.
* Stale state: every successful build/load resets `state.fdf` /
  `state.pyscf` to null and disables the `.fdf` / `.py` download
  buttons, so the user can't accidentally download text from the
  previous structure.

## Forbidden patterns

The Flask app must NOT:

1. Run with `debug=True` by default — Flask's debugger allows
   arbitrary code execution.  Enable only via explicit `--debug`
   CLI flag.
2. Bind to `0.0.0.0` by default — that exposes `/api/load` (which
   reads any local file the server can access) to the network.
   Default `127.0.0.1`; print a loud warning when the user opts in
   to a non-loopback host.
3. Echo unsanitised user input as HTML — every dynamic insertion
   uses `textContent` (not `innerHTML`).

## Test reference

* `test_web.py` — every endpoint × every documented payload variant.
* `test_review_fixes.py::test_s6_web_app_caps_upload_size` — confirms
  the 10 MiB cap fires (HTTP 413).
