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
| `/`         | Build  | this file                         | `web/blueprints/build.py` |
| `/modify`   | Modify | [`tabs/modify.md`](../tabs/modify.md) | `web/blueprints/modify.py` |
| `/watch`    | Watch  | [`tabs/watch.md`](../tabs/watch.md), [`watch-api.md`](watch-api.md) | `web/blueprints/watch.py` |

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
| `/api/build/preflight`        | POST | `{xyz, engine, params}` | `{ok, issues}` (`ok:true`) on success or config-parse fail | 400 missing `xyz` / unknown `engine` / unparseable `xyz`; otherwise 200 (issues carry config-parse errors) |
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
500 with `{"ok": false, "error": "missing dependency: <ImportError>"}`.
The error string carries the raw `ImportError` message (which
typically names the missing module); the CLI side carries
curated install hints — the web layer relies on the operator
already knowing how to install Python deps for the server.

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

Error-response policy:

* Missing `xyz`, unknown `engine`, or unparseable `xyz` → HTTP
  400 with `{"ok": false, "error": "<reason>"}`.  These are
  programmer / wiring errors — the caller should fix the
  request rather than display the failure to the user.
* Config-parse failure (the form sent values that don't coerce
  into the dataclass — e.g., a non-numeric mesh_cutoff) → HTTP
  200 with `{"ok": true, "issues": [{"severity": "error",
  "message": "bad parameters: <exc>", "where": "config"}]}`.
  The same issues panel renders this alongside warn-severity
  field-range messages, so the UI doesn't need a separate
  error-handling branch for "user-typed-something-invalid".

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

## `/api/watch/*` — Watch blueprint (mounted in `web/blueprints/watch.py`)

| route | method | body | response | status |
|---|---|---|---|---|
| `/watch`             | GET  | — | `watch.html` (browser UI) | 200 |
| `/api/watch/formats` | GET  | — | `{ok, formats: [{name, label, description}, ...]}` | 200 |
| `/api/watch/load`    | POST | JSON `{path: "<abs-path>"}` OR multipart `file=` | `{ok, format, label, mtime, uploaded, ...}` | 400 bad input · 403 outside `MOLBUILDER_WATCH_ROOT` · 404 missing file · 500 parse fail |
| `/api/watch/data`    | GET  | `?mtime=<client-cached-float>` (optional) | `{ok, changed, mtime, path, format, label, data, uploaded}` or `{ok, changed: false, mtime}` | 200 (errors carry `{ok:false, error}`) |

The Watch app is **single-user, single-tab** by design (see
`design.md` "Watch — live trajectory viewer"): one global "current
file" dict guarded by a `threading.Lock`.  Two clients pointing at
the same server share state.  This is intentional — the watch app
isn't a multi-tenant service.

**State machine** for `/api/watch/load`:

1. Multipart upload → save to a `tempfile.NamedTemporaryFile`,
   parse, register the temp path as the active file.  A second
   upload deletes the previous temp (atexit cleanup catches the
   final tab-close).
2. JSON `{path: ...}` → resolve to an absolute path, optionally
   constrained by the `MOLBUILDER_WATCH_ROOT` env var (see below),
   detect the parser, parse, store as the active file.

`/api/watch/data` is the polling endpoint; the JS calls it every
~15 s.  When `client_mtime` matches the server's cached mtime,
the response is the minimal `{ok, changed: false, mtime}` so the
common no-change path doesn't reserialise the whole trajectory.

### `MOLBUILDER_WATCH_ROOT` env var

When the operator sets `MOLBUILDER_WATCH_ROOT=/abs/path` before
starting the server, `/api/watch/load` refuses to read any path
outside that subtree (HTTP 403, structured error).  This is the
intended deployment posture on shared / multi-user hosts: scope
the read-arbitrary-file primitive to the operator's intended run
area.  Unset by default; the server trusts the caller's path when
the env var is absent.

### `/api/watch/data` response shape (`changed: true` branch)

```json
{
  "ok":       true,
  "changed":  true,
  "path":     "<absolute path of the active file>",
  "mtime":    <unix epoch seconds, float>,
  "format":   "siesta" | "pyscf" | "molwatch" | ...,
  "label":    "<human label from the parser>",
  "data":     { /* legacy molwatch-v1 trajectory dict */ },
  "uploaded": <bool — true for multipart-upload temp files>
}
```

The `data` sub-dict is the legacy v1 trajectory shape produced by
`parsers/__init__.py::trajectory_to_legacy_dict`:

```json
{
  "frames":        [ [[el, x, y, z], ...], ... ],
  "energies":      [<float|null>, ...],
  "max_forces":    [<float|null>, ...],
  "forces":        [ [[fx, fy, fz], ...] | [], ...],
  "iterations":    [<int>, ...],
  "step_indices":  [<int>, ...],
  "wall_times":    [<float|null>, ...],
  "scf_history":   [ [{"cycle", "energy", "delta_E", ...}, ...], ... ] | [],
  "lattice":       [[ax,ay,az], [bx,by,bz], [cx,cy,cz]] | null,
  "source_format": "<engine name>",
  "run_state":     "ongoing" | "finished" | "errored",
  "error_message": "<string when run_state=errored>",
  "stages":        [<merged stage info — multi-stage runs>] | []
}
```

Multi-stage runs (job-layout v1, multiple `<basename>-stage<N>.molwatch.log`
files in the same directory) get one stage entry per file in
`stages`; the frontend uses those to draw stage-boundary markers
on plots.

---

## `/api/files/*` — Server-side file explorer

Backend for the **Projects tab** — a persistent column-view file
explorer that lets the user browse the *server's* filesystem and
pick a file or directory.  Selection is shared state (sessionStorage)
that other tabs (Spectra, Watch, Modify, Build) observe via the
``storage`` event, JupyterLab-style.

Solves the recurring UX problem that `<input type="file">` opens the
*browser's* local file dialog, which is useless when the data already
lives on the server.  The Projects tab is the canonical place to
"walk into my project tree and find the right file to keep working
on"; other tabs react to its current selection rather than each
maintaining their own file picker.

This API is **additive**: every tab keeps its existing local-file
input + (where applicable) raw-text paste.  The Projects-tab
selection is a third path, not a replacement.

This is **not** a file manager: no upload, rename, delete, or move
in v1.  The Projects tab is a navigation + selection widget.  Files
get created by molbuilder's own generators (Build, run wrapper, the
Phase 2 derive-job flow); they get deleted by the user at their
shell.

### Endpoints

| route | method | query | response | status |
|---|---|---|---|---|
| `/api/files/roots` | GET | — | `{ok, roots: [{path, label, exists}, ...]}` | 200 |
| `/api/files/list`  | GET | `path` (required), `ext` (optional, comma-sep filter) | `{ok, path, entries: [...]}` | 400 outside-root · 404 missing dir · 200 |
| `/api/files/stat`  | GET | `path` (required) | `{ok, path, kind, size, mtime}` | 400 outside-root · 404 missing · 200 |
| `/api/files/read`  | GET | `path` (required), `max_bytes` (optional, default 1 MB) | `{ok, path, kind, size, mtime, text}` | 400 outside-root · 404 missing · 413 too large · 200 |

### Roots — what the picker is allowed to browse

Resolved from `Capabilities.file_picker_roots()`:

1. **Defaults**, always included:
   * `<cwd>/projects/` -- the canonical projects hierarchy (`molbuilder.projects.projects_root()`); included even if it doesn't exist yet so the picker can show "create a project" UX later.
   * `<cwd>` -- the working directory the server was launched from, so a one-off `molbuilder serve` in a scratch dir gives the user immediate access to its files.
2. **User additions** from `molbuilder.json`:
   ```jsonc
   {
     "file_picker": {
       "roots": ["~/scratch", "/data/shared/molbuilder"]
     }
   }
   ```
   Each entry is expanded (`~`, `$VARS`), resolved to absolute, and
   added to the list if it exists on this machine.  Non-existent
   entries are dropped silently (a stale config entry on a fresh
   machine shouldn't break the picker).

Returned `roots` carry a `label` so the UI can show a friendly name
(`"projects"`, `"CWD"`, basename of user roots) without the full
path crowding the tree.

### Path validation

A single helper validates every `path` query arg:

1. Expand `~` and environment variables.
2. Resolve to absolute (`Path.resolve()` -- follows symlinks).
3. Must equal-or-be-inside one of the allowed roots.
4. Reject `..` components in the raw input (defense in depth -- step 3 already covers traversal, but rejecting `..` early gives a cleaner error).

A path that survives validation is canonical absolute; that's what
the response carries back.

### Response shapes

`/api/files/roots`:

```json
{
  "ok": true,
  "roots": [
    {"path": "/home/qqing/molbuilder/projects", "label": "projects", "exists": true},
    {"path": "/home/qqing/molbuilder",          "label": "CWD",      "exists": true},
    {"path": "/data/shared/molbuilder",         "label": "shared",   "exists": false}
  ]
}
```

`/api/files/list` entries:

```json
{
  "ok": true,
  "path": "/home/qqing/molbuilder/projects/tunneling/spectrum",
  "entries": [
    {"name": "BDT_water", "kind": "directory", "size": null, "mtime": 1715773200.0},
    {"name": "BDT_NH2",   "kind": "directory", "size": null, "mtime": 1715773100.0}
  ]
}
```

`kind` is one of `"directory"`, `"file"`, `"symlink"`, `"other"`.
Hidden entries (starting with `.`) are filtered out by default --
the picker is for project files, not dotfiles.

### Job derivation — picker as entry point for new runs (Phase 2)

A second, related workflow the picker enables: **start a new run job
from an existing file** (e.g., take the optimized geometry out of an
`optimization/<structure>/` run and start a `spectrum/<structure>/`
calculation from it).

Realised by:

| route | method | body | response | status |
|---|---|---|---|---|
| `/api/files/derive_job` | POST | `{source_path, target_topic, target_structure?, target_project?, target_tab?}` | `{ok, target_path, structure_xyz, suggested_config}` | 400 invalid source · 400 invalid topic · 409 target exists · 200 |

`derive_job` does NOT write any files itself.  It computes:

* `target_path` -- the canonical `<project>/<target_topic>/<target_structure>/`
  per `molbuilder.projects` (rejected if topic is not in
  `CANONICAL_TOPICS`); defaults: `target_project` = source's project,
  `target_structure` = source's structure name.
* `structure_xyz` -- the geometry extracted from the source.  Source
  types and what we extract:
  * `<job>_optimized.xyz`            -> the XYZ directly.
  * `<job>.molwatch.log`             -> last frame's coordinates.
  * `<job>.spectra.json`             -> `equilibrium.positions_ang` + elements.
  * `<job>.thermo.txt`               -> the relaxed geometry referenced in the header (resolved against the file's own dir).
  * plain `.xyz` / `.pdb`            -> read as-is.
* `suggested_config` -- a tiny dict the receiving tab pre-populates
  its form with (e.g., from a spectra JSON: `{method, functional,
  basis}` lifted from the source's `config` so the new job inherits
  the same theory level unless the user changes it).

The frontend then:
1. Navigates to the target tab (`target_tab` defaults from `target_topic`:
   `spectrum` → Spectra, `optimization` → Build, etc.).
2. Pre-loads the structure + suggested config.
3. The user reviews + clicks Generate.  The Generate flow uses
   `target_path` as the output directory, and the user's tab
   navigates Watch / Spectra at `target_path` for live monitoring
   once the run starts.

This keeps each calc dir self-contained (no symlinks; coordinates
inlined into the new script) per the design.md 2026-05-14 decision.

### Naming constraint (project / structure / topic)

All names that participate in a `projects/<project>/<topic>/<structure>/`
path MUST satisfy ``molbuilder.projects.validate_name`` -- i.e., the
regex ``^[A-Za-z0-9_-]+$``.  Spaces, dots, slashes, unicode are
rejected.  The constraint exists because SIESTA's filename discovery
is basename-based; a structure named ``"my mol.run #1"`` would
silently break that pipeline downstream.

This is enforced at three layers and surfaces at each:

* **Path construction** (``molbuilder.projects.*``): raises
  ``InvalidName`` on bad input.  All directory-creating code paths
  go through these constructors.
* **`/api/files/derive_job`** (Phase 2): validates ``target_project``
  and ``target_structure`` via the same helpers, returns HTTP 400
  with the ``InvalidName`` message verbatim so the UI can echo it
  next to the form field.
* **Future "create new project" UI**: the form will validate
  client-side against the same regex, then re-validate server-side
  on submit.

`topic` is even more constrained: must be one of the six
``CANONICAL_TOPICS`` (``optimization``, ``frequency``, ``spectrum``,
``transport``, ``single-point``, ``scan``).  Validated by
``molbuilder.projects.validate_topic``.

The picker itself does NOT filter on-disk directory names -- it shows
what's there.  A user who hand-creates a directory named
``my project/`` will see it in the picker tree (so they can find and
rename it) but won't be able to use it as the target of a derive-job
without renaming.

### Projects-hierarchy convention

The picker is intentionally *generic* -- it doesn't enforce the
`<project>/<topic>/<structure>/` shape that `molbuilder.projects`
documents, because users may want to load files from outside
`projects/` too (browsing `~/Downloads/`, a one-off scratch dir).

But when the user IS inside `projects/`, the path naturally
reflects the hierarchy:

```
projects/<project>/<topic>/<structure>/<file>
         └─ tree expandable ─┘ └─ FLAT directory; files only by ──┘
                                   convention (no subdirs)
```

The frontend can detect "we're inside projects" by prefix-matching
against the `projects` root and render topic-aware labels.  Beyond
the topic level, the structure directory is flat by job-layout-v1
convention; the picker will still show whatever exists there
(including any subdirs the user created off-spec), so the contract
stays honest about the actual filesystem state.

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
