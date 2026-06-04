# molbuilder — design and roadmap

This document is the durable design reference for molbuilder. It captures
mission, architectural principles, decisions made, and active roadmap items
that span multiple sessions of work. Per-component test contracts live under
[`docs/`](README.md) (categorised by purpose: `protocols/`, `types/`,
`engines/`, `tabs/`); this document sits above them.

When in doubt about whether to do something, read this file first. When the
file is wrong (the decision changed, the constraint shifted), update it in
the same PR as the code change.

---

## Mission

molbuilder builds 3-D molecular structures from sequence / SMILES / name
input, modifies them into derived geometries (e.g. metal-molecule-metal
nanojunctions), generates SIESTA and PySCF input files for those structures,
and provides a live trajectory viewer that monitors the resulting calculations.

The package is a single, internally-coherent toolkit covering the full
pipeline:

```
sequence ──► Structure ──► (modify) ──► SIESTA .fdf  ──► siesta ──┐
                                    └─► PySCF .py    ──► python  ─┴──► .molwatch.log
                                                                              │
                                                           ◄──── live watch ──┘
```

Both halves were initially separate repos (`Qing-LAB/molbuilder` for the
build side, `Qing-LAB/molwatch` for the watch side). They are being merged
into `molbuilder` because they share a single file-format contract
(`.molwatch.log v1`), a single core dataclass (`Structure`), and the same
Flask + 3Dmol.js stack. See "Merge plan" below.

---

## 0. Document index

This file is the **master**: it holds principles, cross-cutting
decisions, architecture, and an index to every subsystem doc.
Each subsystem doc is the **sole source of truth** for its own
surface — when you're working on the sidebar, read the sidebar
doc, not this one. New cross-cutting decisions land here;
subsystem-specific decisions land in the subsystem doc.

### Cross-cutting (this file)

| § | Topic |
|---|---|
| 1 | Mission |
| 2 | Architecture (three layers + four core types) |
| 3 | Design principles (numbered, named) |
| 4 | Decisions log (cross-cutting only) |
| 5 | Anti-patterns we refuse |
| 6 | Watch — live trajectory viewer (legacy spec) |
| 7+ | Migration history, science gaps, file format |

### Protocols — wire / JS / test contracts (`docs/protocols/`)

| Doc | Owns |
|---|---|
| [`web-api.md`](protocols/web-api.md) | HTTP `/api/*` endpoint reference (request/response shapes) |
| [`projects-sidebar.md`](protocols/projects-sidebar.md) | Sidebar architecture, public `projects.*` API, lock model, capability table |
| [`atom-selection.md`](protocols/atom-selection.md) | Selection store, `.molstruct.json` sidecar shape, viewer adapter |
| [`selection.md`](protocols/selection.md) | Python selection rule grammar (`by_element`, `by_index_range`, …) |
| [`results-tab.md`](protocols/results-tab.md) | `/results` dispatch architecture |
| [`runtime-registry.md`](protocols/runtime-registry.md) | `molbuilder-runtime.js` register/whenReady contract |
| [`inspector-registry.md`](protocols/inspector-registry.md) | Inspector `mount(host, file, ctx) → {dispose}` contract |
| [`embedded-viewer.md`](protocols/embedded-viewer.md) | Standard embeddable 3D viewer — `viewer.embed(host, opts) → handle` contract |
| [`playwright-tests.md`](protocols/playwright-tests.md) | Test design patterns + anti-patterns |
| [`job-layout.md`](protocols/job-layout.md) | On-disk basename + `*-runN.out` convention |
| [`cli.md`](protocols/cli.md) | click-based CLI conventions |

`/api/watch/*` endpoints are documented in
[`web-api.md`](protocols/web-api.md) § 8 (they back the
trajectory inspector on `/results`; the legacy `/watch` page is
archived).

### UI tabs (`docs/tabs/`)

| Doc | Owns |
|---|---|
| [`build.md`](tabs/build.md) | `/build` tab — structure-from-input + SIESTA/PySCF form |
| [`modify.md`](tabs/modify.md) | `/modify` tab — atom selection + nanojunction assembly |
| [`spectra/spec.md`](tabs/spectra/spec.md) | `/spectra` tab — IR/Raman generator |
| [`results.md`](tabs/results.md) | `/results` tab — registry dispatch, file picker (planned) |

### Engines (`docs/engines/`)

[`siesta.md`](engines/siesta.md) · [`pyscf.md`](engines/pyscf.md) · [`builders.md`](engines/builders.md)

### Types — L1 data contracts (`docs/types/`)

[`structure.md`](types/structure.md) · [`parsers.md`](types/parsers.md) · [`chemistry.md`](types/chemistry.md)

### Ops & deployment

[`README.md`](README.md) · [`README_install.md`](README_install.md) (four-env install model) · [`deployment.md`](deployment.md)

### Historical (archived — NOT a source of truth)

[`archive/README.md`](archive/README.md) catalogues superseded
docs. Anything date-prefixed (`YYYY-MM-DD-<name>.md`) is
history; read the canonical doc listed in the archive index
instead.

### Why this hierarchy

The folder split is by **kind of contract**, not by feature.
This is the load-bearing reason every spec lands in exactly one
place and there is no "where do I put this?" ambiguity:

- **`protocols/`** — how parts of the system **talk to each
  other**. HTTP wire shapes, JS module surfaces, on-disk file
  conventions, test patterns. Specs here pin contracts between
  components.
- **`types/`** — the **shape of values** flowing between
  components. Structure dataclass, parser output dicts,
  chemistry helpers. Specs here describe data, not behaviour.
- **`engines/`** — per-engine emitter specs. One per downstream
  code we generate input for, plus the build-backend contract.
  Specs here describe **what we write**, not what we do.
- **`tabs/`** — per-UI-tab specs. Single-file specs stay as
  `<tab>.md`; tabs needing multiple assets (bibliography,
  sub-specs) become subfolders, e.g. `tabs/spectra/spec.md` +
  `tabs/spectra/references.bib`. Bibliographies always live
  alongside the spec that cites them — one place to look for
  both.

The split is what makes the **sole-source-of-truth** rule
enforceable: each contract has one canonical home; cross-cutting
principles live here in `design.md` with backlinks.

---

## Architecture

**Three layers and four core types.** The layers describe directionality
of imports and responsibility; the types describe the data that flows
between layers.

```
┌──────────────────────────────────────────────────────────┐
│  L3 — Surfaces                                            │
│  cli.py (click), web/app.py (Flask + Blueprints)          │
│  Convert UI gestures → L2 calls.  No business logic.      │
├──────────────────────────────────────────────────────────┤
│  L2 — Domain verbs                                        │
│  builders/, generators/, parsers/, validation.py          │
│  Each verb is a focused module operating on L1 types.     │
├──────────────────────────────────────────────────────────┤
│  L1 — Core types (nouns)                                  │
│  structure.py, frame.py, config/, issues.py               │
│  + chemistry, residues, trajectory_log/                   │
│  Pure data + minimal serialization.  Field metadata here. │
└──────────────────────────────────────────────────────────┘
```

**Layering rule (load-bearing):** higher layers may import any lower
layer; lower layers must never import higher ones. L1 modules cannot
import from L2 or L3. L3 imports L1 + L2 only through the public
package surface (`from molbuilder import ...`).

This is the single most important architectural invariant. Without it,
the package will recreate the registry/abstraction tangle that was
deleted in favor of dataclass-driven introspection.

### Core types (L1)

Four types are the lingua franca; everything else is a verb operating
on them.

| Type | Role | Layer |
|---|---|---|
| `Structure` | One geometric configuration: elements, positions, PDB metadata. Build-side. | `structure.py` |
| `Frame` | A `Structure` plus per-step physics (energy, forces, lattice, step_index, scf_history). Parse-side. | `frame.py` |
| `SiestaConfig`, `PySCFConfig` | Emission parameters for each backend. Carry the field metadata that drives CLI options, web form schema, and validation. | `config/siesta.py`, `config/pyscf.py` |
| `Issue` | A validation finding: `severity` (error/warn), `message`, `where` (field name or "geometry"). | `issues.py` |

Plain `@dataclass` everywhere. No builder patterns, no pydantic, no
custom base classes. Field metadata via `dataclasses.field(metadata=...)`
(see "Field metadata as the unifier" below).

#### `Frame` and `Trajectory`

`Structure` carries one geometric configuration. Builders emit it; FDF /
PySCF generators consume it. It does **not** carry energies, forces,
lattice, or trajectory metadata — those belong on `Frame`.

```python
@dataclass
class Frame:
    structure:    Structure                       # geometry of this step
    step_index:   int                             # 0-based; preview frame is 0
    energy:       Optional[float] = None          # eV
    forces:       Optional[np.ndarray] = None     # (N, 3), eV/Ang
    max_force:    Optional[float] = None          # eV/Ang
    lattice:      Optional[np.ndarray] = None     # (3, 3) Ang, or None for vacuum
    scf_history:  Optional[List[Dict[str, float]]] = None
                                                  # per-cycle convergence dicts:
                                                  # {cycle, energy, delta_E, ...}
                                                  # keys are engine-specific
                                                  # (gnorm/ddm for PySCF and the
                                                  # molwatch_log; dHmax/dDmax for
                                                  # SIESTA). Consumers should not
                                                  # assume a fixed key set.
```

A trajectory is wrapped in a minimal `Trajectory` dataclass that
carries `source_format`, the list of frames, and an optional shared
`lattice` (the cell when constant across frames):

```python
@dataclass
class Trajectory:
    source_format: str                          # "siesta" / "pyscf" / "molwatch" / ...
    frames:        List[Frame]
    lattice:       Optional[np.ndarray] = None  # (3, 3) Ang, or None
```

The wrapper exists for two reasons.  First, `source_format` is a
file-level string that the molwatch unified-log parser pulls from the
`# engine:` header — a `.molwatch.log` written by a SIESTA run keeps
`source_format="siesta"` even though the parser class is
`MolwatchLogParser`.  An `Iterator[Frame]` alone has no slot for
that.  Second, every current parser produces a single shared lattice
(or none) rather than per-frame lattices; lifting that onto the
trajectory matches the data and avoids redundant per-frame copies.
Per-frame `Frame.lattice` is preserved for variable-cell trajectories
that no current parser produces; today every parser sets
`Frame.lattice = None` and puts the cell on `Trajectory.lattice`.

`Trajectory` supports `len()`, iteration, and indexing so simple
callers can treat it as a frame list.  Phase 3 may grow it (analysis
methods, on-disk serialization); Phase 2 keeps it minimal.

Parser interface (`parsers/base.py`):

```python
class TrajectoryParser(ABC):
    name:  str       # e.g. "siesta"
    label: str       # human-facing
    hint:  str       # one-line user hint shown when detection fails
    @classmethod
    def can_parse(cls, path: str) -> bool: ...
    @classmethod
    def parse(cls, path: str) -> Trajectory: ...
```

The dispatch (`detect_parser`) and registry shape are unchanged from
the molwatch implementation.

Web layer / legacy adapter: the JS client still consumes the molwatch
v1 dict shape, so
`molbuilder/parsers/__init__.py:trajectory_to_legacy_dict` flattens a
`Trajectory` back to the historical dict at the `/api/watch/load`
boundary.  Phase 3 redesigns the JSON to surface Trajectory directly;
the adapter goes away then.

### Domain verbs (L2)

> **Notes on the as-shipped layout.**  The top-level builder modules
> (`molbuilder/peptide.py`, `nucleic.py`, `smiles.py`, `pubchem.py`)
> are the L2 verbs; `builders/` is reserved for `builders/backends/*`.
> The generators ship at `molbuilder/siesta/input.py` and
> `molbuilder/pyscf/input.py` rather than a flat `generators/`
> directory; the per-engine subpackages let the configs and emitters
> co-locate (the configs themselves live at L1 in `molbuilder/config/`
> and are re-exported from each engine package for back-compat).

| Verb | Module | Consumes | Yields |
|---|---|---|---|
| Build | `builders/peptide.py`, `builders/nucleic.py`, `builders/smiles.py`, `builders/pubchem.py` | sequence / SMILES / name + builder backend | `Structure` |
| Build (backends) | `builders/backends/_amber.py`, `_rdkit.py`, `_threedna.py` | builder request | `Structure` (or `BackendUnavailable`) |
| Generate | `generators/siesta.py:render_fdf`, `generators/pyscf.py:render_script` | `Structure` + `Config` | string (the .fdf or .py text) |
| Parse | `parsers/molwatch_log.py`, `parsers/siesta.py`, `parsers/pyscf.py` | trajectory file path | `Trajectory` (i.e. `(source_format, List[Frame], lattice)`) |
| Validate | `validation.py:validate_geometry` | `Structure`, `Config` | `List[Issue]` |
| Write log | `trajectory_log/format.py` | `Frame` (or initial `Structure`) | appends a block to `.molwatch.log` |

Each verb is small, takes L1 types in, and returns L1 types out. No
verb hides state in module-level globals (apart from the parser
registry, which is a literal `PARSERS = [...]` list).

### Surfaces (L3)

**CLI — click, not argparse.** A small (~30-line) bridge walks
`dataclasses.fields(Config)` and adds one `click.option` per field
using the metadata; the rest is plain click. We do **not** write our
own argument parser, registry, or coercion layer — click handles type
conversion, help text, choice validation, and `--help` rendering. The
bridge converts our `field.metadata` dict into click's existing
parameters; no extension framework on top of click.

**Web — Flask + Blueprints.** The Build and Watch route groups become
two `flask.Blueprint`s registered at `/api/build` and `/api/watch`.
Blueprints are Flask's native mechanism for URL prefixing; we don't
roll a custom router. Each route handler is a thin wrapper:
deserialize → call L2 verb → serialize. No business logic.

CLI surface (Phase 5 target):

```bash
molbuilder peptide  ASEQ                  # Structure → stdout XYZ (default)
molbuilder dna      ATGC                  
molbuilder rna      AUGC
molbuilder smiles   "C1=CC=CC=C1"
molbuilder name     "aspirin"

molbuilder fdf      [in.xyz|-]  out.fdf
molbuilder pyscf    [in.xyz|-]  out.py

molbuilder validate [in.xyz|-]  [--config siesta.fdf|pyscf.py]
                                          # → JSON Issue list to stdout
                                          # exit 1 on any error-severity issue

molbuilder watch parse  <traj>            # → JSON frames to stdout
molbuilder watch tail   <traj>            # → NDJSON, one frame per line
molbuilder watch serve  [--port]          # Flask portal
```

Pipe contract:
- `-` reads the appropriate input from stdin where it makes sense.
- Machine-consumable subcommands (`watch parse`, `watch tail`,
  `validate`, anything with `--json-summary`) emit JSON / NDJSON to
  stdout. Default stdout is human text or the generated file body.
- Status / progress / warnings always go to stderr.

Web routes:

> **Both halves are now namespaced.**  Build routes live at
> `/api/build/{molecule,load,fdf,pyscf}` (the verb-builder endpoint
> is `molecule` rather than per-kind sub-routes; per-kind splitting
> can come later if useful).  Watch routes at `/api/watch/*`.
> The two top-level routes shared between tabs (`/api/health`,
> `/api/backends`) stay un-namespaced.

```
GET  /                              # tabbed UI shell
GET  /api/backends                  # available builder backends
                                    # (lifted from build blueprint;
                                    # consumed by both tabs' Backend pickers)

# Build blueprint  (mounted at /api/build)
POST /api/build/peptide
POST /api/build/dna
POST /api/build/rna
POST /api/build/smiles
POST /api/build/name
POST /api/build/load                # XYZ/PDB upload → Structure JSON
POST /api/build/fdf                 # → text
POST /api/build/pyscf               # → text
POST /api/build/validate            # → Issue list JSON

# Watch blueprint (mounted at /api/watch)
GET  /api/watch/formats             # registered parsers
POST /api/watch/load                # trajectory file → Frame list JSON
GET  /api/watch/data                # browser-driven polling (~15s)
```

### Field metadata as the unifier

Every L1 config field carries:

```python
metadata = {
    "label":    "Mesh cutoff",
    "unit":     "Ry",
    "range":    (50, 600),
    "choices":  None,                  # or list of allowed values
    "help":     "Real-space mesh cutoff. Lower = faster but less converged.",
    "tier":     "advanced",            # basic | advanced (default UI visibility)
    "validate": lambda v: None or Issue(...)   # optional callable
}
```

One source feeds:

- **CLI**: a ~30-line `add_dataclass_options(cmd, ConfigCls)` helper
  walks fields and applies `click.option` per field using the metadata.
- **Web form**: `dataclass_to_form_schema(ConfigCls)` returns JSON the
  frontend renders into form controls. Same fields, same labels, same
  ranges as the CLI.
- **Validators**: `validation.py` reads `range` / `validate` per field.
  An out-of-range value yields one `Issue`.
- **Spec docs**: per-engine and per-tab specs under `docs/engines/` and
  `docs/tabs/` can be (semi-)generated from metadata so they don't drift.

This is what makes the dataclass-as-source-of-truth principle real
rather than aspirational.

---

## Watch — live trajectory viewer

The "watch" half of the pipeline. A Flask + 3Dmol.js viewer that points
at an in-progress SIESTA / PySCF run and shows what the calculation is
doing in near-real-time.

### What it shows

For a trajectory file the user loads, the viewer renders:

- The **molecular geometry**, frame-by-frame, in a 3Dmol GLViewer
  (`addModelsAsFrames` movie mode — frames are loaded once, animated
  client-side; no per-frame round-trip).
- **Total energy** vs step (Plotly line plot).
- **Max atomic force** vs step (Plotly).
- **Per-cycle SCF convergence** for the active step — both the energy
  trajectory and the residual norm (`gnorm` / `dDmax`) on log scales,
  so the user can spot stalled or oscillating SCFs while the run is
  still going.

Single page, four control tabs: **Style** (representation, radius,
element coloring, background, cell visibility), **Overlays** (atom
indices, force arrows with magnitude threshold, highlight max-force
atom), **Inspect** (click two atoms in the viewer to see their
indices, elements, per-frame coordinates, and live |A−B| distance),
**Playback** (slider, prev / play / pause / next, speed, loop).
Frame counter "X / N" sits above the slider.

The Inspect picks reset on file load and on file-picker upload, but
persist across polling refreshes (the picked atom indices stay
meaningful as long as the trajectory is the same — only coordinates
update each frame).

### Supported inputs (auto-detect)

`detect_parser(path)` walks the registered `PARSERS` list in order;
first parser whose `can_parse(path)` returns True wins.

| Parser | Input | Detection signal |
|---|---|---|
| `MolwatchLogParser` | `<job>.molwatch.log` (preferred path) | `# molwatch trajectory log v1` header marker on line 1 |
| `SiestaParser` | `run.out` / `siesta.log` (engine stdout) | `Welcome to SIESTA` banner near top of file |
| `PySCFParser` | `<job>_geom_optim.xyz` (geomeTRIC trajectory) | multi-frame XYZ structure; reads sibling `.qdata` for forces and `.log` for SCF history if present |

`MolwatchLogParser` is first because `.molwatch.log` is self-contained
— one file carries trajectory + per-cycle SCF data + initial preview,
no sibling-file discovery. SIESTA and PySCF parsers remain as
fallbacks for runs that weren't generated through molbuilder.

When detection fails (`UnknownFormatError`) the message lists every
supported format with its hint, plus a targeted suggestion for the
two most common foot-guns:

- Loading a SIESTA `.fdf` (it's the **input**, not the output — point
  the user at the redirected stdout).
- Loading a raw PySCF `.log` (parser wants the geomeTRIC
  `_geom_optim.xyz` instead).

### Cross-tool contract: step 0 preview

`molbuilder/siesta/convert()` and `molbuilder/pyscf/render_script()`
write a sibling `<job>.molwatch.log` containing **step 0** — the
initial-geometry preview — at file-emission time, **before any SCF
runs**. Step 0 carries coordinates only; `energy=null`,
`forces=[]`, `scf_history=[]`.

This means a user who has just generated the job can open the
`.molwatch.log` in the watch viewer immediately and see the structure
they are about to compute. As SCF and optimisation progress, the
generated PySCF script's inlined `_MolwatchEmitter` appends step 1,
step 2, … to the same file; SIESTA's parser pulls steps from the
engine's own output independently.

This is the "molbuilder writes, molwatch reads" contract. Verified by
`tests/test_molwatch_preview.py::test_molwatch_can_parse_siesta_preview`
(round-trip: emit preview → re-parse → assert frame 0 matches the
input geometry, energies / forces are null / empty).

### Streaming / refresh model

Polling, not push (today). The client `setInterval`s (~15 s) →
`GET /api/watch/data`. Server compares the active file's `mtime` with
the last-seen value; if advanced, re-parses and returns the full
parsed result; otherwise returns `{changed: false}`. Cheap at idle,
no per-poll re-parse.

The format is **truncation-tolerant**: if the engine is still writing
the trajectory and the final block is half-written when we read, the
parser drops the torn block and returns the complete frames preceding
it. Next refresh picks up the now-complete block plus whatever came
after.

The polling model is intentional: the server has authoritative
knowledge of the file state and decides what to deliver, while the
browser is just a viewer.  SSE / push-style alternatives were
considered and dismissed -- they only help if change *detection* is
sub-second, which would require a background filesystem watcher
(inotify-style).  For SCF steps that take seconds-to-minutes, the
15s polling latency is rarely the bottleneck.

### State model — single user, single file

The Flask app holds one global `{path, mtime, data, parser, uploaded}`
state under a `Lock`. Locking is fine-grained — the app snapshots
path/mtime under the lock, drops it for the (potentially multi-MB)
re-parse, then re-acquires only to commit the result if the active
file hasn't changed under it (defensive against `/api/load` racing
`/api/data`).

This is **not** a multi-tenant service. The expected deployment is
"one user, one tab, one calculation"; for multi-user use, run a
separate process per user. The constraint is explicit, not
accidental — making the app multi-tenant is out of scope.

### What the watch app deliberately does NOT do

- It does not start, monitor, or kill the engine process. It only
  reads output files the engine produces.
- It does not write to the trajectory file. (The `.molwatch.log`
  writer lives in molbuilder's generators, not in the watch app.)
- It does not do downstream analysis (RMSD, principal axes, dipole
  moment time series, hbond detection, etc.). Parsers extract what
  the engine wrote; analysis is the user's job in their own tools.
- It does not warn about non-converged geometry / SCF; that's the job
  of the validator on the build side, not the viewer on the watch
  side.

### Web routes (post-merge)

```
GET  /                              # tabbed UI shell (Build + Watch)
GET  /api/watch/formats             # parser_summary() — list of
                                    # registered parsers with name,
                                    # label, hint (drives the
                                    # "supported formats" UI)
POST /api/watch/load                # body: {"path": "/abs/path"} OR
                                    # multipart upload.
                                    # → {ok, path, mtime, format, label,
                                    #    data: {<parser dict>},
                                    #    uploaded: bool}
GET  /api/watch/data                # poll for changes:
                                    # → {changed: false}
                                    # | {changed: true, data: {...},
                                    #    mtime: ..., format: ...}
```

Build-side routes live under `/api/build/*` (see Surfaces above). The
two route groups are registered as separate `flask.Blueprint`s; the
prefix is the namespace, no custom routing.

---

## Design principles

These are load-bearing. Don't violate without updating this document.

### 1. The dataclass is the lingua franca

Every builder yields a `Structure`. Every generator consumes a
`Structure` + a `Config`. Every parser returns a `Trajectory` (a
thin wrapper over `List[Frame]` plus `source_format` + optional
shared `lattice`). Every validator returns `List[Issue]`. Field metadata — label, type, default,
range, validator, UI hint — lives on the dataclass field, **not** in
parallel registries in the CLI or web layers.

A previous custom registry framework was deleted because dataclass
introspection (plus click for CLI) is the right tool. Three places
declaring the same field metadata (dataclass + argparse + HTML form)
is how silent drift happens. CLI and HTML form must be *generated*
from the dataclass, not maintained in lockstep with it.

### 2. CLI scripts are small, focused, and composable

Each subcommand does one job. They chain through files / stdin / stdout
in classic Unix style. Treat `-` as stdin where it makes sense:

```bash
molbuilder dna ATGC | molbuilder fdf - out.fdf
molbuilder watch tail run.molwatch.log | jq '.energy'
molbuilder dna ATGC | molbuilder validate - --cell 30,30,30
```

Machine-consumable subcommands emit JSON / NDJSON on stdout. Human
subcommands emit text. Status / progress / warnings always go to
stderr so they don't pollute the pipe.

### 3. The web UI is a portal, not a separate product

The UI calls the same Python API the CLI calls. It contains no logic
that isn't trivially also exposed elsewhere. Tabs (Build / Watch)
share the 3Dmol viewer, style controls, atom rendering, and CSS. The
Build tab's "Generate FDF / script" flow drops a "Watch this run"
affordance that pre-fills the Watch tab with the predicted output
path so the user moves naturally from one phase to the next.

UI redesign mandate: concise, easy, visually fluent. Single layout
shell, two views, no duplicated chrome.

### 4. Generated outputs must be both syntactically correct AND scientifically defensible

An FDF that SIESTA accepts but silently produces wrong physics is a
bug, not a feature. A PySCF script that runs but converges to a
broken-symmetry saddle for an open-shell system is a bug.

Code review for this project must include target-platform correctness
checks: are the keywords real? Are the values in scientifically
defensible ranges? Are open-shell / charged / periodic special cases
handled? See "Scientific correctness" below for the validation
requirements and the known gap list.

### 5. Generated outputs are tunable by manual editing

Generated scripts use plain object APIs (no convenience wrappers
that hide what's happening), keep all SIESTA / PySCF configuration
in scope at the natural location, and provide post-processing hook
placeholders for common follow-ups (Mulliken population, dipole
moment, BandLines, PDOS).

Verbose-comments mode (default ON) inlines tuning hints next to
every parameter — the generated FDF / .py is meant to be readable
as a tutorial. Section headers are mandatory; they make `Ctrl-F`
in the file work for someone unfamiliar with the platform.

### 6. Pre-emission geometry validation

Before any FDF or PySCF script is written, run a scientific sanity
pass on the structure + cell. Errors stop emission; warnings print
to stderr but proceed. Validators are pure functions reading field
metadata; they never call out to the engine. See "Validation pass"
below for the check list.

### 7. Generated artifacts are self-contained

The generated PySCF script does **not** import molbuilder at runtime.
A user can `scp` the .py to a cluster that has only `pyscf +
geometric` installed and run it. The molwatch emitter helper class
is pasted verbatim into the generated script via
`inspect.getsource(MolwatchEmitter)` from
:mod:`molbuilder.trajectory_log.emitter` (Phase 4, `3bd5c32`).
The class is the source of truth -- the inline text in
`pyscf/input.py:_emit_molwatch_emitter` is a single
`inspect.getsource` call, not a duplicate string -- and gets unit-
tested directly via `tests/test_molwatch_emitter.py`.  The "generated
artifact has no extra imports" invariant is preserved.

### 8. Don't reinvent wheels

For CLI parsing → click. For routing → Flask Blueprints. For numerical
work → NumPy. For trajectory I/O on legacy formats not covered by our
parsers → ASE may be considered, but only when the maintenance cost of
our own parser exceeds adopting an external dep (revisit if it ever
does). For form rendering → vanilla HTML + the existing 3Dmol viewer
machinery; no SPA framework. For validation → plain functions over
field metadata. Adding a dependency is a decision, not a default; each
new third-party dep needs a one-line justification in the decisions
log below.

---

## Anti-patterns we refuse

These have been considered and rejected; do not reintroduce them.

- **Reverse imports** (L1 importing from L2, L2 importing from L3).
  The package will calcify the way the prior custom registry did.
- **Custom CLI / registry / dispatch frameworks** on top of click,
  argparse, or anything else. A previous version had one; it was
  deleted. Stay deleted.
- **Builder-pattern wrappers around dataclasses**
  (`StructureBuilder().with_atoms(...).build()`). Plain dataclasses
  plus freestanding `build_*` functions stay.
- **Generic plugin discovery via setuptools entry points.** We have
  a small, known set of formats and backends; an explicit
  `PARSERS = [...]` list is easier to read and to audit.
- **Parallel field-metadata tables** in CLI or web layers
  (`FIELDS = {...}` dicts that mirror dataclass fields). Read from
  `dataclasses.fields()` instead.
- **Sync-from-async wrappers in the generated script.** The generated
  PySCF script is a plain top-to-bottom Python file; no event loops,
  no coroutines, no observability framework imports.
- **A separate config file format** (YAML / TOML / INI) for SIESTA or
  PySCF parameters. The user edits the generated `.fdf` / `.py`
  directly; that's the contract.

---

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-30 | Merge `Qing-LAB/molwatch` into `Qing-LAB/molbuilder`. molwatch repo archived after merge stabilizes. | Already coupled by file format spec, web stack, and author. Single repo removes drift surface. |
| 2026-05-01 | Top-level package name remains `molbuilder`. | Established name; "watch" is a verb on it. |
| 2026-05-01 | Keep a `molwatch` console-script shim in `pyproject.toml` post-merge. | Zero cost, real friction saved for existing users / scripts. |
| 2026-05-01 | argparse → click conversion is part of Phase 5 (CLI rework), not Phase 1. | Touching CLI plumbing while moving files makes diffs harder to review. Click is the long-term answer; the short-term concession is to land new merge-driven subcommands as argparse for now. |
| 2026-05-01 | History preservation via `git subtree add`. | Preserves molwatch's commit history with `git log --follow` working. |
| 2026-05-01 | `_MolwatchEmitter` extracted to a real Python file, pasted via `inspect.getsource()`. NOT runtime-imported from generated script. | Keeps generated script self-contained for cluster use; emitter still IDE-checkable and unit-testable. |
| 2026-05-01 | `molbuilder watch parse` / `molbuilder watch tail` are the resolution of issue #81. | Same JSON-over-stdout shape the original handoff was gesturing at, under the unified CLI. |
| 2026-05-01 | 3DNA backend added as `molbuilder/builders/backends/_threedna.py`; auto-detect order becomes `threedna > amber > rdkit`. | True canonical helix; only thing the existing backends do not provide. |
| 2026-05-01 | Parser output type is `Frame` (Structure + per-step physics), **not** `Structure` directly. Parsers yield `Iterator[Frame]`. | `Structure` is geometry + PDB metadata; parser dicts carry energies / forces / lattice / scf_history that have no slot on `Structure`. Promoting parsers to yield `Structure` would silently drop everything except positions. A sibling `Frame` keeps both paths clean. |
| 2026-05-01 | Post-merge web routes are namespaced `/api/build/*` and `/api/watch/*` via Flask Blueprints. | Both pre-merge apps define `POST /api/load` for unrelated payloads (XYZ/PDB → Structure vs trajectory file → frames). Blueprints are Flask's native URL-prefix mechanism; no custom router needed. |
| 2026-05-01 | Phase 1 / commit 3 landed: `siesta`, `pyscf`, `molwatch_log` promoted from single files to subpackages with re-exporting `__init__.py` (commit `e34ede7`). | Re-exports keep external imports stable; subpackages create slots for the parser modules arriving in commit 4. |
| 2026-05-01 | Phase 2: introduce a minimal `Trajectory(source_format, frames, lattice)` wrapper alongside `Frame`; parsers return `Trajectory` (not `Iterator[Frame]` directly). | `source_format` and the shared lattice are file-level metadata that don't fit on any single Frame; the molwatch unified-log parser specifically needs `source_format` from the file's `# engine:` header rather than from the parser class name. The "Trajectory deferred to Phase 3" open question resolves to "yes, minimally, in Phase 2." Phase 3 may grow it (analysis methods, on-disk serialization). |
| 2026-05-02 | Phase 2.5: 3DNA backend's detection chain is `in-tree > $X3DNA > PATH` (not "all three preconditions must hold simultaneously"). | The pre-implementation spec said `is_available()` must require `fiber` on PATH AND `$X3DNA` env var AND the config dir, all simultaneously. In practice the user wanted "expand the tarball at the repo root and it just works"; that's a one-step path that doesn't require shell config. The chain accommodates that *and* the canonical env-var install *and* a PATH-only install, with the same completeness check (`bin/fiber` + `config/`) applied to each candidate root. |
| 2026-05-02 | Phase 2.6: validators read field metadata via `dataclasses.field(metadata={...})` rather than via a parallel registry. | Realises Principle #1 (dataclass is the source of truth) for the first time -- adding a new validated config field is a one-line metadata change, not a multi-file plumbing change. Metadata schema (per-field): `label`, `unit`, `range`, `tier` (basic/advanced), `help`, optional `validate=callable`. The CLI / web layers can introspect the same metadata when they grow form-generation. |
| 2026-05-02 | Phase 2.6 surfaced and fixed a peptide-builder artifact: `AddHs(addCoords=True)` (RDKit) and `OBMol.AddHydrogens` (OpenBabel) occasionally leave Hs at their heavy-atom anchor coordinates instead of computing displaced positions. | The validator's min-distance check fired on `build_peptide("AC")` with a 0 Å pair (CA + H2 both at origin). Real bug, not a false positive. Fixed via `_drop_overlapping_hydrogens` post-pass that removes Hs sitting < 0.05 Å from any other atom -- a safe heuristic since real H positions are always > 0.9 Å from their anchors. The fix lives in `molbuilder/peptide.py` because both protonation paths exhibit the artifact. |
| 2026-05-01 | Configs are L1 nouns: `SiestaConfig` and `PySCFConfig` move out of `siesta/input.py` / `pyscf/input.py` (where they currently live next to the generators) into `config/siesta.py` and `config/pyscf.py`. | Configs are pure data carrying field metadata that the CLI, web form, and validators all introspect. Generators are L2 verbs that operate on configs. Keeping configs L1 lets validation and form-schema generation read them without dragging in the file-emission code. Re-exports preserve external imports. |
| 2026-05-01 | `parsers/` is a flat package (one `<format>.py` per parser sibling-to-`base.py`), not a per-format split where each engine subpackage owns its own parser. | The per-format split (`siesta/parser.py` next to `siesta/input.py`) co-locates modules that share no imports, no state, and run in opposite directions of the data flow. A flat `parsers/` directory has higher internal cohesion. |
| 2026-05-01 | `backends/` moves under `builders/`. | Backends only serve builders; they have no callers outside the build path and shouldn't sit at the top level. |
| 2026-05-01 | `molwatch_log/` renamed to `trajectory_log/` post-merge. | The format isn't molwatch-specific anymore. Renaming with re-exports is cheaper now than later, when users have more code referring to the old name. The on-disk `.molwatch.log` extension stays — that's a user-facing filename convention, not a module name. |
| 2026-05-01 | CLI uses click + a ~30-line dataclass→click-options bridge. We do not write our own argument parser, registry, or coercion layer on top of click. | Don't reinvent wheels; don't reintroduce the custom CLI framework that was previously deleted. The bridge reads `field.metadata` and emits `click.option` decorators; click handles the rest. |
| 2026-05-01 | Web routing uses Flask Blueprints, not a hand-rolled router. | Blueprints are Flask's native URL-prefix primitive; we don't reinvent it. |
| 2026-05-01 | Do not introduce ASE-backed parsers in Phase 1 or 2. | The existing molwatch parsers work and are well-understood; switching would change behavior subtly. Revisit if maintenance cost grows; until then, keep the parsers we have. |
| 2026-05-03 | Watch viewer stays on browser-driven mtime polling; SSE / push-style alternatives are not pursued. | The server has authoritative knowledge of the file state.  An SSE swap would only pay off if paired with sub-second change detection (background watcher / inotify), and SCF runs are slow enough that the 15s poll latency is rarely the bottleneck.  Polling is the right shape for "server tells browser what's available." |
| 2026-05-09 | Modify-tab pair-mode electrode placement defaults to **anchorless**: slabs at `z = ±gap/2` around the world origin (legacy anchor-pair-midpoint mode opt-in via two-atom selection). | Decouples slab placement from molecule centring; the user controls junction geometry via the Geom + Pose subtabs alone.  Realises the user's mental model "slabs are crystallographic, the molecule fits between them" with no per-atom dependence. |
| 2026-05-09 | Modify-tab Undo is **slab-only**; non-slab ops are committed.  Snapshot pushed on a successful response (failed ops do not consume an undo slot). | Matches the original "experiment with electrode parameters and roll back" intent.  General undo across delete / rotate / translate would grow the JS state model materially for a feature few users have asked for; revisit if needed. |
| 2026-05-09 | `GET /api/modify/meta` is the single source of truth for the FCC element + plane dropdowns; HTML must not duplicate the lists. | Realises Principle #1 (dataclass / Python-tuple = source of truth) for the Modify tab's enums.  Adding a metal in `molbuilder.modify.SUPPORTED_FCC_ELEMENTS` reaches the UI automatically. |
| 2026-05-10 | **Job-layout v1** (`docs/protocols/job-layout.md`) codifies the on-disk shape of a molbuilder run: one directory, one basename, named files.  Watch resolves a **run directory** via a documented discovery chain (`*.molwatch.log` → `*.fdf` → `*.py` → fallbacks). | Lets the user point Watch at a directory instead of a specific output file.  Cross-stage continuation (SIESTA `.XV` / `.DM` / `.CG`, PySCF `.chk`) works automatically because the basename stays identical across staged runs. |
| 2026-05-10 | Multiple `*.molwatch.log` files in a run directory are **merged** into one trajectory with stage-boundary markers; live polling pins to the newest log; older stages are static. | Realises the staged-relaxation workflow (coarse → medium → tight) end-to-end.  Polling re-runs the merge over the FULL log set (per-file mtime-keyed cache prevents re-parsing static stages). |
| 2026-05-10 | When the Build-tab "Relaxation stage" preset is non-Custom, the SIESTA + PySCF generators auto-suffix the `.molwatch.log` filename as `<basename>-stage<N>.molwatch.log`.  Basename itself stays unsuffixed so restart files transfer. | Removes the manual-rename step from the staged-relaxation flow.  Suffix rule lives in `molbuilder.trajectory_log.format.molwatch_log_basename` -- ONE source for both engines, no drift. |
| 2026-05-10 | `Trajectory` stays a thin `(source_format, frames, lattice)` wrapper.  Per-trajectory analysis (RMSD, principal axes, dipole, radius of gyration) lives as free functions under `molbuilder/analysis/` if and when a consuming workflow arrives — NOT as methods on `Trajectory`. | The Phase-2 minimum has proven sufficient through v1.0 (Watch, CLI, all parsers).  Adding methods would couple analysis to the parser-output shape and bloat the L1 surface that every consumer pays for, in exchange for ergonomics no caller has asked for.  Pull-on-demand functions stay testable independently and don't pollute the wrapper. |
| 2026-05-10 | CP2K / ORCA generator + parser deferred indefinitely; the per-engine subpackage layout (`molbuilder/<engine>/input.py` + `parsers/<engine>.py`) is already in place and will be reactivated when a real workflow asks for it. | v1.0 ships covering SIESTA + PySCF, the two engines actually used.  Adding speculative CP2K / ORCA support before a use case means committing to test + maintenance burden for code with no caller — exactly the trap Principle-#8 ("don't reinvent wheels") points at one layer up.  The layout split costs nothing to keep dormant. |
| 2026-05-11 | Post-relax frequencies + RRHO thermochemistry are an **opt-in** PySCF script feature (`cfg.compute_frequencies`).  Output is a separate plain-text `<job>.thermo.txt`; no on-disk format change to the existing molwatch log.  The block runs at the converged `mf` at `mol_eq` (no extra SCF) and is wrapped in try/except so a Hessian failure doesn't lose the converged energy or `<job>_optimized.xyz`. | One Hessian is 5-15x the cost of a single SCF — making it default-on would hurt the relaxation workflow.  A separate `.thermo.txt` (instead of mutating the molwatch log) keeps the streaming/append-only contract on the molwatch side intact.  RRHO (not quasi-RRHO) is the PySCF-bundled path; quasi-RRHO is a documented follow-up if low-frequency mode artifacts become a problem in practice. |
| 2026-05-11 | Build-tab form is **schema-driven** end-to-end.  `_shared.py::dataclass_to_form_schema(cls, prefix)` serialises every `SiestaConfig` / `PySCFConfig` field that carries a `"section"` metadata key; `GET /api/build/schema/{siesta,pyscf}` streams it; `web/static/lib/form-schema.js::renderForm()` builds the DOM and `collectForm()` reads values back.  Section ordering is controlled by an optional class-level `_form_section_order` tuple so dataclass field-declaration order doesn't dictate the UI layout. | Closes the last Principle-#1 anti-pattern: adding / removing a Build form field is now a one-line dataclass change.  No HTML touched, no JS form-collect updated, no `FORM_IDS` list maintained.  Field-level metadata (label, unit, range, choices, pattern, tier, help, null_label) drives both the validator AND the UI — the dataclass is the only place these constraints are written.  Pin-tests in `tests/test_web.py` lock the schema shape so a stray field-reorder doesn't silently rearrange the UI. |
| 2026-05-11 | **Transport tab — v1 scope locked** (no code yet).  Target: tier T4 — emit electrode `.fdf` + per-bias device `.fdf` + TBtrans `.fdf` + `run.sh`, parse TBtrans output, plot T(E), I(V), dI/dV, with self-consistent bias sweep.  Publication-quality overlay: rich inline justifications + citation keys in every emitted file, curated bibliography at `docs/transport/references.bib` (each entry verified by the author).  Modify tags atoms with controlled vocab `{electrode_left, electrode_right}` (full slab; Transport later subdivides into lead + buffer per `n_principal_layers`); tags survive in-memory + sessionStorage but NOT round-trip through `.xyz` (Transport falls back to element-heuristic for tag-less loaded structures).  No always-emit `methods.md`; the rich inline comments + bibliography are the single source of truth for what was run, future `molbuilder transport methods <rundir>` extractor distils to Markdown on demand.  TBtrans output parsed via `sisl >= 0.14` (declared in `pyproject.toml` ahead of code so the runtime is ready when the feature lands). | Q1-Q4 of the Transport design conversation (this conversation, 2026-05-11).  Q5 (electrode-calc workflow shape) and Q6 (structure-input sources) still open; will be resolved at Transport kickoff.  Raman pushed ahead of Transport because it builds directly on the `compute_frequencies` plumbing already shipped (v1.1.0) and the implementation is ~1-2 weeks vs Transport's 4-6 weeks. |
| 2026-05-14 | **Projects hierarchy is `<project>/<topic>/<structure>/`** (topic-first, not structure-first).  Six canonical topics hard-coded in `molbuilder.projects.CANONICAL_TOPICS`; non-canonical names rejected at path-construction time.  Innermost `<structure>/` is a flat job-layout-v1 directory (no subdirs inside).  `molbuilder run <script>` emits a sibling `<basename>.run.sh` shell wrapper that activates the routed conda env and execs the tool; molbuilder does NOT manage the resulting process — monitoring is the existing Watch tab pointed at the run directory.  Cross-stage input chaining (e.g. spectrum reading from a prior optimization) is done at generation time by inlining coordinates into the new script, not via FS symlinks or runtime path resolution — keeps each calc dir self-contained and tool-agnostic.  See `docs/protocols/job-layout.md` § "Project-tree organisation" and § "Run wrapper". | Topic-first lets "compare the same analysis across structures" (Raman across the gold-thiol variants) be a simple `ls projects/<proj>/spectrum/`, which matches the most common review intuition; structure-first would scatter that across multiple subtrees.  Hard-coded vocab prevents the fragmentation of running "raman", "Raman", "raman-spectra", "spectra-raman" subtrees across teammates.  Shell wrapper instead of direct subprocess-spawn keeps molbuilder out of the "background process manager" business — that role is owned by the user's shell / cluster scheduler, and the wrapper composes with both transparently. |
| 2026-05-15 | **SCHEMA_VERSION bump to 2 + field-name trim + legacy alias retirement.**  Field names: ``eigenvector_canonical_cart_mass_weighted`` → ``eigenvector_canonical``; ``eigenvector_display_max_abs_unit`` → ``eigenvector_display`` (the normalisation convention each carries is documented in the ``ModeData`` docstring so the field name doesn't have to repeat it).  Dropped the v1-era ``eigenvector_free`` from the JSON wire format and from the ``ModeData`` Python attribute -- it was only consumed by ``viewer.js``, which now reads ``eigenvector_display`` directly.  ``ModeData.from_dict`` and ``parsers.spectra_json`` continue to accept v1 documents on read (``_READABLE_SCHEMA_VERSIONS = {1, 2}``); v1's single ``eigenvector_free`` field is copied into both canonical and display slots (best-effort -- the true canonical normalisation isn't recoverable after the fact).  Doc updates: ``docs/tabs/spectra/spec.md`` describes both forms with conventions and the v1 → v2 transition; ``test_types.py::test_schema_version_pinned`` updated to ``SCHEMA_VERSION == 2``. | Long field names cluttered the JSON and the call sites without adding information beyond what the class docstring already carries -- the trim makes the wire format readable while keeping the convention precise.  Dropping the legacy alias removes one consistency surface to maintain (was: keep three names in sync; now: two).  Keeping v1 readable on input is correct: users have saved spectra files in the wild (e.g., ``projects/tunneling/BDT_Raman/spectra.spectra.json``) that should still load. |
| 2026-05-15 | **Spectra script scientific-correctness pass.**  Two real-numbers bugs and one schema clarification.  (1) Imaginary-mode wavenumber handling was inconsistent across PySCF versions: the all-free Hessian path either lost the magnitude (purely-imaginary complex `0+500j` → `FREQ_CM1=0`) or mis-flagged the mode (signed real `-500` → `HAS_IMAG=False`).  Introduced a ``_signed_wavenumber(w)`` helper in the generated script that normalises both shapes to "signed real cm⁻¹"; ``HAS_IMAG = (f < 0)`` then matches the partial-Hessian path's convention.  (2) Normal-mode normalisation differed between the two Hessian paths: the all-free path stored PySCF's mass-weighted ``norm_mode`` (canonical convention) while the partial path normalised to ``max(|L|)=1`` for UI animation, then used the SAME array for the Raman projection -- meaning ``raman_activity_a4_amu`` was in different units depending on which path ran.  Now both paths compute and keep ``NORM_MODES_CANONICAL`` (mass-weighted, ``Σ_k m_k |L_k|² = 1``, used for the Placzek 45a²+7γ² projection) and ``NORM_MODES_DISPLAY`` (max-abs=1, used for animation and the fixed-amplitude ES probe).  The output JSON exposes both under explicit labels: ``eigenvector_canonical_cart_mass_weighted`` (science) and ``eigenvector_display_max_abs_unit`` (UI), with ``eigenvector_free`` retained as a legacy alias of the display form for the current 3Dmol viewer until SCHEMA_VERSION bumps. | (1) PySCF's freq_wavenumber shape has shifted across releases without a documented stable convention; the script needs to normalise both at the boundary.  (2) Mixing "science" and "UI" normalisations under one field name produced answers that quietly disagreed with literature and with the all-free path -- exactly the kind of bug that doesn't trip a test but lands in a publication.  Storing both forms with self-describing names makes the consumer's choice explicit; the new comments in the Hessian and Raman blocks state which form goes with which use case and why.  See ``pyscf_script._emit_hessian_block`` (rewritten) and ``pyscf_script._emit_raman_block`` (projection now reads NORM_MODES_CANONICAL not the JSON alias). |
| 2026-05-20 | **/watch retirement + post-removal cleanup.**  Phase 1: ``lib/trajectory/core.js`` adopted the ``_on()``/``_cleanups`` dispose pattern from ``lib/spectra/core.js`` (16 element-listener registrations rerouted; dispose walks cleanups in reverse before per-resource teardowns).  Phase 2: deleted ``templates/watch.html``, ``static/watch/{viewer,page}.js``, the ``/watch`` route + ``cmd_watch_serve`` + ``warn_if_remote``, the ``WATCH_PATH_KEY`` sessionStorage block, the loader-bar + applyHandoff wiring inside ``lib/trajectory/core.js``, the ``.loader-bar`` + ``.workflow-*`` CSS, and the Watch tab from ``_app_header.html``.  KEPT: ``_trajectory_inspector.html`` (served by ``GET /partials/trajectory-inspector``), all ``/api/watch/*`` endpoints (consumed by the /results trajectory inspector), ``molbuilder watch parse``/``tail`` CLI utilities.  Phase 3: deleted duplicate ``body { ... }`` rules from per-tab stylesheets (page-shell.css is canonical); deleted duplicate ``.card { ... }`` from the build-tab stylesheet (modify + spectra keep their intentional overrides); per-page ``header { }`` rules stay (intentionally divergent padding/border per page).  Phase 4: added the missing ``form-schema.css`` link to ``templates/index.html``.  Phase 5: ``molbuilder/runwrap.py`` SIESTA-MPI wrapper now emits ``export OMP_NUM_THREADS=1`` + ``MKL_NUM_THREADS=1`` + ``OPENBLAS_NUM_THREADS=1`` before ``mpirun`` so each MPI rank doesn't oversubscribe with BLAS threads (single-process SIESTA + PySCF wrappers untouched -- they want the BLAS threading).  Post-removal audit fixes (six review groups): #1 rewired ``_run_watch_serve_entrypoint`` to alias ``molbuilder serve`` instead of the deleted ``molbuilder watch serve``; #2 added ``AbortController`` to the four inner fetches in trajectory + spectra cores (loadByPath, pollOnce, generateScript) -- each abort()s its previous in-flight + dispose() aborts all in-flight; #3 repointed ten ``/watch``-driving tests in ``test_modify_e2e.py`` to ``/results``; #4-#10 swept stale ``/watch``-era prose across templates + docstrings; #5 exposed ``handle.load(path)`` on the spectra core (parity with trajectory); #13 harmonised dispose's 3Dmol cleanup to ``viewer.clear()`` across both inspectors; #15-#16 added five static tests in ``TestInspectorErrorRendering`` + two Playwright behavioural tests in ``TestInspectorErrorCardRuntime`` pinning that adapter ``.catch()`` handlers render an ``.inspector-card.error-card`` AND that user-triggered ``AbortError``s do NOT trigger the error card (so a rapid file-switch doesn't show a spurious error).  Deferred (#11 rename watch.py → api_trajectory.py, #12 extract ``_on()`` into a shared module, #14 extract ``.ctab`` wiring) -- each is high-churn-low-value at current scale; the duplication is ~20 lines and lives in two modules that are each their own entry point, not shared libraries.  714 tests pass after the sweep; 1900 across the full suite. | Two architectural milestones land: (a) /results is now the single canonical inspection surface, fully replacing /watch -- one URL, one inspector dispatch via the Inspector Registry, no parallel /watch.html maintenance burden.  (b) trajectory and spectra cores have the same dispose contract structure (``_cleanups`` + per-resource teardowns + AbortControllers), so the next inspector author has a single reference shape rather than two divergent ones to choose from.  The post-removal audit found and fixed real follow-up bugs the migration could have left behind (dead entrypoint shim that would have crashed on legacy ``molwatch`` invocations; race-prone fetches; stale docs that lie to the next reader); doing the audit AS a follow-up phase rather than mixed into the removal kept each diff focused and reviewable.  Per the user's policy "do the fucking correct thing": the removal didn't paper over /watch with redirects -- it actually deleted the route, the assets, the CSS, and every test that pinned the removed surface; the API that /results genuinely needs stayed; the docstrings tell the new story; the entrypoint scripts still work for legacy callers via an alias.  Total cleanup: 11 phases / sub-phases / audit findings landed across 2026-05-18 → 2026-05-20. |
| 2026-05-18 | **Spectra inspector lift complete (migration step 2.2–2.5 of task #58); /spectra is generate-only.**  Mirrors the trajectory-inspector lift (steps 1.1–1.6) for the spectra inspector.  (2.2) ``static/spectra/viewer.js`` body lifted into ``static/lib/spectra/core.js`` wrapped in ``mountInspector(rootEl, opts)``; the IIFE exports ``window.molbuilder.spectraInspector.mount`` and returns a ``{dispose()}`` handle that stops the live-watch poller (``state.watchTimer``), cancels the mode-animation rAF loop (``state.animTimer``), tears down the 3Dmol mode viewer, and purges the Plotly spectrum chart.  All ``document.getElementById``/``querySelectorAll`` lookups rewired through a rootEl-scoped ``$ = (id) => rootEl.querySelector("#" + id)``; two call sites that escaped to ``document`` (the ES-field disable loop + the ``.modes-table .es-col`` toggle) now go through ``$``/``rootEl.querySelectorAll`` so the inspector mounts cleanly on either /spectra (``document``) or /results (``#inspector-host``-rooted partial container).  init() splits into ``hasGenerateSide`` and ``hasInspectSide`` blocks gated on the presence of ``spectra-form-container``/``watch-path``, so the same module is the single source of truth for both consumers without conditional listeners leaking across.  (2.3) ``results.html`` loads ``lib/spectra/core.js`` before ``lib/inspectors/spectra.js`` (script-order load-bearing -- the inspector adapter reads the mount API at module-eval time).  (2.4) ``lib/inspectors/spectra.js`` rewritten from a placeholder card into a real adapter that fetches ``GET /partials/spectra-inspector``, assigns the response HTML to ``host.innerHTML`` (same trust-boundary justification as the trajectory adapter: same-origin autoescaped Jinja render with no user input), calls ``api.mount(host, {file})``, and chains the inner handle's ``dispose()`` into the registry's tear-down.  ``AbortController`` cancels in-flight fetches when the user switches files mid-mount; ``_renderError`` falls back to the same textContent+createElement error card the trajectory adapter uses.  (2.5) ``{% include "_spectra_inspector.html" %}`` dropped from ``spectra.html``; /spectra is now generate-only (form + Methods preview + script preview).  ``static/spectra/page.js`` deleted -- its load-from-selection-button + workspace-indicator wiring concerned the inspect-side ids that no longer live on /spectra; ``spectra/viewer.js`` reduced to a 47-line bootstrap that calls ``mount(document)`` on DOMContentLoaded.  Tests: 44 in ``tests/spectra/test_blueprint.py`` pass after repointing inspect-side id pins at ``/partials/spectra-inspector`` and symbol pins at ``/static/lib/spectra/core.js`` (the old pins at ``/static/spectra/viewer.js`` would now be checking a 47-line bootstrap stub); new ``test_page_is_generate_only_after_step_2_5`` pins the 20 inspect-side ids that MUST NOT appear on /spectra so a regression bringing back the include lands as a clear failure.  ``tests/test_xss_audit.py`` ALLOWLIST entries moved from ``spectra/viewer.js`` to ``lib/spectra/core.js`` (same patterns, code lifted); ``lib/inspectors/spectra.js`` added to the ALLOWLIST for its single trusted ``host.innerHTML = partialHtml`` site (matches the trajectory adapter's allowlist line); ``lib/inspectors/spectra.js`` dropped from the PURE_FILES list since it now has the trusted-innerHTML site.  150 XSS-audit tests + 44 spectra-blueprint tests + 244 results+xss+no-inline+no-js-errors tests all pass. | Closes the merge half of task #58: /results is now the single canonical "look at the output of a calculation" surface, and /spectra is the single canonical "design a spectra calculation" surface.  Two architectural wins fall out: (a) the spectra inspector code lives in exactly one place (lib/spectra/core.js) instead of being conceptually duplicated across /spectra and /results, so a bug fix lands once and both consumers pick it up; (b) the /results inspector ecosystem now has TWO live inspectors (trajectory + spectra) plus the two simpler ones (structure + source), which exercises the Inspector Registry's partial-fetch + dispose contract against real complexity, not just placeholders.  The hasGenerateSide/hasInspectSide gates inside init() are the textbook seam for "same module, different consumers": both consumers' DOM shapes are recognised at mount time and the module wires only what's present, no per-consumer fork.  Per the user's "do the fucking right thing. this is serious programming" mandate from earlier this session: the lift didn't paper over the old code, it actually moved + cleaned up the dead bootstrap, deleted the obsolete spectra/page.js, repointed the tests at the new homes, and treated every behavioural assertion as a contract worth re-pinning at its new location rather than a test to delete-and-forget. |
| 2026-05-17 | **Round-3 fresh-eyes review: 4 P0 fixes; mount-handle contract + XSS safety + opts.file land.**  Third review pass with a sharper lens ("could I write Stage 1C against this code TODAY?") surfaced 4 real bugs the prior rounds missed.  (1) XSS via ``innerHTML`` string-concat with ``_basename(file)`` in both placeholder inspectors (``trajectory.js`` + ``spectra.js``).  In practice filename validation upstream blocks metachars, but the inspectors were doing zero defense in depth -- a shell-cp'd file with `<` in its name would inject HTML.  Rewrote both placeholders with textContent + createElement DOM construction.  (2) ``mountInspector(rootEl)`` returned ``undefined`` -- Stage 1C/1D had nothing to ``dispose()``.  Now returns ``{dispose(), load(path)}`` handle.  (3) Timer + listener leaks on rapid mount: ``state.pollTimer`` (15s poll), ``state.playTimer`` (frame playback), and ``window.addEventListener("resize", _onResize)`` all created inside mountInspector but never torn down.  Stage 1D would have leaked them on every file selection on /results.  ``dispose()`` now stops both timers + removes the resize listener + tears down 3Dmol bookkeeping (removeAllShapes/Labels/Models).  (4) ``mountInspector`` took no ``opts`` -- Stage 1D needs to pass an initial file to load.  New signature ``mountInspector(rootEl, opts={file?})``; if ``opts.file`` set, auto-loads via ``loadByPath(opts.file)``.  Plus: IIFE refactored to ``(function (root) {})(window)`` so the new ``root.molbuilder.trajectoryInspector = {mount: mountInspector}`` export lands cleanly -- Stage 1C's registry-side trajectory inspector now has a clean delegation point (``window.molbuilder.trajectoryInspector.mount(host, {file, ctx})``) and doesn't need to fork the inspector code.  Tests: 8 new pins in ``test_trajectory_inspector_partial.py`` (``TestMountInspectorHandle`` + ``TestInspectorPlaceholderNoInnerHTMLInterp``) covering signature + handle contract + dispose cleanups + export + no-innerHTML-interp.  Plus one pre-existing test signature pin updated from strict-arity to ``mountInspector(rootEl\b`` to accept the extra opts arg.  27 trajectory-partial + 33 results-blueprint + 1 web tests all pass after the changes. | Earlier rounds caught the LAYOUT issues (root-scoping, partial extraction, guard pattern) but missed the CONTRACT issues (no return handle, no opts, XSS).  Reading each file with the question "could I write the next stage against this?" found gaps a generic code review never asks about.  The 4 P0 fixes mean Stage 1C is now a genuinely mechanical move (``lib/inspectors/trajectory.js``'s mount() becomes a 5-line clone-and-delegate to ``window.molbuilder.trajectoryInspector.mount``); without these, Stage 1C would have had to invent new contracts under time pressure and likely introduced inconsistencies.  Per the user's "we need a clean and correct design and implementation from bottom up": the foundation is now demonstrably ready -- the contract is explicit, the lifecycle is owned end-to-end, and security defense-in-depth replaces "safe by upstream convention". |
| 2026-05-17 | **Use-case-driven review pass: NPE guards on watch/viewer.js + 6 Stage-1D gaps captured.**  Second-round review evaluated the code in the true context of the next step (Stage 1C/1D mounting the trajectory inspector inside /results' host).  Surfaced one real bug: my Stage 1B refactor left 8 unguarded ``$doc("X").<member>`` dereferences inside ``mountInspector(rootEl)``.  Today's only caller passes ``document``, so /watch works.  But Stage 1D's planned call site ``mountInspector(panel)`` against /results' inspector-host -- which has no loader bar -- would throw NPE at ``$doc("path-input").value`` etc.  Fix: every page-level $doc lookup now goes through a captured-and-guarded local (``const _el = $doc("id"); if (_el) _el.foo``).  Plus ``setStatus`` short-circuits when the status banner is absent.  New test in ``test_trajectory_inspector_partial.py::TestViewerJsRootScoping::test_no_unguarded_dollar_doc_dereference`` blocks regressions.  ``test_web.py::test_watch_viewer_js_honours_path_url_param`` relaxed to semantic ingredients (URLSearchParams + applyHandoff function name + path-input id presence) instead of pinning a specific JS expression that broke twice in 24 hours due to refactor churn.  Captured 6 Stage-1D readiness gaps in REVIEW_FINDINGS.md: partial-cloning mechanism (recommend ``<template>`` clone), ``opts.file`` parameter for inspector mount, Plotly script tag on /results, trajectory CSS sharing between /watch and /results, dispatcher de-dupe on same-file no-op events, 3Dmol+Plotly+timer disposal cleanup.  Each gap has a concrete recommended fix sized for a focused commit. | The user's mandate was "evaluate the code in the true context of our next step" -- generic static review only catches generic code smells; use-case-driven review catches the bugs that surface when call sites change.  The NPE-guard pattern + the documented gaps mean Stage 1C lands against a code base that won't surprise the user at runtime, and Stage 1D has a clear punch list of "five small concrete changes" rather than "figure out what's missing as you go".  Per the user's words: "we have a solid foundation for next step" -- the foundation is now demonstrably ready (existing tests still pass; new invariant test blocks the class of bug from creeping back; explicit gap inventory means no surprises). |
| 2026-05-17 | **Static-review cleanup pass: 5 P1/P2 fixes before the next inspector lands.**  Holistic review of the registry foundation surfaced six real issues (out of 12 candidate findings from an Explore-agent pass + my verification); five fixed, one rejected as misdiagnosed.  (1) Mount-context construction moved OUT of ``registry.mount()`` and INTO the dispatcher via a new ``createDefaultContext(host)`` helper -- so a future dispatcher with caching / telemetry / custom error UI swaps contexts by passing its own object, no patching the registry.  ``mount(host, file, ctx)`` now REQUIRES ctx (TypeError otherwise) -- explicit beats magic defaults.  (2) ``setFile`` dropped from the Inspector handle contract -- YAGNI, no inspector uses it; reintroduce only when a real need shows up.  (3) ``_basename()`` -- duplicated across 5 inspector modules + 2 viewer.js files -- centralised in new ``lib/path-utils.js`` (``window.molbuilder.path.basename``).  Each inspector now imports via a tiny fallback-safe const.  (4) Backend depth-check duplicated between ``/api/files/write`` + ``/api/files/upload`` + ``/api/files/delete`` -- extracted ``_depth_inside_root(resolved) -> Optional[int]`` helper; all three endpoints share one implementation of the security boundary.  (5) Dead code in ``source.js`` -- ``r.truncated`` branch was checking a field the backend never sets (``/api/files/read`` returns 413 on oversize, no separate truncate path); removed + comment clarifies the all-or-nothing read semantics.  (6) Dispatcher startup validation in ``results/viewer.js`` -- logs a loud ``console.error`` when the registry is empty at init, catching "inspector ``<script>`` tag failed to load / parse" before the user sees a blank panel.  Tests: 2 new pins in ``test_results_blueprint.py`` (createDefaultContext export + dispatcher-build-mount-context-once invariant + registry-validation-at-init).  Rejected findings: "fetch listener leak" (the disposed flag actually gates the write -- Promise just resolves to no-op), "mount error swallowing" (ctx.showError IS called; intentional graceful degradation), "registration order untested" (already covered by test_pick_dispatches_compound_log_to_trajectory).  37 backend file-ops tests still pass after the depth-helper extraction; 33 /results blueprint tests pass after the contract refactor. | Per the user's mandate: "rather than packaging piles of hacking or patched code into sealed pockets, use higher-level data structure design and isolation of modules" -- the registry foundation now (a) has a single, clean injection point for mount-context customisation (vs. the magic default that was buried inside mount()), (b) doesn't carry unused interface surface (setFile), (c) shares the basename helper across the inspector ecosystem instead of duplicating, (d) shares the security-boundary depth check across all three write-side endpoints, and (e) refuses to start without verifying its preconditions.  Stage 1C (trajectory inspector lift) now has a fully solid foundation -- the next inspector that lands plugs into a clean contract with explicit context injection, shared utilities, and zero dead code paths to step on. |
| 2026-05-17 | **Trajectory inspector root-scoping (migration step 1B of task #58).**  ``static/watch/viewer.js``'s IIFE body is now wrapped in a function ``mountInspector(rootEl)``; the auto-bootstrap at the bottom calls ``mountInspector(document)`` so /watch behaves identically.  Inside the function: 38 ``$("id")`` call sites on partial-declared ids stay on a SCOPED helper (``$ = (id) => rootEl.querySelector("#" + id)``); 4 page-level loader-bar ids (``path-input``, ``load-btn``, ``status``, ``file-picker`` -- all in watch.html OUTSIDE the partial) moved to ``$doc()`` which stays document-wide.  This is the mechanical refactor stage 1C will lift into ``lib/inspectors/_trajectory_core.js`` next session: same code, but called by the registry's trajectory inspector when /results mounts a ``.molwatch.log``.  Static validation script confirms all 38 scoped ids match the partial's declared set, 4 $doc ids are all page-level, exactly one ``document.getElementById`` call site (inside ``$doc``'s own definition), braces balanced (232 pairs).  Tests: 7 new in ``test_trajectory_inspector_partial.py::TestViewerJsRootScoping`` pin every invariant (mountInspector function exists, bootstrap call present, $ uses rootEl, no direct document.getElementById leaks, scoped ids match partial, $doc helper is document-scoped, page-level ids don't go through $).  One stale string-pin in ``test_web.py::test_watch_viewer_js_honours_path_url_param`` updated from ``$("path-input")`` to ``$doc("path-input")``.  119 passed in the watch+sidebar sweep after the fix; no behavior change on /watch. | Sets up stage 1C (lifting the body to lib/) as a mechanical move-and-rename rather than a careful refactor.  Per the user's "clean and correct from bottom up" mandate: instead of duplicating code between /watch and /results, this stage proves the inspector body CAN be scoped to an arbitrary host element while preserving every /watch behavior.  Test pins make accidental regression of the scoping invariants impossible -- the next refactor that drops the ``$doc`` helper or rewires a partial-id back through ``document.getElementById`` lands with a clear failure rather than a runtime mystery on /results. |
| 2026-05-17 | **Inspector Registry architecture lands; /results is registry-driven (task #58 mid-stage).**  Higher-level data-structure refactor that replaces /results' hardcoded "5-panel template + 5-match-rule literal in dispatch JS" with a single Inspector interface + a registry.  New surface in ``static/lib/inspectors/``: ``registry.js`` (the contract + ``register/pick/mount/list/_clear`` API), plus four self-registering inspector modules: ``source.js`` (real -- read-only listing of .fdf/.py/.out/.log/.json/.txt/.md via /api/files/read), ``structure.js`` (real -- 3-D preview of .xyz/.pdb via mol-viewer.js factory + an "Open in Modify" link), ``trajectory.js`` (placeholder -- the real lift from watch/viewer.js is migration step 1B+C+D), ``spectra.js`` (placeholder -- the real lift is migration step 2).  Interface: ``{name, displayName, match(filepath), mount(host, file, ctx) -> {dispose, setFile?}}``.  Mount context provides ``showError`` + ``readFile`` wrappers so inspectors don't reinvent HTTP plumbing.  Registry orders by registration; first-match wins (compound extensions like ``.molwatch.log`` MUST register before ``.log``-claiming inspectors).  ``register()`` is idempotent on ``name`` -- swap a placeholder for a real implementation with one require + no other code changes.  Template collapses from 5 panels to one ``#inspector-host`` + a fallback section the dispatch detaches at mount time.  ``results/viewer.js`` is now 3 logical steps: subscribe to projects.onChange, call ``registry.mount(host, file)``, dispose the previous handle before the next mount (timer/listener-leak guard).  Per-inspector style under ``static/results/style.css``'s ``.inspector-card``/``.source-card``/``.structure-card`` namespace.  Tests: 31 in ``test_results_blueprint.py`` (route + sidebar + script load-order invariants + module-served checks + interface introspection + per-extension match pins + dispatch JS code-shape) + 17 Playwright-gated tests in ``test_inspector_registry_e2e.py`` (live registry behavior: pick null on unknown, compound > plain ordering, idempotent register, mount returns dispose-able handle, dispose clears host).  ``pyproject.toml`` package-data globs widened for ``lib/inspectors/*.js``. | The architectural shape Initiative 2 needed BEFORE the bulky trajectory/spectra lifts: each future inspector is a self-contained module that adds via ``<script>`` tag + a register call -- no template edits, no dispatch-rule edits.  ``source`` + ``structure`` ship as REAL functional inspectors (the .xyz/.pdb peek + the .fdf/.py source view are both useful today) so the architecture is exercised against working code, not just placeholders.  Placeholders for trajectory + spectra preserve user agency (link straight to /watch and /spectra respectively) and document where the real lift goes.  Per user mandate: "we need a clean and correct design and implementation from bottom up" -- the registry IS the bottom-up architecture; everything else slots into it. |
| 2026-05-17 | **`DELETE /api/files/delete` is live (closes Initiative 1).**  Replaces the 501 stub.  Validation contract matches the JS-side ``_isDeletableEntry`` gate already shipping in the sidebar's per-entry × button: (a) path resolves inside an allowed picker root, (b) depth >= 1 (refuse to delete the picker root itself), (c) if path is a directory AND its name is in ``CANONICAL_TOPICS`` AND it sits at depth 2 directly under a project, refuse (would orphan the layout; user goes to shell + recreates via + New project), (d) non-empty directories require explicit ``recursive=true``.  The depth-2 canonical-topic guard fires only for directories -- a plain file at depth 2 named "spectrum" is still deletable.  No second-confirm at the backend: the sidebar's native confirm() dialog is the single confirmation point.  15 new tests in ``tests/test_web_files.py::TestFilesDelete``: 3 happy paths (file, empty dir, recursive non-empty), 12 rejection paths (missing body/path, 404, outside-root, ``..``, picker-root, two canonical-topics, file-named-topic OK, depth-3-under-topic OK, recursive flag, project dir with recursive). | All formerly-stubbed file-I/O endpoints (write, upload, delete) are now functional -- Initiative 1 ("sidebar as canonical entry point for all file I/O") closes for the v1 op set.  Edit-and-save in the preview modal remains UI-disabled but the backend ``/api/files/write`` with ``expected_mtime`` is ready when the modal is wired.  Rename is the only file op without a design yet; defer until a real use case shows up. |
| 2026-05-17 | **Trajectory-inspector partial extracted (migration step 1, stage A of task #58).**  Moved the run-state badge + viewer card + viewer-row (3Dmol mount + frame strip + four ctab panels) + SCF banner + plots row out of ``templates/watch.html`` into a new shared partial ``templates/_trajectory_inspector.html``.  ``watch.html`` now consumes it via ``{% include %}``.  Page-specific markup (loader bar with path-input + Load button + status banner; staged-relaxation workflow guide) stays in ``watch.html`` -- the loader bar's path-input + Load button UX is /watch-specific (the /results tab in step 4 will drive loading from the sidebar selection instead).  Zero JS changes: ``watch/viewer.js`` reaches the same ids unchanged.  8 new tests in ``tests/test_trajectory_inspector_partial.py`` pin: (a) the partial declares the canonical 40-id set, (b) the partial contains no page-specific markup (with Jinja+HTML comments stripped so the partial's docstring can name the excluded ids), (c) watch.html includes the partial, (d) watch.html no longer inlines any partial-owned id (no duplicate render), (e) every partial id round-trips into rendered /watch HTML, (f) no duplicate ids in rendered /watch, (g) page-specific ids preserved.  Validation script confirmed 68 unique ids in rendered HTML, all 40 partial ids present, all 5 canonical page-specific ids preserved. | The structural seam migration steps B + C + D (JS-side root-scoping + lib factory + /results trajectory panel inclusion) build on.  Done as its own milestone so any future regression in the markup-extraction lands with a clear failure mode (the new test suite catches "partial leaks page-specific markup" / "watch.html inlines a partial-owned id" / "rendered HTML has duplicate ids") without entangling with the JS refactor risk.  Stages B + C + D remain pending for a focused follow-up session; each is on the same magnitude as stage A and earns its own milestone + tests.  Per the user's policy: "we need a clean and correct design and implementation from bottom up" -- this stage is the bottom. |
| 2026-05-16 | **`/results` dispatch shell lands (migration step 3 of task #58).**  New route ``GET /results`` served by ``web/blueprints/results.py``; template ``results.html`` carries five inspector containers (trajectory / spectra / structure / source / none) all hidden at page-load except the fallback.  ``static/results/viewer.js`` subscribes to ``window.molbuilder.projects.onChange``; on each selection change it picks one panel via a deterministic dispatch table (``.molwatch.log`` → trajectory; ``.spectra.json`` → spectra; ``.xyz``/``.pdb`` → structure; ``.fdf``/``.py`` → source; else → fallback) and toggles ``hidden``.  Compound extensions are matched BEFORE plain ones so the trajectory / spectra inspectors win over a generic source view.  Each panel is a placeholder in step 3 -- real inspectors lift here from Watch + Spectra in migration steps 1 + 2 (see ``docs/protocols/results-tab.md`` § 4).  Tab nav grows a "Results" entry; ``Watch`` stays until step 6 so trajectory inspection has a working home during the migration window.  17 tests in ``tests/test_results_blueprint.py`` (route + sidebar inclusion + 5 panel ids + hidden-by-default + status header + nav inclusion + active-tab marker + non-cross-active + dispatch JS string pins for every documented extension).  ``/results`` added to the cross-page no-JS-errors guard (``tests/test_pages_no_js_errors.py``). | Builds the seam the rest of the merge (steps 1, 2, 4) plugs into without disturbing the existing tabs; ``/watch`` and ``/spectra`` both still work unchanged.  Doing the shell first means the dispatch architecture is validated -- it routes correctly today, just to placeholders -- so when the inspector modules lift in, the only new variable is the inspector itself.  Per-extension matching in the JS (rather than via a server-side ``GET /api/results/<file>`` lookup) keeps the seam clean: no new HTTP endpoints, no new wire contract; just a thin selector that calls the existing ``/api/watch/*`` + ``/api/spectra/*`` + ``/api/files/*`` endpoints once a real inspector mounts.  Compound-extension priority matters because future ``.log`` / ``.json`` plain-source views must NOT pre-empt the canonical inspectors. |
| 2026-05-16 | **Upload feature lands (task #56) — `/api/files/upload` is no longer a stub.**  POST multipart with ``target_dir`` + ``file``; same depth-aware rule as ``/api/files/write`` (target_dir must be inside the picker root + at depth >= 1 + already exist).  Filename validated by ``_UPLOAD_FILENAME_RE = ^[A-Za-z0-9][A-Za-z0-9._-]*$`` -- distinct from ``validate_name`` so extensions work but path separators / spaces / leading dots / shell metacharacters are out.  No implicit overwrite (409 on name clash; user deletes first).  ``os.path.basename(upload.filename)`` defangs any client-supplied path prefix.  Size cap honours the global ``MAX_CONTENT_LENGTH = 50 MB`` from app.py (413 handler already in place).  12 new tests in ``tests/test_web_files.py::TestFilesUpload`` (happy path + missing fields + depth-0 + missing dir + dir-is-a-file + outside-root + ``..`` rejection + 409 + bad-chars + dotfile + client-prefix stripping).  Frontend wiring was already in place from sidebar v5; this commit replaces the stub with real behaviour without touching the UI. | Pre-merge prep for the ``/results`` tab work (task #58): the inspect tab will routinely ingest user-supplied ``.molwatch.log`` and ``.spectra.json`` files, and the existing "scp / mv into projects/" workaround is friction.  Reuses the same path-validation + depth-check helpers as ``/api/files/write`` so the security boundary is shared (one place to audit).  Restrictive filename regex matches the sidebar's own hidden-filter (no leading dots) so a successful upload always becomes visible in the next sidebar list call.  Keeping ``DELETE /api/files/delete`` as a 501 stub for now -- destructive ops want their own UX round. |
| 2026-05-16 | **Projects sidebar v5: upload + delete stubs (501) + file-preview modal (view fully functional, save stubbed).**  Sidebar grows three new affordances whose UX is fully wired today but whose destructive / write-side behaviour is deferred to focused follow-ups: (a) ``+ Upload file`` foldable section below ``+ New subdir`` (same depth-aware visibility -- hidden at projects/ root); submit POSTs multipart to ``/api/files/upload`` which currently 501s with an explanatory message ("scp / mv into projects/ for now"); the inline-error UX renders the message exactly like the future real backend's 409/403/etc. would.  (b) Per-entry × button on hover for deletable entries; JS-side ``_isDeletableEntry`` gates at depth 0 (cannot delete projects from UI) + depth 1 (cannot delete canonical-topic dirs); native confirm() dialog before sending; ``DELETE /api/files/delete`` 501s for now.  (c) Preview modal (body-level floating window): actions section grows a Preview button enabled when a file is selected; click fetches via ``/api/files/read`` (already shipped) and renders the text in a ``<pre>``; Save button visible-but-disabled with ``title="Save is not implemented yet"``; Esc / Close / backdrop click all dismiss.  This is *fully functional* for view -- the 413 / non-UTF-8 / 404 errors from the read endpoint surface inline in the modal's error slot.  11 new tests in test_web_files.py (3 backend-501 pins; 8 sidebar markup + JS-source pins).  74 file-tests total. | Two interleaved goals: (1) commit to the UX surface so the user can review what each future feature will feel like without waiting for backend work; (2) ship the read-side preview feature whose backend (``/api/files/read``) was already done -- the only cost is the modal, and the user immediately gets "peek at any .spectra.json / .fdf / .py / README without leaving the UI" for free.  Choosing 501 + the standard ``{ok:false, error:...}`` shape (rather than a frontend-only "coming soon" toast) means the inline-error UX is the same code path the real backends will use; no special-case branch to retire later.  The per-entry × on hover (vs. a separate "Delete selected" button) keeps the action close to the target while avoiding clutter; the JS-side eligibility check matches the future backend's depth rules so the user never sees a control they can't use.  See ``docs/protocols/selection.md`` § File-manipulation endpoints for the row-by-row status. |
| 2026-05-16 | **Projects sidebar v4: `+ New project` form, `user` canonical topic, per-subdir READMEs, pseudopotential design.**  (1) Sidebar gains a foldable ``+ New project`` form (above the existing ``+ New subdir``) that bootstraps the full canonical skeleton via ``POST /api/projects/create``: strict-conflict (409 with clear message if the name exists), atomic (partial-failure rolls the project dir back via ``shutil.rmtree``).  Wraps a new ``molbuilder.projects.populate_project_skeleton(path, project_name)`` helper that the web blueprint + future CLI / Python API can share.  (2) Extended ``CANONICAL_TOPICS`` with ``user`` -- a free-form workspace at depth 1 where the user is the decider; depth 2+ inside ``user/`` accepts any ``validate_name``-valid name, no canonical-topic restriction.  Preserves the strict depth-1 vocabulary across the other 8 entries (workflow consistency across projects) while letting users opt out cleanly when they need ad-hoc structure.  (3) The bootstrap writes a short ``README.md`` in every canonical subdir + a project-level ``README.md`` describing the layout -- teaches a new user (or a colleague handed a project tarball) what each dir is for without leaving the file tree.  Content lives in ``molbuilder.projects._TOPIC_READMES``.  (4) Sidebar JS hides the ``+ New subdir`` foldable section when ``current_dir`` is at the ``projects/`` root -- keeps the root clean (only project dirs there).  ``+ New project`` stays visible everywhere.  (5) Captured the future pseudopotential-management design (task #55): integrated into the SIESTA script generator as a reusable ``molbuilder.pseudos`` module; the rendered script checks the project-local ``pseudopotential/`` dir at run-time, falls back to a configurable pseudo-dojo download.  The ``pseudopotential/`` subdir + its README are the storage half of that contract today.  (6) Captured the upload-feature naming rules (task #56): same depth-aware location rules apply; filename validation will need a regex that allows dots for extensions (different from the dir-name regex).  4 new tests (user topic, READMEs, free-form-inside-user, depth-aware visibility); 63 file-tests total. | The ``+ New project`` form is the natural entry to setting up a new run -- one click bootstraps every canonical dir + the READMEs guide the user through what each is for, so they don't have to read docs before exploring.  The ``user`` topic resolves the tension between "consistent vocabulary across projects" (strict depth 1) and "user is the decider" (the user's framing): both win, with ``user/`` as the explicit escape hatch.  Per-subdir READMEs cost nothing to write and pay back every time a new user or a returning user lands on the project tree.  The depth-aware ``+ New subdir`` visibility implements the user's "clean root" preference -- the projects/ root has exactly one creation affordance, and it's the right one.  Pseudopotential management lands in a separate PR with focused design (default pseudo-dojo set per functional, per-project vs global cache, HPC offline mode, first-time consent prompt). |
| 2026-05-16 | **Projects sidebar v3: Inquire model + `+ New subdir` button.**  Sidebar reduced to a pure file browser + state holder + file-manipulation widget; tabs pull from the new ``window.molbuilder.projects.{getCurrentDir,getCurrentFile,onChange,readCurrentFile,relativeToProjects,refresh}`` API on their own user-triggered events (no more sidebar-side "Open in <Tab>" buttons, no per-tab auto-load hooks).  The Inquire API + sidebar interaction rules + file-manipulation endpoint surface are spec'd in ``docs/protocols/selection.md``.  v1 ships one file-manipulation operation: ``POST /api/files/mkdir`` with depth-aware naming validation -- depth 0 (under projects/) and depth 2 (under a topic) use ``molbuilder.projects.validate_name`` (regex), depth 1 (under a project) uses ``validate_topic`` (CANONICAL_TOPICS).  Each subscriber tab grows a small "Load from current selection" button (disabled until current_file's extension matches the tab's accept list), wired via ``proj.onChange`` so the enable/disable state updates live.  Spectra additionally renders a workspace indicator showing the current_dir as the implicit output target for the next Generate.  51 tests in ``tests/test_web_files.py``. | The pull-based design proposed by the user; pivots from the v2 push-based (sidebar dispatched cross-tab actions) to a clean separation: sidebar = file browser, tabs = action owners.  Smaller code (no OPEN_TARGETS dict, no projects-selection shim), clearer ownership (each tab independently testable), and easier to extend (new tab? sidebar unchanged).  The ``+ New subdir`` button enables the "set up a new run" workflow without leaving the browser: navigate to projects/proj/spectrum/ → click "+ New subdir" → "water_v2" → sidebar lands in projects/proj/spectrum/water_v2/ → switch to Spectra → Generate (output lands in water_v2/, via the workspace indicator's awareness of current_dir).  Rename / Delete / Upload are deferred -- each is a real feature deserving its own design + UX round.  See ``docs/protocols/selection.md`` § 8 for the explicit anti-patterns this design retires. |
| 2026-05-15 | **Projects pivot: tab → persistent left sidebar; single root (`projects/`); browser `<input type=file>` dropped from Spectra/Watch/Modify.**  Same-day follow-up to the Projects-tab landing: per user feedback, refactored to a persistent left sidebar (JupyterLab style) instead of a top-level tab; the sidebar is included in every tab so the selection-state contract no longer requires tab-switching.  Replaced the multi-column file explorer with a single-column + breadcrumb design (much more compact, fits in the sidebar width).  Dropped the multi-root contract: ``Capabilities.file_picker_roots()`` now returns exactly ``((projects_root(), "projects"),)``; CWD default + ``file_picker.roots`` config section retired (plural tuple shape preserved so re-adding is one line if it earns its complexity).  Dropped the browser-local ``<input type=file>`` from Spectra/Watch/Modify -- a server-side compute script can't read a laptop file regardless, so the picker advertised a contract it couldn't honour; raw-text paste preserved where it exists, sidebar is now the primary loading path.  Added "Upload to projects/" as a future feature (deferred -- only if a real need shows up; design = file actually copied to disk so the sidebar can see it).  Modify viewer.js exposes ``window.molbuilder.loadXyzText(text, filename)`` so the sidebar selection feeds straight into the parse-and-apply flow via ``/api/build/load``'s JSON variant (no DataTransfer hack).  41 tests in ``tests/test_web_files.py``. | The "tab" framing made the explorer feel like a destination ("go to Projects to pick a file"); the "sidebar" framing matches the actual workflow ("the explorer is always there, like Finder's sidebar").  Persistent sidebar also eliminates the cross-tab UX awkwardness of switching to Projects, selecting, switching back.  Single root + no CWD reflects molbuilder's actual scope: ``projects/`` is the canonical home for run-state, full stop.  Dropping the browser-file-input is honest -- the in-memory parse-and-discard flow it had created a contract gap (the user thinks they uploaded a file; the server discards it after parsing) that "upload to disk" would fix only as a real feature, not as a label change.  See spec.md updates for Spectra/Watch/Modify per-tab notes (each tab now states the sidebar is the primary loading path). |
| 2026-05-15 | **Projects tab (server-side file explorer) + shared selection state across tabs.**  New top-level Projects tab (first in the nav) is a column-view file explorer (macOS-Finder-style; matches the ``projects/<project>/<topic>/<structure>/<file>`` depth naturally).  Backend: ``/api/files/{roots,list,stat,read}`` under ``molbuilder/web/blueprints/files.py``, gated by ``Capabilities.file_picker_roots()`` which always includes ``projects/`` + CWD and accepts user-added roots from a new ``file_picker.roots`` section in ``molbuilder.json``.  Path validation: raw ``..`` rejected, resolved path must be inside an allowed root (defense in depth).  Selection state is shared via sessionStorage (``molbuilder.current_dir``, ``molbuilder.current_file``); other tabs (Spectra, Watch, Modify) subscribe via the cross-tab ``storage`` event AND a same-tab ``molbuilder.selection`` CustomEvent.  Each subscriber tab gains a ``.projects-banner`` element + a one-line wiring call to ``molbuilderProjectsSelection.init({bannerEl, extensions, onLoad})``; clicking the banner's "Use this file" button populates the tab's existing loader.  Additive: every tab keeps its local-file input + raw-text paste.  Read-only in v1 (no upload / rename / delete -- this is a *navigation + selection* widget, not a file manager).  Naming constraint inherited from ``molbuilder.projects.validate_name`` (regex ``^[A-Za-z0-9_-]+$``) applies to derived job creation (Phase 2); the picker itself shows whatever's on disk so users can find and rename misnamed dirs.  31 tests in ``tests/test_web_files.py``. | Solves the recurring "the browser file dialog can't see server files" UX pain across every tab.  Persistent panel (not modal) was the user's preference and matches JupyterLab's file-browser pattern; the cross-tab event bus is cheap and lets the picker stay the single source of truth for "what file am I working on right now."  Column view fits the project hierarchy depth (project / topic / structure / file) without the "where am I in the tree" problem.  Phase 2 (job derivation -- start a new run from an existing file) is documented in the same web-api.md section and lives in ``derive_job`` design but is deferred to a separate PR. |
| 2026-05-15 | **IR add-on scaffold landed (compute_ir=True); absolute magnitudes NOT YET VALIDATED.**  Setting ``compute_ir=True`` (with the v1 constraint that ``compute_raman`` must also be True) populates ``ir_intensity_km_mol`` on every mode.  The IR FD piggybacks on the existing Raman displaced-SCF loop -- PySCF's ``mf.dip_moment(unit='Debye')`` is essentially free after an already-converged SCF, so the cost is zero extra SCFs.  Projection: ``dμ/dQ_n = Σ_{k,α} (dμ/dR_{k,α}) · L_canonical_{k,α,n}``; intensity ``I_n = 42.2561 · |dμ/dQ_n|²`` km/mol (the standard Gaussian / ORCA / psi4 constant for the (D/Å)²/amu → km/mol unit conversion).  v1 constraint enforced at script-render time with a clear ``ValueError``.  Generated scripts carry a prominent ``NOT YET VALIDATED`` banner in the header docstring when IR is on; ``compute_ir`` help text + ``docs/tabs/spectra/spec.md § 13.1`` document the validation gap and the four-step plan to close it.  7 new unit tests in ``tests/spectra/test_script.py::TestPySCFScriptIRScaffold`` pin the emission shape, the prefactor constant, the explicit ``unit='Debye'`` argument, the banner, and the v1 constraint. | The user's preferred sequencing: get the IR data flowing through the pipeline so the UI can be designed against real (if unvalidated) numbers, then close the validation gap before declaring it production-ready.  The "free with Raman" framing is honest: the math + the constant are textbook, what's unverified is whether PySCF's dipole convention matches the formula's assumptions to < 2% (the bar the Raman validation cleared).  Keeping the scaffold visible-with-warning is better than holding the feature back: it lets the UI team work concurrently and lets users explore relative IR intensities and qualitative patterns now. |   Cross-checked molbuilder's full ``build_from_smiles`` → ``pyscf.input.render_script`` (relax) → ``spectra.pyscf_script.render_spectra_script`` (Hessian + Raman) pipeline against a hand-written raw-PySCF reference script (no molbuilder code) starting from the same MMFF water geometry.  Both reach bit-for-bit identical relaxed geometry (max Δ 1.1×10⁻⁷ Å), frequencies (1638.77 / 3791.22 / 3886.54 cm⁻¹, max Δ < 10⁻³ cm⁻¹) and Raman activities (6.816 / 76.905 / 36.818 Å⁴/amu, max Δ < 10⁻⁶ Å⁴/amu).  Frequencies match literature B3LYP/def2-SVP water within ~5 cm⁻¹; Raman activities sit in the standard Å⁴/amu literature range, confirming the BOHR_TO_ANG**6 unit conversion is correct in magnitude.  Full method + result table: ``docs/tabs/spectra/spec.md § 12.1``.  Also surfaced one real bug along the way: ``mf.disp = "none"`` was crashing PySCF's check_disp at the first SCF; fixed in ``molbuilder/pyscf/input.py`` (relax + preopt sites) with two regression tests. | Per the user's "publication-defensible" bar, the framework needed an external sanity check before declaring the pipeline scientifically correct.  The bit-for-bit agreement against an independent code path shows molbuilder adds no computational artifact; the literature-ballpark match on absolute Raman magnitude validates the unit conversion specifically.  Documented in the per-feature spec (spec.md §12.1) rather than the design log so future readers find it next to the schema + conventions it validates.  IR add-on, when scaffolded, gets the same treatment before its absolute values are trusted. |
| 2026-05-15 | **Raman activity unit fix (factor of Bohr⁶/Å⁶ ≈ 0.022).**  The Placzek scalar `45 a² + 7 γ²` was being stored under `raman_activity_a4_amu` without the polarizability unit conversion: PySCF reports α in atomic units (volume = Bohr³), so the projected `dα/dQ` and the resulting activity were in `(a.u. polariz)² / (Å² · amu)`, not the textbook Å⁴/amu the field name advertised.  Relative spectrum shape was correct (uniform scale), but absolute values were ~50× smaller than Gaussian/ORCA Raman activities for the same molecule.  Fix: multiply by `BOHR_TO_ANG**6` once per mode after `45 a² + 7 γ²` (single line in `_emit_raman_block`); the comment block above the loop documents the unit chain end-to-end so a future reader can audit it.  No SCHEMA_VERSION bump — v2 was still in dev with no shipped JSONs, so the units fix lands inside v2's umbrella. | Caught during a fresh-eye review of unit consistency.  Without this factor a user comparing molbuilder Raman intensities against literature would see the right peak positions but consistently wrong magnitudes — exactly the "looks plausible, silently wrong" failure mode that scientific code should not have.  Documenting the conversion at the math (not just the code) lets a reviewer verify it without re-deriving. |
| 2026-05-15 | **GPU coverage probe in the spectra script.**  The generated PySCF spectra script runs a small runtime probe right after the equilibrium SCF converges: it instantiates ``mf.Hessian()`` (no compute, just an object) and inspects its class's ``__module__``.  If the module is ``gpu4pyscf.*`` the script runs ``mf.Hessian().kernel()`` directly on the GPU; otherwise it rebuilds ``mf`` on CPU via the existing ``_build_mf_at(..., force_cpu=True)`` path and runs the Hessian there.  Two flat flags (``_GPU_HAS_HESSIAN``, ``_GPU_HAS_POLARIZABILITY``) carry the result; the latter is hard-coded False because gpu4pyscf does not yet expose analytic CPHF (the Raman polarizability path already forces CPU at the call site).  A scientist sees one of two lines printed: ``GPU coverage gaps: ['Hessian']`` (with CPU rebuild) or ``GPU coverage: SCF + Hessian.`` (full GPU path). | gpu4pyscf's coverage matrix is a moving target -- as of 2026-05 it covers RKS/UKS Hessian but lags on others, and polarizability is unimplemented.  Hard-coding the matrix in the generator would drift; probing at runtime adapts to whatever the user has installed.  The "instantiate + check module" approach (instead of ``try: mf.Hessian().kernel(); except``) avoids wasting an SCF on a path that can't continue: the probe is free, the SCF-rebuild cost is paid only when there's a gap.  See ``pyscf_script._emit_gpu_coverage_probe``. |
| 2026-05-14 | **Capabilities snapshot** (`molbuilder.diagnostics.Capabilities`) is the single source of truth for "what's available on this machine" (conda binary, set of conda env names, parsed `molbuilder.json`).  Host-PATH lookups happen on demand via `shutil.which` inside `Capabilities.tool_available`, not pre-probed -- which is cheap and sidesteps a leaky abstraction (pre-probed PATH would have looked frozen but masked stale state).  Built once at startup by `detect()`, queried O(1) thereafter, frozen so callers cannot mutate.  Backends' `is_available()` reads from it; `run_tool` dispatches based on it.  Tests inject synthetic snapshots via `set_capabilities()` instead of mocking `subprocess.run`.  The bootstrap point is `cli.py:main()` (which calls `diagnostics.initialize()` before click handles the command) and `web/app.py:create_app()` (which calls it before any blueprint registers).  `runtime_config.py` is UI-agnostic (raises `RuntimeConfigError`); `cli.py` translates that to `click.UsageError`, web layer would translate to HTTP 400. | Replaces per-call `shutil.which` + `subprocess.run("conda env list")` + `read_config()` probes that the Phase 1 first draft scattered across modules.  Single snapshot enables: (a) O(1) availability queries, (b) consistent view across all backends and the `/api/backends` endpoint, (c) zero-subprocess tests via direct construction, (d) explicit refresh semantics for the long-running web app.  See code review 2026-05-14: PATH-first preference was a leaky abstraction (system tools silently shadowing curated envs); env-first dispatch now matches the four-env model's intent. |
| 2026-05-14 | **Four-env model**: molbuilder coordinates from a user-named *host env* and dispatches to four named backend envs — `molbuilder-siesta`, `molbuilder-pySCF`, `molbuilder-MDtools`, `molbuilder-tests`.  Heavy tools live in their named env and are invoked by subprocess (`conda run -n <env> ...`); build-time chemistry (rdkit, openbabel, ase, sisl, PeptideBuilder, biopython) stays in-process from the host.  Env recipes live in `docs/README_install.md`; molbuilder does NOT auto-bootstrap conda or auto-create the envs — users follow the documented recipes for whichever tools they actually need.  `molbuilder/__main__.py` added so `python -m molbuilder ...` works without `pip install -e .` from the host env.  Implementation work (`molbuilder.envs.run_in_env`, amber-backend subprocess wrap, `available_backends()` probe extension, `molbuilder run` subcommand + Flask `/api/runs/start`) is deferred — this decision documents the contract only. | Solver experiments (this session, 2026-05-14) showed that collapsing AmberTools-dac=26 + siesta-mpi + playwright + cupy-cuda13x into a single env produces three independent unresolvable conflicts: ambertools needs `numpy<2` (cupy needs `numpy>=2`); ambertools needs `libnetcdf>=4.10` (siesta-mpi linked against 4.9.3); ambertools' X11 stack needs an `icu` version playwright's `nodejs` can't accept.  Naming the envs and letting the user prepare them lets each backend keep its native pin set without poisoning the others.  The `molbuilder-MDtools` env stays separate from the host *specifically* to free the host from ambertools-dac's numpy-1.x lock (which is in numpy upstream-maintenance-only mode since 2024-06); merging would re-couple host releases to dacase's release cadence indefinitely.  The host's "user picks the name" choice mirrors Principle-#1's anti-coupling: molbuilder doesn't need to control where users put their envs, only what names the env-dispatch helper looks for. |
| 2026-06-02 | **Selection blueprint hardening: sidecar concurrent-write lock + comprehensive error-path coverage (tasks #148 + #147).**  (1) ``/api/selection/save`` had a classic lost-update race on the ``.molstruct.json`` sidecar: two concurrent ``setShared``-style clicks both read state X, both compute disjoint mutations, the second writer's ``os.replace`` clobbers the first.  Added ``molstruct_json.with_lock(sidecar_path)`` context manager -- exclusive ``fcntl.flock`` on a SIBLING ``<sidecar>.lock`` file (NOT the sidecar itself, since ``save()`` atomic-replaces the sidecar via ``os.replace`` and a lock held on the OLD inode doesn't carry across the swap).  ``O_CLOEXEC`` on the lock fd so a forked child can't hold up the parent's next save.  Windows fallback to no-op (the import path still works; serialisation correctness is only guaranteed where ``fcntl`` is present, which matches the POSIX deployment target).  The endpoint's load-mutate-save block now sits entirely inside ``with molstruct_json.with_lock(sidecar_path):``.  Tests: 3 unit tests on the lock primitive (concurrent RMW from two threads with a 50 ms in-block sleep widening the race window -- both keys land in the final sidecar; lock file is created sibling-not-sidecar; .lock created but the sidecar isn't) + 1 integration test that spins up a real werkzeug server and fires two parallel POSTs from worker threads (Flask's ``test_client`` isn't thread-safe; the real-server pattern matches production wire-up).  (2) Error-path coverage: 29 new tests across 5 classes (``TestAtomsErrorPaths`` / ``TestSaveErrorPaths`` / ``TestEvalErrorPaths`` / ``TestToggleErrorPaths`` / ``TestCrossCutting``) so every error branch in ``selection.py`` has at least one test pinning its status code + message shape -- path traversal, non-structure extensions, file-not-found 404, non-string ``structure_path``, missing / non-list / non-dict body parts, parametrised "non-JSON body returns 400" across all four endpoints.  Selection blueprint suite grew 38 -> 67 tests; 102/102 total across molstruct + selection + the modify-layout phone-width test. | The sidecar race was a latent bug the modify UI's "Apply" flow could trigger by clicking two region tags in quick succession.  The atomic-write in ``save()`` (tempfile + os.replace) only addressed the "no half-written file" property, NOT the "no lost updates" property -- atomicity per single-writer is different from serialisation across writers.  Lock-on-sibling rather than lock-on-sidecar is the load-bearing detail: fcntl locks live on inodes, ``os.replace`` swaps inodes, so a lock held during the write doesn't follow through to the new file -- the next reader would re-race.  Lock-on-sibling keeps the serialisation cookie stable across saves.  The error-path coverage was a deliberate response to the user's standing "test all middle layers" directive: each endpoint had happy-path tests + a few of the most common bad inputs, but no systematic sweep of "every ``_bad_request`` callsite in the blueprint".  The new error-path classes ARE that sweep -- a regression that swallows an error path (returns 500 / 200 instead of the intended 400 / 404) or drifts a status code is caught at CI rather than discovered by a frustrated user.  ``TestCrossCutting`` parametrised over all four endpoints catches the shared body-validation paths (non-JSON content type, non-object top level) without duplicating boilerplate. |
| 2026-06-02 | **Projects sidebar: narrow-viewport drawer pattern (task #182).**  At desktop widths the sidebar is a 18 rem fixed-left aside with body ``padding-left: 18rem``; at 360 px viewport that produced a 292 px horizontal overflow (the regression test ``test_modify_layout_phone_width_no_horizontal_overflow`` had been xfailed since 2026-06-01).  At <= 640 px the sidebar now collapses to a left-edge DRAWER: ``transform: translateX(-100%)`` slides it off-canvas, body padding-left = 0, a fixed hamburger button (``#ps-mobile-toggle``) at top-left toggles a ``has-mobile-sidebar-open`` body class that brings it back as an overlay with a click-to-dismiss backdrop.  Z-index layering (bottom up): page content (none) -> backdrop 85 -> drawer 90 -> toggle 95 -> file-preview modal 100.  The modal explicitly sits above the drawer so a modal opened FROM the drawer (Preview button in the sidebar) renders on top rather than hidden behind -- closed modals don't participate in stacking (``[hidden] { display: none }``) so the desktop case is unaffected.  Behaviour: button click toggles + flips ``aria-expanded``; backdrop click closes; Escape closes (standard modal-overlay convention); window resize past the 640 px breakpoint auto-closes so portrait->landscape rotation doesn't leave a stale "open" state.  ``initMobileDrawer`` in ``lib/projects-sidebar.js`` no-ops gracefully when the optional toggle / backdrop elements are absent.  Above 640 px every drawer rule is inert (``display: none`` on the toggle + backdrop; padding-left + sidebar transform reset to desktop values).  Also: a residual 4 px overflow after the drawer collapse came from ``grid-template-columns: 1fr`` at the existing 1024 px breakpoint in ``modify/style.css`` -- ``1fr`` respects children's ``min-content`` so a child with an unbreakable element wider than the available column pushes the column past its container.  Fixed by ``minmax(0, 1fr)`` which opts out of the implicit ``min-width: auto``.  Test: xfail removed from ``test_modify_layout_phone_width_no_horizontal_overflow``; 113/0/0 modify_e2e suite (was 113 with 1 xfail).  ``docs/protocols/projects-sidebar.md`` gained a new § 9.3 documenting the drawer architecture + z-index table. | UX preserved end-to-end: above 640 px nothing changes; below it users get a tap-to-open drawer instead of a horizontal scrollbar.  Drawer chosen over "hide sidebar entirely" because the sidebar carries the project file picker -- the only way users select files for modify / build / spectra -- so removing it entirely on phone would orphan the whole tool.  Drawer chosen over "permanent narrow sidebar" because at 360 px even a 6 rem sidebar leaves 264 px of content area which the build form's section headers / 3Dmol viewer / spectra plots can't render usefully in.  Z-index ordering relative to the file-preview modal was the second round of fixes (a holistic-review finding after the initial drawer landed at z-index 110, above the modal at 100 -- a modal opened from the drawer would render hidden behind the drawer).  The ``minmax(0, 1fr)`` grid fix is a load-bearing CSS-grid idiom -- the implicit ``min-width: auto`` on grid items routinely produces overflow when a child has fixed-width content; explicit ``minmax(0, ...)`` is the standard remediation. |
| 2026-06-01 | **/results: tab-level auto-detect file-picker + grouped dropdown + transient status (5 commits).**  Per user directive 2026-06-01: "when i enter a project with output results, the result tab does not automatically detect that there are available files for display - can we make this automatical?  the scan and generation of drop-down files that contains known results (either it is .out for siesta, or .json for spectra, or others for pySCF) can be automatically generated, and the tab will respond to this change.  currently the scan is not done, and the other types are not using this drop-down menu.  in other words, the drop-down menu is only for siesta out, we can make it a general feature for the result tab".  Pre-refactor: a tiny dropdown lived INSIDE the trajectory inspector at ``lib/trajectory/result-list.js``, hit a SIESTA-specific endpoint ``/api/files/result-list`` and only surfaced ``.out`` / ``.molwatch.log``.  Refactor lifted the dropdown to a tab-level widget at ``lib/results/file-picker.js`` (mounted under ``#results-file-picker-bar`` in ``templates/results.html``) and rebuilt its sourcing on the inspector registry: every inspector now declares ``isResult: bool`` (catch-all viewers like ``source`` opt OUT so the picker isn't flooded by configs / READMEs) and ``resultCategory(file): string`` (rendered as ``<optgroup>`` headers in the dropdown).  Picker fetches ``/api/files/list?path=<dir>`` and filters client-side via ``registry.pickResult(file)`` -- the inspectors are the single source of truth for "what counts as a result".  Auto-pick fires ``projects.setShared(dir, newestResult)`` so entering a directory immediately mounts the most-recent run without an extra sidebar click.  Status surface (meta line under the dropdown, forced to a second row via ``flex-basis: 100%``) doubles as a busy indicator: "Scanning for output files…" during the directory listing fetch; "Parsing <basename>…" while the matched inspector loads (cleared by a ``document.dispatchEvent(molbuilder:inspector:ready)`` fired by each result inspector after its first paint -- deferred via double-``requestAnimationFrame`` so the browser commits the new ``frame-tot`` / 3Dmol / Plotly state to pixels BEFORE the picker meta returns to steady state; user explicitly reported 2026-06-01 that a synchronous dispatch caused the meta to flash to "8 of 12 · X ago" while the viewer still read "Frame 0 / 0", which is the rAF-after-applyNewData race the double-tick wait closes).  Group order in the dropdown: groups are sorted by their newest entry's mtime descending so the category containing the user's most-recent run floats to top; entries inside each group preserve the newest-first input order.  Engine-flavoured category labels live on the inspectors -- ``trajectory`` discriminates per-file (``.out`` -> "SIESTA optimization"; ``.molwatch.log`` -> "PySCF optimization"); ``spectra`` -> "PySCF spectrum"; ``structure`` -> "Structure" -- so adding a new result type is a one-line ``resultCategory`` on the new inspector.  Deletions: ``lib/trajectory/result-list.js`` retired (282 LOC); ``GET /api/files/result-list`` endpoint deleted from ``web/blueprints/files.py`` (its only consumer is gone); ``tests/test_result_list_js.py`` superseded by ``tests/test_results_file_picker_js.py`` (27 tests: 4 ``parseDir``, 6 ``formatRelativeTime``, 9 ``filterToResultFiles``, 5 ``groupResultFiles``, 2 ``_labelForResult``, 1 API-surface).  ``test_trajectory_inspector_partial.py`` and ``test_web_files.py`` trimmed of the now-obsolete required-ids + endpoint test class.  Playwright ``/results`` no-JS-errors page-boot smoke green.  Frozen-snapshot fixtures included for the SIESTA dispatch refactor's behaviour-equivalence suite (1.5 MB across 5 .out files covering ``finished`` / ``error: SCF_NOT_CONV`` / ``error: propor ERROR``) -- ``.gitignore`` amended to whitelist ``tests/watch/fixtures/`` since the global ``*.out`` rule would otherwise drop the frozen copies. | The "moved-to-tab-level" decision is the inversion of the old in-inspector design: the dropdown was previously the trajectory inspector's helper (only it knew about sibling .out files), which meant adding spectra-result navigation would have required THREE separate in-inspector dropdowns + three endpoints.  Lifting to tab level + delegating "is this a result?" to the inspector registry collapses N dropdowns + N endpoints into 1 dropdown + 0 new endpoints, using the existing ``/api/files/list`` + the existing ``registry.pick`` mechanism extended with one boolean.  The ``isResult`` opt-in is the deliberate guardrail against the "source inspector matches everything" problem -- the source viewer covers ``.fdf`` / ``.py`` / ``.log`` / ``.json`` / ``.txt`` / ``.md`` as a generic text fallback, which would have flooded the dropdown with input files + READMEs if the picker had just filtered on "any inspector matches".  Per-inspector ``resultCategory`` (rather than a hardcoded category table in the picker) was the user's explicit ask after the first pass surfaced engine-flavoured grouping; keeping the discriminator on the inspectors keeps the picker engine-agnostic.  Status messages on a SECOND line (not inline at the right of the dropdown) followed user feedback that the inline placement let long filenames push the status off-screen -- ``flex-basis: 100%`` forces the wrap.  Double-``requestAnimationFrame`` on the ready-event dispatch was the second round of fixes after the user observed picker-meta-clears-before-viewer-renders: a synchronous dispatch arrives in the SAME event-loop tick as the textContent update, so the browser hasn't painted yet; double-rAF defers the dispatch past one full paint cycle which is the minimum needed for 3Dmol + Plotly to flush their queued render work.  ``test_molwatch_combined_dispatch.py`` + the new picker tests + the existing inspector-registry e2e tests collectively pin the contract: ``pickResult`` honours ``isResult``, ``resultCategory`` falls back to ``displayName``, groups sort by newest-mtime, auto-pick lands on ``results[0]``, ``molbuilder:inspector:ready`` is dispatched by all three result inspectors. |
| 2026-06-01 | **molwatch_log parser: port to the combined-regex dispatch primitives (commit bc718ea).**  Second parser ported to the rule-based dispatch + combined-regex pre-filter introduced for SIESTA in commit e1a517d (Strategy D commit 2).  Pre-refactor's ~12 individual checks per line (4 cross-block ``re.match`` probes + 2 boundary ``re.search`` probes + 6 ``startswith`` checks) collapse into ONE DFA scan per line, with per-rule verification only on a hit.  Architecture: two ``CompiledRules`` tables (``out_block_rules`` / ``in_block_rules``) switched per iteration on the ``in_block`` flag.  ``block_begin`` lives in BOTH tables so a stray ``==== begin ====`` mid-block abandons the partial frame and starts fresh (preserves pre-refactor recovery behaviour).  Multi-line sections (coords / forces / scf) use ``SectionRule.consume`` returning ``CONTINUE`` / ``END_SECTION`` / ``END_BUBBLE`` -- ``END_BUBBLE`` re-feeds the terminating line through scan-state rules so a ``coordinates`` block ending on ``forces (eV/Ang):`` still triggers the forces section.  Tests: 6 new tests in ``tests/watch/test_molwatch_combined_dispatch.py`` covering torn-begin-mid-block recovery (no prior coverage; the pre-refactor parser handled this in inline ``_reset_block()`` and the rule-based path mirrors it via dual-table membership), runtime_info-inside-block-ignored (guards the ``OUT_BLOCK_RULES``-only placement of the runtime rule), concluded-then-error-precedence (reverse-order case of the existing test), rich-block frozen golden equivalence (all section types + None-residual SCF cycles + runtime_info dict including ``None`` value + concluded footer), determinism, and cross-parse independence.  23 existing spec-derived ``test_molwatch_log_parser.py`` tests continue to pass unchanged.  All 202/202 watch tests green. | The combined-regex primitives in ``_rules.py`` (Strategy D commit 1) were always intended to be engine-agnostic -- SIESTA was the first port, and the design doc note ("Future ORCA / NWChem / Gaussian text parsers reuse _rules.py unchanged") explicitly anticipated additional engines.  The molwatch port verified the primitives ARE reusable for a parser with substantially different shape (two-state state machine driven by a ``begin`` / ``end`` marker pair, rather than SIESTA's monotonic forward scan).  Two-table switching is the cleanest way to model the in-block / out-of-block dichotomy: the alternative -- one table with state-aware predicates on each rule -- would have demoted every rule to predicate-only (no regex-pre-filter participation), defeating the whole point of the refactor.  PySCF's main ``.xyz`` trajectory parser was deliberately NOT ported: it's a format-driven parser (fixed-shape header → exactly n_atoms atom rows) with no per-line scan loop, so the combined-regex abstraction doesn't fit.  Per-iteration-rich tests + the frozen synthetic fixture (no real ``.molwatch.log`` files exist in the repo) lock the parse output so a future refactor that breaks any field surfaces immediately. |
| 2026-06-01 | **SIESTA parser: combined-regex section dispatch (Strategy D, 3 commits).**  Closes the perf gap surfaced by the 2026-05-31 D1 experiment.  D1 (commit 26cf89f) cached ``line.lower()`` / ``line.lstrip().lower()`` per iteration via module globals populated by a driver hook -- ~14% speedup but a fragile contract (matchers invoked outside the driver hit STALE globals and silently returned wrong answers).  Strategy D replaces D1 with a regex-DFA pre-filter: every literal-pattern rule contributes a pattern fragment to one ``re.compile`` alternation; the driver runs the combined regex once per scan-state line; on a hit, per-rule individual regex verifies registration-order tie-break (NOT the leftmost-position the combined regex's group order would imply); predicate-only rules (closures over parser state like ``_max_force_match``) still iterate individually with § 6 per-rule error isolation.  Three commits: (1) ``_PatternMatcher`` abstraction + ``matches_regex_ci`` + ``CompiledRules`` + ``compile_rules`` + ``SectionRule.__post_init__`` validation + 38 unit tests covering helper shape, pattern-callable equivalence, ``any_of`` pattern composition, ``compile_rules`` edge cases (empty / all-predicate / mixed / duplicate-name), and ``find_match`` dispatch order (registration tie-break, predicate-before-regex, regex-before-predicate, three-rule cascade, no-match, predicate exception isolated).  (2) ``siesta.py`` driver swap + SCF lambda rules converted to ``matches_regex_ci`` so they participate in the combined-regex fast path + frozen-fixture parse-equivalence suite at ``tests/watch/test_combined_dispatch.py`` (5 fixtures spanning ``finished`` / ``error/SCF_NOT_CONV`` / ``error/propor ERROR`` + determinism + cross-parse independence + perf envelope).  (3) Cleanup: stale ``from . import _rules`` removed from siesta.py, unused ``Union`` import removed from _rules.py, module-level docstring updated to reflect the new primitives.  Behaviour preserved verbatim: dispatch trace logs on the same frozen .out are byte-for-byte identical between OLD and NEW (266 rule firings on the 3279-line sample, zero divergence); the apparent "drift" during initial debug was caused by SIESTA actively appending to the live ``stage3-run1.out`` between runs -- a frozen-fixture invariant catches this in the test design.  Perf delta on the 813 KB stage2-run3 fixture (42 frames): pre-D1 ~1.0 s, D1 ~0.86 s, this commit ~0.15 s (~6.6× faster than pre-D1, ~5.7× faster than D1).  All 196 watch tests pass (187 prior + 9 new). | The D1 cache was an honest mistake the user caught: caching across matchers in a hot loop SOUNDS right but the contract leaks at the matcher boundary -- a matcher's return depends on hidden module state, which makes the matcher impossible to call correctly from anywhere else without first invoking the hook.  Strategy D inverts this: each matcher is self-contained (regex pattern + callable fallback); the perf optimisation lives at the DISPATCH layer (combined regex pre-filter), not inside the matchers.  The dispatch layer can be replaced without touching matcher correctness, and the matchers can be called from anywhere without setup.  Registration-order tie-break preserved by per-rule individual regex (NOT the combined regex's named-group order) so the existing rule precedence ("fatal_*" before "scf_converged" before "scf_not_converged" etc.) keeps working without re-ordering.  Frozen fixtures were the second lesson: my first equivalence test compared a live SIESTA-output file to a golden generated 30 seconds earlier and the file had grown between captures, manifesting as a phantom 5-frame drift; copying the file to ``tests/watch/fixtures/siesta_frozen/`` removes the moving target.  The test corpus deliberately covers every terminal ``run_state`` (finished / error/SCF_NOT_CONV / error/propor ERROR) so a future dispatch tweak that mis-orders one fatal rule fails LOUDLY on its specific fixture, not vaguely on a generic equivalence run.  Per the user's verbatim 2026-05-31 directive: "we should be careful about test designs, we need to make sure our tests are thoroughly capturing all possible expected results and detect error paths correctly". |
| 2026-05-27 | **Post-review cleanup: 5 layer-spanning bugs + 5 latent correctness gaps + 2 cross-blueprint refactors.**  Triple-reviewer pass (UI / middle / API) on the 2026-05-26 commit set turned up 12 real issues that all landed in one focused run.  (1) ``.disabled-tip`` CSS was in style.css but spectra.html never linked it -- the Save-button wrapper on /spectra fell back to ``display: inline`` and broke baseline alignment; moved to form-schema.css.  (2) ``render_spectra_script`` skipped ``validate(struct, cfg)`` -- CLI / library callers bypassed every cross-cutting check (open-shell metal, parity, frozen-atom-consumed, peptide protonation).  Added ``report(validate(...))`` at the top, mirroring siesta/pyscf.  (3) ``render_fdf`` rebuilt the validation_struct without ``frozen_atoms`` or ``regions`` -- the validator's ``_check_frozen_atoms_consumed`` saw an empty list even when the caller had set them.  (4) Generate handlers didn't clear ``state.savingFdf`` / ``state.savingPyscf`` on failure -- a click mid-Save left the button enabled against a stale state.fdf.  (5) install-wrapper ran even when install-pseudos missed elements -- the .run.sh referenced non-existent .psml files and SIESTA aborted at startup.  (6) ``_ATOMIC_NUMBER`` table stopped at Z=84 -- actinides + transactinides silently read 0 in the parity check; extended to Z=1-118.  (7) ``_validate_pyscf`` read ``el`` without ``.capitalize()`` -- masked by 108c7ff but a latent dependency.  (8) ``md_target_temperature`` had no range metadata while siblings did.  (9) Save buttons enabled at the projects root (raw ``!!dir`` truthy) and then errored "no current_dir" -- added ``projects.atRoot()`` + both check sites use it.  (10) Sidecar notice lost on ValidationError branch -- response now merges exc.issues with the pre-render issues (incl. sidecar warn).  (11) ``_apply_sidecar_if_possible`` moved from spectra blueprint to ``web/blueprints/_shared.py`` so Build doesn't reach across siblings to import it.  (12) Generate handlers got AbortController so two rapid clicks no longer race -- the older click's response is aborted, never overwrites state.  Plus a defensive engine_key-badge unit-suffix dedupe rule for the (hypothetical) case where a label includes the unit literally.  ~ 800 tests pass on the touched suites. | Two themes: ALL fixes were "middle layer between two correctly-tested ends was broken" -- exactly the pattern the user's prior mandate ("you better fucking test all middle layers") was trying to prevent.  The remedy here was the 3-reviewer review pass itself: by reading each layer's recent diff with the question "what does the layer ABOVE / BELOW assume", the reviewers found the gaps that fall through the seams.  The fixes are individually small (one-line race fix, one-import refactor, one-validation-call add) but each was real and each would have shipped to the user as a regression discovered later in the wild.  The ``.disabled-tip`` find is the cleanest example: the workaround I shipped 24 hours earlier worked on /index but silently degraded on /spectra because the per-page stylesheet inventory diverged -- exactly the kind of bug only a cross-file structural review catches. |
| 2026-05-26 | **Hardening pass: element-case canonicalisation + Save reentrancy + engine_key tests + dev-loop infra.**  Three real bugs the 2026-05-25 review surfaced.  (1) ``Structure.from_pdb`` returned ``elements=['FE', ...]`` (PDB cols 77-78 carry NO case convention; PDB Bank canonical files emit ``FE`` / ``CL`` / ``NA`` upper-cased).  Downstream ``render_fdf`` crashed with ``KeyError: 'FE'`` for any PDB with a transition metal.  Fixed at the parser boundary: ``.capitalize()`` in both ``from_pdb`` and ``from_xyz`` so the canonical ``Fe`` / ``Cl`` / ``Na`` form reaches every consumer.  (2) Save handlers (SIESTA + PySCF) were not reentrant -- a 4-step async pipeline (write .fdf/.py, install pseudos, install wrapper, refresh sidebar) ran without disabling the button, so a double-click triggered TWO pipelines into the same dest dir.  Added ``state.savingFdf`` / ``state.savingPyscf`` flags + button disabled + text swapped to "Saving…" + ``finally`` block restoring state.  Pipelines factored into ``_runSaveFdfPipeline`` / ``_runSavePyscfPipeline`` so the guard wraps cleanly.  (3) The 102 ``engine_key`` declarations across SiestaConfig + PySCFConfig had zero test coverage -- a regression deleting one would render no UI badge invisibly.  Five new tests pin: every form field carries engine_key; molbuilder-only markers use ``(molbuilder`` prefix; load-bearing SIESTA keywords (SpinPolarized, MeshCutoff, etc.) match exact spellings; load-bearing PySCF keywords (gto.M(...) signatures) match; method= mentions all four class names.  (4) Dev-loop: ``Makefile`` with ``make test`` (fast ~20s subset) / ``make test-all`` / ``make web`` / ``make web-bg``; ``.pre-commit-config.yaml`` with pyflakes + pre-commit-hooks + local pytest-fast + node -c JS syntax check; ``docs/templates/github-workflows-test.yml`` as a manual-push template (CLAUDE.md memory: PAT lacks workflow scope so workflows can't be auto-committed).  ``e2e`` + ``slow`` pytest markers added so the test subsets are addressable.  327+ tests across the touched suites pass. | Each of the three bugs followed the same pattern -- the engine-emitter end of a chain was unit-tested, but the data-loading or UI-interaction end was not, so a regression that broke the wiring shipped to the user.  The fixes alone don't prevent the next class of the same bug; the dev-loop infra (pre-commit + CI template + Makefile) is what closes the loop so this same shape of bug can't land again silently.  Per the user's "you better fucking test all middle layers of all things" mandate from earlier this session: the 102-field engine_key tests AND the integration test that POSTs through the actual /api/build/fdf endpoint (rather than calling render_fdf directly) are the explicit middle-layer pins the prior coverage was missing. |
| 2026-05-25 | **Three-stage contract: sidecar→Build wiring + propor preflight tightening + suggest_spin_total expansion.**  (1) The 2026-05-24 frozen_atoms commit made the SIESTA + PySCF emitters CAPABLE of honoring ``Structure.frozen_atoms`` (emitting ``%block Geometry.Constraints`` for SIESTA and the ``$freeze`` constraints file for PySCF/geomeTRIC), but /api/build/fdf + /api/build/pyscf NEVER applied the .molstruct.json sidecar -- so the user's /modify freeze list silently never reached render_fdf / render_script.  Spectra had the ``_apply_sidecar_if_possible`` hop; Build did not.  Fixed: both Build endpoints accept ``structure_path`` and call the same helper right after parsing.  JS Generate handlers now read ``projects.getCurrentFile()`` and include it as ``structure_path`` in the POST body.  Integration test class ``TestBuildSiestaHonorsSidecarFrozenAtoms`` walks the full pipeline (sidecar write → POST /api/build/fdf → assert %block Geometry.Constraints + position 1-based indices).  (2) ``_check_siesta_spin_polarized_needs_spin_total`` exited early on ``spin_total is not None``, including ``spin_total=0.0`` -- the exact case the check exists to catch (zero net spin on a d/f shell is the propor IMAX=0 trigger).  Now fires on ``spin_total is None OR float(spin_total) == 0.0``.  (3) ``suggest_spin_total`` recommended 1.0 for Cr / Mo / W / Gd / lanthanides -- all of which ARE in ``OPEN_SHELL_METALS`` so the check fires but the recommendation was wrong (Cr d⁴ HS wants 2S=4; Gd 4f⁷ wants 2S=7).  Expanded ``_SPIN_TOTAL_DEFAULTS`` to cover full first-row d-block (Sc/Ti/V/Cr/Mn/Fe/Co/Ni/Cu) + 2nd/3rd-row TMs (Mo/Ru/Rh/W/Re/Os/Ir/Pt) + lanthanides (Ce..Yb at free-ion Hund's-rule 2S) + selected actinides (U/Np/Pu).  226 tests green. | The prior commit fixed half of the three-stage contract (emitters CAN honor frozen_atoms) but skipped the data-loading hop that makes that capability reachable from the UI -- the same "middle layer not tested" failure mode that produced the SpinPolarized v5 incident.  The integration test (POST through the actual Flask endpoint, assert on the emitted FDF body) is what would have caught the gap at commit time.  The propor and suggest_spin_total fixes are responses to the 2026-05-25 fresh-eyes review explicitly flagging these as latent: the validator's own check was missing exactly the case it was added to prevent, and the suggested-value table covered only half of the open-shell metals it was suggesting values for. |
| 2026-05-30 | **Runner ``--continue`` + ``-runN`` output series (SIESTA + PySCF).**  User directive 2026-05-30: "due to the converging problem etc, we have the need to ask the script to continue trying another time. but currently we have to re-create a new script with a new name etc, and this is inconvenient.  can we have an option --continue ... so we don't have to re-create the script again ... we want it to automatically track the run times. for example, the first run would result in an output file with a suffix -run0, and continue runs will not overwrite this but continue to generate -run1, -run2 etc."  Implemented via three pieces in ``runwrap.py``: (1) a SHARED ``_continue_force_args_parser`` helper that strips ``--continue`` / ``-c`` and ``--force`` / ``-f`` from $@ before engine-specific args (-np for SIESTA) parse the rest; (2) a SHARED ``_run_index_resolver`` helper that scans ``{basename}-run*.out``, picks the next free N, exports ``$_out_file`` to ``{basename}-runN.out``; (3) every reference to ``{basename}.out`` in the SIESTA + PySCF wrappers replaced with ``$_out_file`` (launch redirect, propor grep, banner, completion message).  Semantics locked 2026-05-30: first run produces -run0; ``--continue`` advances to max(existing)+1; without --continue, refuse to overwrite an existing -run0 unless ``--force`` is passed (and even then the prior file is NOT deleted -- only the run-index sequence resets so SIESTA's redirect can clobber the existing -run0.out).  PySCF chkfile auto-continuation: ``pyscf/input.py::render_script`` now emits a startup shim that ``if _os.path.exists(_chk_path) and _os.path.getsize(_chk_path) > 0: mf.init_guess = "chkfile"`` -- so ``bash job.run.sh --continue`` warm-starts from the saved density.  SIESTA continuation is already free because the generator defaults ``use_save_dm / use_save_cg / use_save_xv = True`` -- SIESTA auto-reads .DM / .CG / .XV on the next run.  Tests: 13 new in ``tests/test_runwrap.py`` covering first-run, --continue, max+1, refuse-without-flag, --force-overwrites, short ``-c`` form, defensive "no prior run" fallback, combined ``--continue -np 8``, ``-h`` lists new flags, PySCF wrapper redirects via $_out_file + supports --continue, PySCF refuses-overwrite.  2 new in ``tests/test_pyscf.py`` pinning the chkfile-continuation shim.  Test helper ``_emit_truncated_wrapper`` strips conda activation + truncates after the resolver so bash-level checks run without conda or the engine binary.  Backward compat: existing ``{basename}.out`` files in old projects are untouched + ignored by the resolver (only -runN.out files affect index counting). | The user's hemeC stage3 SCF_NOT_CONV scenario is the canonical case: re-running with a different ``-np`` (the wrapper already supported that) only fixes the propor crash; SCF non-convergence is a SEPARATE class of failure that just needs MORE iterations from a better starting density.  --continue closes that gap with zero new state machine (the resolver is ~25 lines of bash; the chkfile shim is 4 lines of Python).  Strict refuse-without-flag was chosen over silent advance because the user's mental model is "this script produces one result"; silently bumping N on every invocation would surprise users who weren't trying to continue.  --force is the escape valve.  The 100-line truncated-wrapper test helper is the trick that lets us actually exercise the bash logic in CI without needing the conda envs or SIESTA / Python binaries -- we run bash on the args-parsing portion only and inspect $_out_file. |
| 2026-05-29 | **SIESTA parser: detect non-convergence / fatal-error exit status (#179).**  Per user directive 2026-05-29: "in previous runs when script ended with not-converging error, the result could not tell that the script has exited with an error. so i think the parser should be more intelligent to detect the finishing/exit status as well."  Pre-existing parser was binary (``>> End of run`` -> finished, else ongoing) -- no error path.  ``Trajectory.run_state`` / ``error_message`` slots were already plumbed end-to-end (parser -> JSON -> /results badge with red "Error" styling); the parser just never populated the error case.  This commit adds **4 fatal-marker rules** (``siesta: ERROR`` / ``propor: ERROR`` / ``Stopping Program from Node`` / ``siesta died``) that fire substring-matched + case-insensitive in scan state, plus **2 SCF-convergence flag rules** (``SCF Convergence by ...`` -> last_scf_converged=True; ``SCF did NOT converge`` / ``SCF_NOT_CONV`` -> False).  Priority rule: fatal > finished > error-by-SCF-strict > ongoing.  At EOF, if run_state is still "ongoing" AND the LAST SCF block did not converge, we flip to "error" with a synthetic message naming the failure mode (the user's reported scenario: SIESTA truncates on SCF_NOT_CONV without an explicit fatal marker).  ``>> End of run`` does NOT downgrade a prior "error" -- once error_message is set it persists.  ``scf_converged`` matcher requires the ``by`` keyword to guard against accidentally matching a diagnostic line that mentions both "SCF Convergence check" and "did NOT converge".  First-fatal-wins for error_message (subsequent crashes usually cascade from the original cause).  16 new tests at ``tests/watch/test_siesta_parser_exit_status.py`` -- baselines, all 4 fatal markers individually, case-insensitivity, first-fatal-wins, end-of-run-preserves-error, strict SCF EOF check, mid-relax-recovers semantics, the converged-matcher "by" tightening, and a JSON round-trip pinning the wire shape.  Total watch-parser tests: 146 green. | The user's badge was showing "Ongoing" on a SCF_NOT_CONV'd run -- the most confusing failure mode because the file looks similar to a still-running file (no clean-exit marker).  The fatal-marker list mirrors the wrapper's grep heuristic in ``runwrap.py`` (line ~495) so the live wrapper + the post-mortem parser surface the same set of failures.  Strict SCF-EOF policy was chosen over liberal ("any non-convergence anywhere = error") because relaxation runs commonly hit non-converged intermediate SCF steps that the next geometry update recovers from; the LAST block is what determines if the calculation actually failed.  Per-step "this frame's SCF was bad" highlighting is deferred (it would expand the Frame schema; user can revisit if the binary good/error badge isn't enough). |
| 2026-05-29 | **SIESTA parser: rule-based section dispatch + case-insensitive markers (#171).**  Closes the user's standing 2026-05-28 directive: "detection of when a section starts, what that section is about and how the data are printed should be basic capability. the detection of names should be immune to capitalization, small spelling differences etc.  we should have a smart parser that will be defined by rules.  this could be designed as a state machine structure that can be flexible and easy to change if needed."  New ``molbuilder/parsers/_rules.py`` carries three primitives -- ``SectionRule`` dataclass (name + start matcher + optional on_start + optional consume + alias list), three matcher builders (``starts_with_ci`` / ``contains_ci`` / ``any_of``), three driver sentinels (``CONTINUE`` / ``END_SECTION`` / ``END_BUBBLE``).  ``parsers/siesta.py`` ports the section dispatch from a 220-line ``if`` chain into a 75-line rule table + 30-line state-machine driver, with **8 SectionRules** (cell, end_of_run, coords, e_ks, forces, scf_header, scf_data, max_force) closed over the parser locals via ``nonlocal``.  Section markers are now case-insensitive end-to-end -- ``OUTCOOR:`` / ``Outcoor:`` / indented ``  outcoor:`` all parse identically to the canonical ``outcoor:``; same for outcell / E_KS / Atomic forces / iscf / End of run.  ``END_BUBBLE`` sentinel preserves the pre-refactor fall-through semantics: a line that ends a multi-line section but ITSELF starts another section (e.g. a torn outcoor block at EOF immediately followed by ``>> End of run``) gets re-fed through the rule list instead of being dropped.  Scope locked 2026-05-29: SIESTA-only port; case-insensitive only (no Levenshtein fuzz -- typos would invite false positives on comment lines); per-rule alias list (a rule's matcher can OR multiple equivalent headers via ``any_of``).  Future ORCA / NWChem / Gaussian text parsers reuse ``_rules.py`` unchanged.  Tests: 16 unit tests on the rule primitives + 12 end-to-end SIESTA cases pinning case-insensitivity + END_BUBBLE escape hatches.  All pre-refactor SCF / runtime-info / coords / forces / max-force / lattice tests stay green. | The 2026-05-28 Level-3 fix made SCF *columns* header-driven; this commit extends the rule philosophy to *section boundaries* -- the next step in "the parser is intelligent about output structure, not hard-coded against one SIESTA version's exact wording."  The split between rule-based section detection and orthogonal runtime-info regex probing is deliberate: section markers benefit from declarative rules (they evolve across SIESTA versions; they need case-insensitivity), runtime-info markers don't (they're specific lines that never cross section boundaries; an inline regex probe is simpler and faster).  Rejecting Levenshtein: a parser that accepts typos in section headers risks treating a comment line that's one letter from a real marker as a section boundary; the realistic SIESTA version-skew case is capitalisation, not spelling.  Three-stage rollout option was rejected -- the abstraction is small enough that landing it together with the SIESTA port keeps a single reviewable diff. |
| 2026-05-28 | **#165 audit deferrals: cell-volume sanity floor + MeshCutoff production floor + Makov-Payne notice (correction implementation deferred).**  (1) Cell-volume sanity check (``siesta/input.py``) now raises if the cell volume drops below ``n_atoms × 1.0 Å³``; the previous threshold (``vol < 1.0``) caught only the absolute pathological case and missed the realistic failure mode (unit confusion: 10×10×10 *nm* misread as 10×10×10 Å gives a too-small cell for a many-atom system).  Error message names the resolved volume + the per-atom floor + the likely causes (unit confusion / coplanar lattice vectors / typo).  (2) MeshCutoff: slider lower bound raised 50→100 Ry (anything below 100 Ry is unphysically coarse for PAW + DZP); validator emits a ``warn`` Issue for ``100 ≤ MeshCutoff < 150 Ry`` flagging production-grade as 200–300 Ry (400+ Ry for tight-basis).  Help text updated.  Double-warn with the metadata-range rule avoided by gating the new rule on the ``[100, 150)`` band.  (3) Makov-Payne **notice** (correction itself deferred per user 2026-05-28: "give a notices there but no implementation so we keep this open"): when ``net_charge ≠ 0``, the validator emits a ``warn`` Issue + the FDF emits an inline comment block near the ``NetCharge`` line, both naming the artefact (E_bias ~ q² × α / (2 × L × ε_r) ~ 0.5–1.5 eV typical), the paper (PRB 51, 4014, 1995), the use cases that REQUIRE handling it (redox potentials / pKa / deprotonation / charged binding), the use cases that can ignore it (single-point on a fixed cell where the artefact cancels), and that molbuilder does NOT auto-apply the correction.  Workarounds named in the notice: switch to a neutral analogue, extrapolate via supercell-size sweep, or post-process by hand.  Future work: implement the actual correction as an opt-in flag once the formula+ε_r workflow is designed.  Tests: 4 mesh_cutoff boundary tests (5/50/100/120/150/300), 3 cell-volume tests, 3 Makov-Payne validation tests (positive q / neutral / negative q), 2 Makov-Payne FDF emission tests (charged emits block / neutral does not). | The user's standing directive ("each task should be followed by holistic and thorough review without presumptions; tests should be evaluated in their coverage, thoroughness and depth") shaped the order: smallest-blast-radius first (cell-vol arithmetic + slider knob), then the more subjective MeshCutoff-band warning, then the open-capability Makov-Payne notice.  The notice-without-correction split is deliberate: the correction needs ε_r + Madelung-α from the user and a workflow design that isn't ready; surfacing the artefact in the artifact the user is most likely to read (the .fdf) + the place they validate before submit (the validation pass) closes the silent-failure mode without committing to a half-baked implementation.  Empirically validated: 2258 tests pass / 18 skipped / 0 failed across full non-Playwright sweep. |
| 2026-05-24 | **SpinPolarized v4 emission + engine_key metadata + frozen_atoms emitter capability.**  (1) SIESTA 5.4.2's v5 unified ``Spin polarized`` parser path does NOT subsequently read auxiliary ``Spin.Fix`` / ``Spin.Total`` keys -- molbuilder was emitting all three but the user's hemeC-dithiol run aborted at ``propor: ERROR: IMAX = 0`` because Spin.Total never reached the initial-DM constructor.  Empirically verified by hand-editing the user's .fdf to use v4 ``SpinPolarized .true.`` instead and diffing fdf.<timestamp>.log: v4 form -> Spin.Fix=T + Spin.Total=4.0 honored; v5 form -> both at default.  ``siesta/input.py`` flipped to emit v4 form; v5 is marked deprecated in the manual but is fully accepted in the parser AND triggers the auxiliary reads we depend on.  (2) Source-of-truth UI tag: ``engine_key`` metadata added to all 47 SIESTA + 48 PySCF dataclass fields -- the exact engine keyword each form field writes (or ``(molbuilder: ...)`` marker for preprocessing / wrapper / filename knobs that don't reach the engine).  JS ``engineKeyBadge()`` renders a ``<code class="schema-engine-key">`` next to each label with the keyword text.  Solid border for engine keys, dashed italic + ``.is-molbuilder-only`` class for molbuilder-only markers so the user knows NOT to search the SIESTA / PySCF manual for those.  (3) Generate/Save split: "Generate" is now render-only (validate + render + preview pane, no disk writes); "Save to current dir" is the explicit commit-to-disk action (writes .fdf/.py + pseudos + .run.sh wrapper).  Live psml_lib caption shows the resolved absolute path as the user types.  After Save, psml_lib field rewrites to a path RELATIVE to dest_dir (privacy + survives copying the whole projects/ tree elsewhere).  ``_apply_sidecar_if_possible`` hop NOT yet added to /api/build/fdf (caught by the 2026-05-25 review).  (4) ``propor: ERROR: IMAX = 0`` preflight: when ``spin_polarized=True`` + ``spin_total is None`` + open-shell metal in structure, validator emits an ERROR-severity Issue naming the failure mode + proposing a starting Spin.Total value from a per-element table (Fe=4, Mn=5, Co=3, Cu=1, Ni=2) + listing alternatives to sweep.  Catches the failure in molbuilder before SIESTA's 30-second init-then-abort. | The v4-vs-v5 SpinPolarized discovery was empirical (diff fdf.log after a 10-second test run); the manual would have said v5 is the right form.  The lesson the user demanded: when you claim a thing works, run it.  The ``engine_key`` UI tag is the explicit answer to "user is unaware of which form fields actually make a difference" -- every field's badge tells the user the SIESTA / PySCF keyword (or "this is molbuilder-only") so the UI -> generated script -> engine-manual chain is auditable.  Generate/Save split removes the prior surprise that "Generate" was secretly writing 3-7 files into the sidebar dir; portability rewrite removes the ``/home/<user>/...`` leak from form state.  The propor preflight closes the actionable-error gap for the most-likely-to-hit failure on open-shell metal workflows. |
| 2026-05-22 | **Runtime info registry + SIESTA .out parser + threading knobs.**  (1) New ``molbuilder/runtime_info.py`` central registry of "runtime-info" keys (CPU count, MPI ranks, OMP threads, BLAS lib, max memory, GPU info) that BOTH SIESTA wrappers AND PySCF / spectra scripts emit at run time -- single source for the cross-cutting "report what you ran with" recipe.  PySCF + spectra scripts inline ``emit_runtime_info_capture_lines`` (Python printing).  SIESTA wrappers emit comments into the .fdf (``# runtime.omp_threads_requested: N``) since SIESTA itself can't introspect at runtime.  (2) ``molbuilder/parsers/siesta.py::SiestaOutParser`` gained a runtime_info extraction pass: reads the wrapper-emitted comments back from the .fdf so /results trajectory inspector can show "this run used 4 ranks, 1 OMP thread, OpenBLAS" inline next to the energy curve.  (3) ``SiestaConfig`` gained ``mpi_np`` / ``omp_threads`` / ``max_memory_mb`` form fields in the "Parallel execution" section.  Wrapper consumes them at render time (``mpirun -np N siesta``, ``export OMP_NUM_THREADS=N``, ``ulimit -v MB*1024``).  Default omp_threads=physical_cores/mpi_np (auto-resolved); explicit zero is rejected.  (4) /spectra promoted runtime tuning from "Generator-only" to a first-class form section; threading numbers flow through ``lib.num_threads()`` + ``OMP_NUM_THREADS`` at the top of the generated script. | Runtime-info-as-data was scattered: PySCF scripts emitted some, SIESTA wrappers emitted some, neither talked to the other, and /results had no way to show "this run was 4-rank MPI with 1 OMP thread per rank".  The registry collapses the recipes into one module + one set of emission helpers.  The .fdf-comment trick (SIESTA can't introspect at runtime, so the comments ARE the runtime info) is what lets the .out parser recover the launcher's intent later -- the badges I added in 2026-05-24 had to be updated in 2026-05-26 to stop falsely claiming "not in .fdf" because the comments are technically there. |
| 2026-06-02 | **Doc restructure: hierarchical master + archive (task #196).**  ``docs/design.md`` becomes the master holding cross-cutting principles + cross-cutting decisions + § 0 index pointing at every subsystem doc.  Each subsystem doc owns its own decisions log; new contracts that span subsystems (uniform envelope, AbortSignal, cache:no-store default, pageshow refresh, sole-source-of-truth rule) land here with backlinks.  Two NEW protocol docs ([`runtime-registry.md`](protocols/runtime-registry.md), [`inspector-registry.md`](protocols/inspector-registry.md)) and two NEW tab docs ([`tabs/build.md`](tabs/build.md), [`tabs/results.md`](tabs/results.md)) created.  Three legacy docs (`REVIEW_FINDINGS.md`, `protocols/watch-api.md`, `tabs/watch.md`) moved to ``docs/archive/`` with the date-prefix convention (`YYYY-MM-DD-<name>.md`); every still-live invariant migrated to the canonical replacement BEFORE the archive move.  Pre-restructure audit dropped ~15 substantive contracts to one-line summaries; review pass restored them (see ``protocols/web-api.md`` § 13 for the rewrite-history list).  ``docs/README.md`` slimmed to a thin pointer at ``design.md`` § 0 — maintaining two indexes invites drift; one is enough. | The folder split is by KIND-OF-CONTRACT (protocols/ = how parts talk to each other; types/ = shape of values; engines/ = what we WRITE; tabs/ = per-UI specs), not by feature.  That's the load-bearing reason every spec lands in exactly one place and the sole-source-of-truth rule is enforceable — each contract has one canonical home; design.md holds cross-cutting principles with backlinks.  The archive principle ("substance of every still-live contract must be migrated to the canonical doc BEFORE the archive move") is the lesson from the restoration pass: an archived file is a snapshot for historical reference; it must not be the only place a live invariant lives. |
| 2026-06-02 | **Stale-state refresh on tab re-entry — established as cross-cutting principle (tasks #192 + #194).**  Every cached-data UI surface that caches server state across page navigations MUST refresh on `pageshow` + `visibilitychange→visible`.  Without these handlers a bfcache restore (Chromium/Firefox default for back/forward navigation) leaves the user staring at the cached snapshot from the previous visit -- file generated in another tab never appears, log file appended on disk never re-fetches.  Manifestations closed in this session: (#192) /results file-picker stale dropdown — fixed via `_forceRescan` + `cache: "no-store"` on the listing fetch; (#194) trajectory inspector 15 s poll paused by bfcache — fixed via immediate `pollOnce()` on the events; (#194) spectra inspector mount-once with no refresh — fixed via `loadByPath()` on the events guarded by `state.results !== null`.  Pattern documented in [`protocols/inspector-registry.md`](protocols/inspector-registry.md) § 4 (per-inspector implementation table) and the canonical test recipe in [`protocols/playwright-tests.md`](protocols/playwright-tests.md) § 9.6 (second-visit + external-change). | The /results bug was filed by the user with a precise workaround ("manually get out of the directory and get back in"); audit traced it to a `lastScannedDir !== newDir` guard that bailed when the dir was unchanged but the disk state HAD changed.  The audit also surfaced that browser HTTP cache served the previous response on identical URLs even when the rescan DID fire — both halves had to be fixed.  Establishing this as a named principle prevents the next "subscriber-on-state-change + no refresh-on-revisit" bug from shipping silently; new cached-data inspectors now have a documented contract to inherit. |
| 2026-06-02 | **Uniform `{ok}` envelope is mandatory on every endpoint (task #187).**  Every Flask blueprint route returns `{ok: true, ...}` on success and `{ok: false, error: "..."}` on failure.  HTTP status codes classify (200 / 4xx / 5xx) but the body shape does NOT depend on status — the JS apiX wrappers (`lib/projects/api.js`) branch on `body.ok`, not on `Response.ok`.  Closed the audit gap in `web/blueprints/selection.py`: `_bad_request` returned `{error: msg}` without `ok: false`; `/atoms` / `/eval` / `/toggle` success returns dropped `ok: true`.  JS worked by accident (branched on HTTP status), but a future consumer that reads `body.ok` would have broken silently.  Pinned by `tests/test_selection_blueprint.py::TestUniformEnvelope` (6 tests covering success + parametrised error paths).  Now codified as a hard rule in [`protocols/web-api.md`](protocols/web-api.md) § 1.1. | The 2026-05-25 sidebar M3 work introduced the principle for the sidebar surface; the audit found ONE blueprint (selection) had drifted from it.  Naming the principle in the master decisions log AND in web-api.md § 1.1 establishes "every blueprint" — not just sidebar consumers — as the contract scope.  A regression that swallows the `ok` field on a new endpoint is now catchable by inspection of web-api.md before code review, instead of needing JS consumer testing to surface. |
| 2026-06-02 | **`cache: "no-store"` is the default for every projects.* live-data GET (task #193).**  The central `_fetchEnvelope` wrapper in `lib/projects/api.js` sets `cache: "no-store"` on every fetch unless the caller overrides.  This is the second-half fix for the #192 stale-dropdown bug class generalised: every live-data GET (`apiList` / `apiRead` / `apiReadRange` / `apiStat` / `apiRoots`) now reaches the server on a same-URL revisit instead of serving the browser HTTP cache.  Override path preserved (a future caller that legitimately wants browser caching can pass `cache` explicitly).  `cache: "no-store"` is a no-op for POST/DELETE/PUT (browsers never cache non-GET responses), so the default applies uniformly via the central wrapper without affecting mutator semantics.  Pinned by 7 node-driven tests in `test_projects_api_envelope_js.py::TestNoStoreCache`.  Now codified in [`protocols/web-api.md`](protocols/web-api.md) § 1.3. | The file-picker fix (#192) added `cache: "no-store"` to its OWN raw fetch but the underlying defect lived in the central wrapper.  Fixing the central wrapper immunises every existing AND future projects.* live-data consumer in one place — much stronger than patching each caller as they're discovered.  The principle ("live-data GETs default no-cache; immutable-blob GETs override explicitly") is now the contract for any future API wrapper. |
| 2026-06-02 | **`Structure.frozen_atoms` and `Structure.regions` carry through every copy / transform / concat (task #186).**  Pre-fix `Structure.copy() / translated() / centered() / concat()` and all five `modify.py` ops (`delete_atoms / add_atom / orient_along_axis / rotate_around_axis / add_electrode_slab`) constructed a fresh Structure without the two transport-metadata fields, defaulting them to empty via `default_factory`.  End-to-end: the user picks frozen atoms in /modify, does ANY modify op, submits to /build — the emitted FDF has ZERO frozen atoms despite a perfectly-formed sidecar on disk.  Eight droppers fixed in one commit.  `modify.delete_atoms` needed reindex bookkeeping (`_reindex_transport_metadata` helper) because deleting atom i shifts atom i+1 down; pure-rotation ops carry through verbatim; `concat` adds per-input atom-offset bookkeeping + label-keyed region merging.  Pinned by 14 new tests (`test_structure.py::TestStructureTransportMetadataCarryThrough` + `test_modify.py` carry-through suite). | Violates the [three-stage contract principle](#) (UI → config → script, no silent absorption) at the in-process layer — the principle had been thought of as a "Python-to-engine" contract but the audit surfaced it ALSO applies inside Python between Structure transforms.  Establishing this as a documented invariant means future modify ops can't quietly add to the list — they MUST account for transport metadata, either by passing through or by remapping if they change the index space.  See [`types/structure.md`](types/structure.md) for the per-op contract table (planned). |
| 2026-06-02 | **Source inspector v2: paginated range-read viewer (task #119).**  The v1 source inspector called ``ctx.readFile`` with a 16 MB cap and dropped the whole text into one ``<pre>`` -- worked for tiny configs (a few KB ``.fdf`` / ``.json``), slow + memory-heavy for MB-scale logs, and hard-failed at the 16 MB ceiling (which the user hit on a multi-stage SIESTA ``.out``).  Replaced with a server-side range-read endpoint + client-side chunked pager.  (1) ``GET /api/files/read_range`` (``molbuilder/web/blueprints/files.py``) reads ``max_bytes`` (default 256 KB, ceiling 16 MB) at ``offset`` (negative = from EOF for tail reads).  UTF-8 boundary trimming walks back up to 3 bytes if the chunk edge cuts a codepoint -- callers always receive valid UTF-8 even on arbitrary byte windows.  Offset exactly at EOF returns an empty chunk with ``eof: true`` so the client's "are we done?" check is one comparison.  15 tests in ``tests/test_web_files.py::TestFilesReadRange`` pin the contract: default head read, explicit offset+max_bytes, EOF semantics, negative offset (tail) + clamping when ``-offset`` exceeds file size, error envelopes for missing / directory / past-EOF / non-UTF-8 / bad args, file_size stability across calls.  (2) ``molbuilder/web/static/lib/inspectors/source.js`` rewritten from single-shot to chunked.  ``PAGE_BYTES = 256 KB`` matches the server default so a future server-side bump can't silently change client behaviour.  Initial mount loads the first PAGE_BYTES at offset 0; the inspector listens on a ``.source-scroller`` wrapping div (NOT the inner ``<pre>``, so the scroll math reads from one element with stable dimensions), and when ``scrollHeight - scrollTop - clientHeight < NEXT_PAGE_TRIGGER_PX (200 px)`` it auto-loads the next chunk via a ``mode === "head"`` forward fetch.  Chunks ``textContent +=`` onto the existing pre (one large text node, not one per chunk -- keeps DOM count O(1) at the cost of a single string concat per page; right trade-off for the 1-100 MB range the inspector targets).  A new "Jump to end" button switches into ``mode === "tail"``, issues ``offset = -PAGE_BYTES`` to grab the last chunk, replaces the body, scrolls to bottom, and disables further auto-load.  Head/tail mixing within one mount is intentionally NOT supported -- supporting it would need gap detection + a "..." separator UI that's a v3 problem.  (3) Status surface ("Showing head/tail: N KB of M KB (P%)") rendered into ``.source-controls`` (flex row in the card header).  ``[disabled]`` state on Jump-to-end mirrors EOF correctly so a small file that fits in one head page disables the button on first paint.  (4) CSS: new ``.source-controls`` / ``.source-scroller`` / ``.source-status`` / ``.source-jump-end`` rules in ``molbuilder/web/static/results/style.css``.  Overflow + ``max-height: 60vh`` moved from ``.source-body`` to ``.source-scroller`` (the new contract for inspector scrollers: the WRAPPER owns overflow, the inner ``<pre>`` is plain text -- cleaner DOM, simpler scroll math).  (5) ``docs/protocols/projects-sidebar.md`` § 12 capability table updated: seven primitives → eight (the range-read endpoint is the eighth).  Tests: 10 Playwright tests in ``tests/test_source_inspector_e2e.py`` cover initial chunk landing at byte 0 + correct length, partial-progress status text for big files, Jump-to-end button enabled when more remains, small-file-fits-in-one-page eof+disabled, scroll-driven next-page append, mode stays "head" after one auto-load, click-Jump-to-end loads tail + contains the actual last line of the fixture + scrolls to bottom + disables button, and dispose() empties the host.  All 116 / 34 / 10 tests green across ``test_web_files.py`` / ``test_inspector_registry_e2e.py`` / ``test_source_inspector_e2e.py``. | The v1 ceiling was the obvious failure mode (user hits 16 MB, gets an error); the latent v1 failure was DOM weight -- a single 5 MB ``<pre>`` made the /results tab visibly janky on selection / scroll even before the cap fired.  Chunked reads cap both: server returns 256 KB even if asked for more; client only builds DOM for what the user has actually scrolled past.  Negative-offset tail reads were chosen over a separate ``/api/files/read_tail`` endpoint to keep the surface to ONE primitive -- the offset convention (positive from start / negative from end) is the same pattern the Range header carries, and one primitive means one validation path + one test class.  Lock the overflow + max-height on the WRAPPER (not the ``<pre>``) was the load-bearing CSS detail: putting them on ``.source-body`` made the inner element the scroll root, which (a) made the scroll-math read off a node that grows on every chunk append, (b) caused intermittent "stuck at bottom" jumps as the new text was committed.  The wrapper-owns-overflow pattern matches what the trajectory inspector already does for its frame strip + is now the contract for any future scrolling inspector.  Lifting overflow off the ``<pre>`` also surfaced that the v1 60vh cap was hiding text on tall viewports -- the v2 cap stays per-inspector but is now overridable by the host page if the inspector grows into a different layout slot.  True virtual scrolling (line-index + viewport sliver) was considered and rejected for v2: significantly more complex (line-break tracking, viewport-windowed rendering, partial-line rendering at edges), and the primary use case (multi-MB SIESTA / molwatch logs) is well-served by chunk loading.  Revisit if a >100 MB use case arrives. |
| 2026-05-21 | **Terminology unification: "frozen" is canonical for held-in-place atoms.**  "Fixed atoms" and "frozen atoms" mean the same thing in computational chemistry; molbuilder previously mixed both (UI sections said "Frozen atoms", but the Python dataclass field was `Structure.fixed_atoms`, the sidecar key was `"fixed_atoms"`, the HTTP target was `"fixed_atoms"`, and `SpectraConfig.fixed_{elements,residue_names,indices}` carried `fixed_`).  Renamed across the stack to use **"frozen"** exclusively: `Structure.frozen_atoms`, sidecar `frozen_atoms` key, HTTP target `"frozen_atoms"`, atom-list response `is_frozen`, JS panel `FROZEN_TARGET` / `FROZEN_TAG_LABEL` / `tag-frozen`, viewer-adapter `FROZEN_MARKER`, `SpectraConfig.frozen_{elements,residue_names,indices}`, `SpectraResults.frozen_atom_idxs`, generated script vars `FROZEN_ELEMENTS` / `FROZEN_INDICES_USER` / `FROZEN_ATOM_IDXS` (`FREE_ATOM_IDXS` kept as the complement).  Sidecar schema bumped v2 → v3; the parser refuses v1/v2 (`"fixed_atoms"` key) with a clear error rather than silently coercing — per user direction, no backward compatibility, internal data structure unifies to one.  Re-export sidecars from /modify to produce v3 files. | "Frozen" is the more widely accepted scientific term in the spectroscopy / quantum-chemistry literature ("frozen-atom partial Hessian", "frozen-core approximation"), which is the primary user-facing context for this concept in molbuilder.  Unifying the dataclass field, the sidecar key, the HTTP wire format, and the UI labels removes the parallel-metadata cognitive load the old mix imposed and makes the dataclass-as-source-of-truth rule actually true again (Principle: "no parallel metadata in CLI/web layers").  The v3 schema bump is the explicit boundary: a sidecar either uses `frozen_atoms` (v3, current) or doesn't load.  Hard cut beats silent coercion because the coercion's "1.5 → 1" semantics had already produced a class of latent bugs (see review history); refusing the load forces a re-export which surfaces the rename to the user once instead of confusing them later. |

---

## Module init contract — runtime registry + ready Promises

> **Founding directive (2026-05-21).** *"can we structurally and
> logically make sure the sequence how these modules are init and
> connected to each other? is there a data structure or a global api
> that we can integrate?"*  The answer is yes — a single tiny
> registry replaces the previous "grab `window.molbuilder.foo` and
> hope script-tag order is right" pattern.  Any code that exposes a
> global namespace MUST use it.  Reviewers reject polling /
> guess-and-check patterns by reference to this contract.

### The problem

Classic `<script>` tags execute in document order.  `<script
type="module">` tags execute AFTER all classic scripts + the
initial parse complete.  Mixing them in one page means a classic
script that does `(window.molbuilder || {}).projects` at IIFE time
sees `undefined` because the projects-sidebar module hasn't
initialised yet.  We hit this bug for real in Build's sidebar
wiring: picking a `.pdb` did nothing because the consumer
captured `_proj = undefined` at IIFE time and silently never
re-checked.

### The contract

A tiny runtime module loaded FIRST in every template:

```html
<script src="lib/molbuilder-runtime.js"></script>   <!-- before any other molbuilder script -->
```

Exposes:

```js
window.molbuilder.runtime.register(name, api)         // producer: "I'm ready"
window.molbuilder.runtime.whenReady(name) -> Promise  // consumer: await readiness
window.molbuilder.runtime.get(name)                   // sync peek (debugging)
window.molbuilder.runtime.listRegistered()            // diagnostics
window.molbuilder.runtime.listPending()               // diagnostics: who's waiting for whom
```

**Producer side** (any module that exposes a global namespace) adds
ONE line to its IIFE:

```js
window.molbuilder.runtime.register("selection.store", store);
```

**Consumer side** (anyone that depends on another module's API):

```js
window.molbuilder.runtime.whenReady("projects").then((proj) => {
    proj.onChange((sel) => { ... });
});
```

No more `if (window.molbuilder && window.molbuilder.foo)`.  No more
`setTimeout(check, 100)` polling.  No more "this code happens to
work because the script tags are in the right order."

### Module name registry

Names are flat, dotted, lowercased.  Current modules:

| Name | Module file | Notes |
|---|---|---|
| `viewer` | `lib/mol-viewer.js` | 3Dmol viewer factory |
| `style` | `lib/mol-style.js` | style-spec builder |
| `fmt` | `lib/mol-format.js` | formula formatter |
| `formSchema` | `lib/form-schema.js` | dataclass-driven form renderer |
| `projects` | `lib/projects-sidebar.js` | Projects sidebar (file picker) |
| `selection.store` | `lib/selection/store.js` | Selection state + HTTP |
| `selection.panel` | `lib/selection-panel.js` | Selection DOM panel |
| `selection.viewerAdapter` | `lib/selection/viewer-adapter.js` | Selection 3Dmol overlay |
| `modify.handle` | `modify/viewer.js` | per-tab embed handle (NOT raw 3Dmol) |
| `modify.loadXyzText` | `modify/viewer.js` | XYZ loader callback |
| `inspectors` | `lib/inspectors/registry.js` | Inspector dispatch |

Adding a new module-with-a-global: pick a name in this scheme, call
`register()` at the END of the IIFE, add a row to this table.

### Backward compatibility

The registry is **additive**.  Producers also keep their existing
`window.molbuilder.<foo> = api` assignments — unmigrated consumers
keep working without change.  Migrate at your own pace; the
registry is the new contract but the old globals stay as escape
hatches.

### Diagnostics

From devtools:

```js
window.molbuilder.runtime.listRegistered()
// e.g. ["modify.loadXyzText", "modify.viewer", "projects",
//       "selection.panel", "selection.store", "selection.viewerAdapter"]

window.molbuilder.runtime.listPending()
// names with waiters but no registration yet -- diagnoses
// "consumer hung forever" bugs.
```

### Migration log

| Date | What | Why |
|---|---|---|
| 2026-05-21 | Land `lib/molbuilder-runtime.js`; register `projects`, `selection.store`, `selection.panel`, `selection.viewerAdapter`, `modify.viewer`, `modify.loadXyzText`.  Migrate Build's `viewer.js` sidebar wiring + Spectra's `core.js` onChange subscription to `whenReady`. | Build's `.pdb` sidebar pick was silently doing nothing because the consumer captured a `window.molbuilder.projects` snapshot at IIFE time, before the deferred `type="module"` sidebar script had initialised.  Polling fixes the symptom; the registry fixes the structure. |

### Tests

  * `tests/test_runtime_registry_js.py` (planned) — pure-JS unit
    tests of register / whenReady / list*.
  * `tests/test_modify_e2e.py::test_runtime_modules_registered` —
    smoke-test that the expected modules register on /modify.
  * The `whenReady` migration is regression-tested by the existing
    `test_send_to_build_visible_across_all_op_subtabs` (which
    drives a sidebar pick + waits for the Build load to land).

---

## Sidecar-driven boundary conditions — the three-stage contract

> **Founding guidance (verbatim, 2026-05-21).**  This contract is the
> direct codification of the user's explicit design directives.  The
> quotes below are preserved as the load-bearing principle every
> implementation in this area must satisfy:
>
> > *"fixed atoms will be used in many modeling scenarios so we
> > need to make sure this is reserved [preserved]."*
> >
> > *"we just need to make sure that the script/input generator
> > fully respects the frozen setting because our modeling would
> > require scientifically correctness with the correct assumption.
> > the starting facts/boundary condition of simulation must be
> > explicit, consistent and fully respected from config to actual
> > calculation."*
> >
> > *"when building script/input files the atom information is
> > built into the script/input file. therefore i find option 2
> > [pre-fill the form from sidecar; form is authoritative] to be
> > more solid."*
> >
> > *"UI → config makes sure user's intention is captured
> > correctly, config → script/input faithfully delivers the
> > information and the generator understands what those labels
> > mean and correctly use it for boundary conditions. if labels
> > are not consistent or not recognized, the script should give
> > explicit warning so that the user know there could be an
> > issue. no silent absorption of config."*
>
> Any code in this repository that touches sidecar metadata,
> simulation boundary conditions, or label-driven config flow
> MUST satisfy this contract.  Reviewers reject "silent
> absorption" patterns by reference to these quotes.

The boundary conditions of a simulation (which atoms are frozen,
which regions partition the system, fixed cell vectors when they
exist, …) are user input.  molbuilder routes them through a strict
three-stage contract: **UI → config → script**.  Each stage has an
explicit, testable obligation, and divergence between stages
surfaces as a visible issue — never as silent absorption.

Today the contract is fully implemented for `frozen_atoms` against
the PySCF spectra engine; the design is the template every future
engine + every future label type follows.

### Stage 1 — UI → config: capture the user's intention correctly

The `/modify` selection panel writes `Structure.frozen_atoms` into
the sidecar (`.molstruct.json` v3, key `frozen_atoms`).  When the
user opens `/spectra` against a structure that has a sidecar, the
schema endpoint (`GET /api/build/schema/spectra?structure_path=…`)
reads the sidecar and **pre-fills** the form's "Freeze by atom
index" field with the comma-separated indices.  The user **sees**
what's about to be frozen before clicking Generate.

The form is then **authoritative**.  The user can:

  * leave the pre-fill alone (the script will freeze those atoms),
  * add more indices (script will freeze the union),
  * clear the field (script will freeze nothing — a deliberate
    override).

Pre-fill makes the boundary condition **visible**; the
user-editable form makes it **consistent** (one source of truth at
the moment of Generate).  If the sidecar can't be applied (atom
count mismatch, corrupt JSON), the schema response carries a
human-readable `notice` field rather than silently failing.

### Stage 2 — config → script: faithfully deliver, no silent merge

The script generator (`molbuilder/spectra/pyscf_script.py`) emits
`FROZEN_INDICES_USER = list(cfg.frozen_indices)` as a Python
literal.  Nothing else.  The script does **not** read any sidecar
at run time; it does **not** silently union with `struct.frozen_atoms`
at emit time.  Whatever the user committed in the form lands in
the script verbatim — that's the script's promise.

The runtime `_emit_frozen_mask()` then computes:

```python
_frozen = set(FROZEN_INDICES_USER) ∪ {i : ELEMENTS[i] ∈ FROZEN_ELEMENTS}
FROZEN_ATOM_IDXS = sorted(_frozen)
FREE_ATOM_IDXS   = [i for i in range(N_ATOMS) if i not in _frozen]
```

The partial-Hessian path then operates on `FREE_ATOM_IDXS`.  No
hidden inputs, no engine-private state, no silent extension of the
frozen set.

### Stage 3 — engine understands the labels and warns on what it can't use

The engine's `preflight()` is the contract enforcer.  Two checks
at the engine layer, plus one at the render-endpoint boundary:

**A. Divergence warn — sidecar set vs config set.**
If `struct.frozen_atoms` is non-empty AND
`struct.frozen_atoms ⊄ cfg.frozen_indices`, preflight emits a
WARN-severity `Issue` (where: `config.frozen_indices`) naming
the divergent indices.  The user is told **before** Generate
that the script is about to omit a subset of the sidecar's
frozen atoms.  This catches the case where the user navigated
to `/spectra` before picking the structure (form pre-fill
hadn't run) or where the sidecar was updated in another tab
since the schema was last fetched.

**B. Unrecognized-label notice — labels the engine can't use.**
The selection panel writes both `frozen_atoms` AND `regions`
(e.g. `L-electrode`, `bridge`).  Only `frozen_atoms` is
meaningful to the spectra engine; `regions` are reserved for
the transport engine.  If a structure carries regions and the
user runs `/spectra`, preflight emits an INFO/WARN notice
explaining: "the structure has regions [L-electrode, bridge,
…], which the spectra engine doesn't consume.  These don't
affect the calculation but stay in the sidecar for /transport."
Same shape applies to future engines + future label types:
every label that's NOT understood by the current engine MUST be
named explicitly in a preflight issue.  **No silent absorption.**

**C. Sidecar-failed-to-apply notice — at the render endpoint.**
The render endpoint loads the sidecar (when `structure_path` is
supplied) and applies it to the Structure before preflight runs
— that's what activates checks A + B.  If the sidecar exists
but FAILS to apply (path rejected, malformed JSON, atom-count
mismatch with the pasted XYZ), the endpoint emits a
WARN-severity `Issue` (where: `structure_path`) explaining
that "the form's freeze rules are the sole boundary condition
for this run".  Without this, a stale or wrong-structure
sidecar would silently produce a render with `struct.frozen_atoms
= []` and the user would never know their sidecar didn't flow
through.  Implementation: `_apply_sidecar_if_possible` returns
a notice string instead of silently catching the error.

### Why this matters

Boundary conditions ARE the calculation's starting facts.  A
script that silently freezes more (or fewer) atoms than the user
configured is not a different number — it's a different
calculation.  Scientific correctness requires that the user can
look at the form (config) and the issues panel (engine
understanding) and know exactly what's going to happen.  This
contract gives that:

  * **Explicit**: every relevant label appears in the form or in
    an issue; nothing is implicit.
  * **Consistent**: at the moment of Generate, the form is the
    single source of truth.
  * **Fully respected**: the script does what the form says.

### Extending to new engines / new label types

When adding a new engine (e.g. the future transport engine,
#135) or a new label type (e.g. surface-anchor markers), the
contract requires:

  1. Decide which sidecar labels the engine consumes.  Document
     in the engine module's docstring.
  2. Add a schema-endpoint pre-fill if the label maps to a form
     field (mirror `_seed_frozen_indices_from_sidecar` in
     `web/blueprints/spectra.py`).
  3. Add a preflight divergence warn matching pattern A above.
  4. Add a preflight unrecognized-label notice matching pattern B
     above for every sidecar field the engine does NOT consume.

Tests should pin each: a regression where a future engine
silently absorbs a label, or silently drops one, must fail
loudly.

---

## Package layout

The L1/L2/L3 split below is the as-shipped layout.  Re-export shims
keep external imports stable across the deliberate splits (`config/`
re-exported from `molbuilder.siesta` / `molbuilder.pyscf`,
`builders.backends.*` aliased as `molbuilder.backends.*`).

```
molbuilder/
  # ----- L1: core types -----
  __init__.py              # public API: re-exports L1 types + key L2 verbs
  structure.py             # Structure dataclass + readers / writers
  frame.py                 # Frame + Trajectory dataclasses
  issues.py                # Issue(severity, message, where) + ValidationError
  config/
    __init__.py            # re-exports SiestaConfig, PySCFConfig
    siesta.py              # SiestaConfig
    pyscf.py               # PySCFConfig
  chemistry.py             # element table, masses, valences, H placement
  residues.py              # PDB residue templates + 1-letter parser
  trajectory_log/
    __init__.py
    format.py              # write_initial_preview, molwatch_log_basename
    emitter.py             # MolwatchEmitter (inlined into generated PySCF scripts)

  # ----- L2: domain verbs -----
  peptide.py               # build_peptide
  nucleic.py               # build_dna / build_rna
  smiles.py                # build_from_smiles
  pubchem.py               # build_from_name
  modify.py                # delete / add_atom / orient / rotate / electrode ops
  validation.py            # validate(struct, cfg) -> List[Issue]
  builders/
    backends/
      __init__.py          # is_available(), dispatch()
      _amber.py            # tleap-driven extended chain
      _rdkit.py            # ETKDG embedded conformer
      _threedna.py         # canonical B/A/Z-form helix via fiber
      _common.py
  backends/                # back-compat shim -> builders/backends
  siesta/
    __init__.py            # re-exports SiestaConfig, render_fdf, convert
    input.py               # render_fdf / convert / FDF body builders
  pyscf/
    __init__.py            # re-exports PySCFConfig, render_script, convert
    input.py               # render_script / convert / inlined emitter wiring
  parsers/
    __init__.py            # PARSERS registry + detect_parser + Trajectory legacy adapter
    base.py                # TrajectoryParser ABC; parse() -> Trajectory
    molwatch_log.py
    siesta.py
    pyscf.py
  data/
    README.md              # citations for every numeric value below
    fcc_lattice.json       # supported FCC metals (closed list)

  # ----- L3: surfaces -----
  cli.py                   # click-based; add_dataclass_options bridge
  web/
    __init__.py            # create_app
    app.py                 # Flask app + Blueprint registration + 413 handler
    blueprints/
      _shared.py           # body parsing, issue serialisation, type coercion
      build.py             # /api/build/* routes (molecule, load, fdf, pyscf, preflight)
      modify.py            # /api/modify/* routes (8 endpoints; see tabs/modify.md)
      watch.py             # /api/watch/* routes + directory-mode + multi-stage merge
    templates/
      _app_header.html     # shared header + tab nav partial
      index.html           # Build tab page
      modify.html          # Modify tab page
      watch.html           # Watch tab page
    static/
      viewer.js            # Build viewer
      style.css
      lib/
        tokens.css         # CSS custom properties (one home for theme tokens)
        tabs.css           # top-of-page Build/Modify/Watch nav
        mol-style.js       # shared 3Dmol style-spec builder
        mol-format.js      # chemical-formula renderer
        mol-pick.js        # shared wireframe-halo helper (Modify + Watch)
      modify/{viewer.js, style.css}
      watch/{viewer.js, style.css}

tests/
  conftest.py
  test_structure.py         test_frame.py         test_chemistry.py
  test_residues.py          test_peptide.py       test_nucleic.py
  test_smiles_and_siesta.py test_pyscf.py         test_pyscf_spec.py
  test_validation.py        test_science_gaps.py  test_review_fixes.py
  test_load.py              test_pdb_ter.py
  test_output_correctness.py
  test_molwatch_preview.py  test_molwatch_emitter.py
  test_pubchem.py           test_backends.py
  test_cli.py
  test_modify.py            test_modify_e2e.py    # Playwright E2E
  test_web.py               # Build + Modify Flask
  watch/                    # Watch parser + Flask
    test_registry.py
    test_molwatch_log_parser.py
    test_siesta_parser.py
    test_pyscf_parser.py
    test_api_load.py
    test_app_concurrency.py

docs/
  design.md                       # this file (master: principles, decisions, § 0 index)
  README.md                       # quick pointer to design.md § 0
  protocols/                      # cross-cutting interfaces (HTTP/JS/test/on-disk contracts)
    web-api.md                    #   HTTP /api/* endpoint reference
    projects-sidebar.md           #   sidebar architecture + projects.* API + lock model
    atom-selection.md             #   selection store + .molstruct.json sidecar shape
    selection.md                  #   Python selection rule grammar
    results-tab.md                #   /results dispatch architecture
    runtime-registry.md           #   molbuilder-runtime.js (register / whenReady)
    inspector-registry.md         #   inspector mount/dispose + pageshow refresh
    playwright-tests.md           #   test patterns + anti-patterns
    job-layout.md                 #   on-disk basename + -runN.out convention
    cli.md                        #   click-based CLI surface
  types/                          # L1 data-type contracts (shape of values)
    structure.md                  #   Structure dataclass + frozen_atoms/regions + I/O
    parsers.md                    #   parser-plugin output shape (per engine)
    chemistry.md                  #   element table + charge/spin auto-detect helpers
  engines/                        # per-engine emitter specs (what we WRITE)
    siesta.md                     #   SIESTA .fdf generator
    pyscf.md                      #   PySCF .py generator
    builders.md                   #   build-backend contract (peptide / DNA / RNA / SMILES / name)
  tabs/                           # per-UI-tab specs (subfolders when multi-asset)
    build.md                      #   /build tab — structure-from-input + Generate
    modify.md                     #   /modify tab — atom selection + nanojunction assembly
    results.md                    #   /results tab — registry dispatch + file picker (planned)
    spectra/                      #   /spectra tab (multi-asset)
      spec.md                     #     full spec
      references.bib              #     bibliography
  archive/                        # superseded docs (NOT a source of truth)
    README.md                     #   catalogue of what was archived + why
    YYYY-MM-DD-<original-name>.md
  img/                            # README screenshots

tools/
  capture_screenshots.py    # idempotent README screenshot capture
```

External imports that callers may already use stay valid via
re-exports:

- `from molbuilder.siesta import SiestaConfig, render_fdf, convert`
- `from molbuilder.pyscf  import PySCFConfig, render_script, convert`
- `from molbuilder.molwatch_log import write_initial_preview`
- `from molbuilder.parsers import detect_parser, TrajectoryParser`

The new canonical paths (`molbuilder.config.siesta`, `molbuilder.generators.siesta`,
`molbuilder.trajectory_log`) become preferred for new code, but the
older paths are not deprecated — they are the public surface.

---

## Merge plan — historical

The molbuilder + molwatch merge ran in six phases (subpackage promotion;
flat parsers layout; Flask blueprint split with namespaced routes;
`Frame` / `Trajectory` dataclasses; 3DNA backend; field-metadata-driven
validation; layering compliance; UI redesign with shared viewer; emitter
extraction; argparse→click; v0.4 scientific polish closing 10 known
gaps).  All phases are complete.  Reconstruct any specific phase from
`git log --oneline --grep="Phase\|review-fix\|merge"` as needed; the
post-merge **package layout** below is the current shape.

Tests are green at every commit on every phase branch — no "intermediate
broken state" commits.

---

## Scientific correctness

### Spin + charge: the most important pair of inputs

For ANY DFT/HF calculation, `(charge, spin)` together define the
electronic state.  Wrong values give wrong electronic structure,
which manifests as huge forces, non-convergence, or (worst) silent
convergence to a fictitious state that LOOKS reasonable but is the
wrong minimum.  The 2026-05-22 hemeC-dithiol incident
(below) was exactly this.

**Why these are easy to get wrong:**

  * Defaults look innocent: charge=0, spin=0 (closed-shell singlet)
    works for ~90% of organic molecules.  But ANY structure
    containing Fe / Mn / Co / Ni / Cu / Mo / W (open-shell
    transition metals) is in the other 10%.
  * The spin convention varies across codes: PySCF uses `spin = 2S
    = n_unpaired` (NOT multiplicity = 2S+1).  ORCA, Gaussian use
    multiplicity.  SIESTA uses `SpinPolarized` plus `SpinTotal`
    (in μ_B).  Easy to be off by one.
  * Wrong (charge, spin) often DOES converge SCF -- just to a
    different electronic state with different energy / forces /
    HOMO-LUMO ordering.  No obvious error message.
  * The "right" spin depends on coordination chemistry, not just
    element identity.  4-coordinate Fe(II)-porphyrin = S=1
    (intermediate); 5-coord with one weak axial ligand = S=2 (high);
    6-coord with two strong-field axial = S=0 (low).  No general
    formula -- depends on the experimental data.

**Checks molbuilder provides** (`molbuilder/chemistry.py` +
`molbuilder/validation.py`):

| Helper | What it catches |
|---|---|
| `total_electrons(struct, charge)` | sum(Z) - charge for any structure (raises on unknown element symbol) |
| `check_spin_charge_parity(struct, charge, spin)` | spin=0 requires even electron count; spin=1 requires odd; etc.  PySCF raises this AT RUN TIME; we catch it pre-emission for a clearer message. |
| `detect_open_shell_metals(struct)` | Returns list of open-shell transition metals present.  Empty for pure organics. |
| `explain_metal_spin(element, spin)` | One-line description of what (Fe, spin=4) implies (Fe(II) high-spin, S=2, 4 unpaired -- e.g. deoxy-heme). |
| `_check_open_shell_metal()` (validation.py) | Shared by `_validate_pyscf` AND `_validate_siesta`: warns when a structure with an open-shell metal is paired with a closed-shell SCF (PySCF RKS/RHF + spin=0; SIESTA SpinPolarized=False).  SAME warning regardless of engine -- same chemistry. |

**Why we didn't catch this earlier (post-mortem 2026-05-22):**

The bug surfaced when the user ran hemeC-dithiol (an Fe-porphyrin
with two thiol side chains) through PySCF spectra.  Symptom: forces
~10 eV/Å on a structure already near experimental equilibrium.
Root cause: `SpectraConfig.charge` and `SpectraConfig.spin` did not
exist as fields -- the spectra script's `gto.M()` call silently
used PySCF's defaults (charge=0, spin=0) regardless of what the
user wanted.  Fe(II) in a 4-coordinate porphyrin (no axial ligands
within bonding distance in the user's geometry) is
intermediate-spin S=1 (spin=2), not closed-shell S=0.  The SCF
converged to a fictitious low-spin state with unphysical orbital
occupancies, hence the enormous gradient.

What enabled the silent failure:

  1. `SpectraConfig` had `method` but not `charge` / `spin`.
     `_emit_build_mol` in `spectra/pyscf_script.py` emitted
     `gto.M(...)` without `charge=` / `spin=`, falling through to
     PySCF's (0, 0) default.
  2. The validation pass exists (`validation.py::_validate_pyscf`)
     but only ran from Build's `render_script`, not from the
     spectra script's emit path -- the spectra engine's `preflight`
     had its OWN list of checks that didn't include the open-shell-
     metal rule.
  3. The user has no way to specify spin from the form because
     the field didn't exist.  Silently using a wrong default with
     no input surface is the worst combination.

**Fixes that landed:**

  * Add `charge` + `spin` to `SpectraConfig` with detailed help text
    explaining the convention + giving Fe(II) / Fe(III) examples.
  * Emit them in the script's `gto.M(...)`.
  * Add the open-shell-metal check to BOTH `_validate_pyscf` and
    `_validate_siesta` (via shared `_check_open_shell_metal` helper)
    AND to `PySCFSpectraEngine.preflight` -- triple coverage so any
    surface that calls either entry point sees the warning.
  * Add `total_electrons` + `check_spin_charge_parity` +
    `explain_metal_spin` as standalone helpers for any future engine
    that needs to do the same checks.
  * The /spectra and /build forms now show the help text inline (the
    field metadata's `help` is rendered as a tooltip / aside by
    form-schema.js); the spin field's help enumerates the common
    Fe(II) / Fe(III) spin combinations so the user has a starting
    point without reading the literature.

**Cross-engine consistency rule:** ANY scientific check that depends
on chemistry (charge / spin / coordination / basis suitability)
MUST live in a shared helper called from BOTH `_validate_siesta`
AND `_validate_pyscf` -- same physical facts, same warning.  Don't
duplicate the check inline in one validator and forget the other;
add a helper.

### Validation pass (pre-emission)

Runs before `render_fdf` / `render_script` writes any output. Implemented
in `molbuilder/validation.py:validate_geometry(struct, cfg) -> List[Issue]`.
Errors stop emission; warnings print to stderr.

`Issue` is the L1 dataclass:

```python
@dataclass
class Issue:
    severity: Literal["error", "warn"]
    message:  str
    where:    str    # e.g. "geometry.min_distance" or "config.pao_energy_shift"
```

The validator pulls per-field rules from the `Config` field metadata
(`range`, `validate=` callable) plus the geometric checks below.

| Check | Severity | Rationale |
|---|---|---|
| min atom-atom distance < 0.3 Å | error | Atoms on top of each other; SCF will diverge |
| min atom-atom distance 0.3 – 0.7 Å | warn | Likely broken structure (failed protonation, bad backend output) |
| H/heavy ratio < 0.3 | warn | Heavy-atom skeleton — wrong electron count for DFT; user may have intentionally opted out of H-add (e.g. `build_dna(..., add_hydrogens=False)`) for hand-processing, hence warn not error |
| polymer residue listing reversed (structural 5' end ≠ residue_ids[0]) | warn | Every backend builds 5'→3' (lowest residue_id at 5' end). A reversed listing breaks downstream orientation-sensitive code (terminal-phosphate stripping, FDF residue numbering); likely a backend regression |
| polymer has multiple residues with no preceding O3'-P bridge (single-chain input) | warn | Disconnected backbone or unintended branching — single-chain input expected one 5' end |
| atom-to-nearest-image distance < 2 × cell_padding (vacuum case) | warn | Image-image interaction; suggest larger padding |
| cell volume / atom-bounding-volume < 3 | warn | Cell suspiciously tight |
| cell determinant ≤ 0 | error | Left-handed or degenerate cell |
| `kgrid != 1` along axis with extent < 10 Å | warn | k-points along a vacuum direction is wasted |
| `kgrid == 1` along axis with extent > 10 Å (periodic system) | warn | Likely under-converged k-grid |
| net dipole > 1 D in vacuum (no dipole correction) | warn | Image-image dipole; suggest dipole correction or bigger cell |
| atom outside [0, 1) fractional with `wrap_into_cell=False` | warn | Atom in neighbor cell; visualisations will look broken |
| explicit `Spin.Total` set but `spin_polarized=False` | warn | Total-spin pin will be silently ignored |

Reused by both SIESTA and PySCF generators. Unit-tested against fixtures
in `tests/conftest.py`. The CLI `molbuilder validate` subcommand emits
the same `List[Issue]` as JSON to stdout for shell-driven pre-flight
checks.

### Known SIESTA / PySCF science gaps — historical

Ten gaps were identified in the 2026-05-01 design review (SIESTA
`SpinTotal` / `SpinPolarized` keyword forms, dispersion-correction
emission, `mf.stability_analysis()` for open-shell, `PAO.EnergyShift`
default, post-processing hook templates, SIESTA version pinning,
ECP auto-emit for heavy atoms with non-def2 bases, post-relax
`mf.kernel()` re-evaluation, `mf.diis_space` / `mf.damp` exposure).
All ten are closed and pinned by tests in `tests/test_science_gaps.py`
(0 xfails).  Reconstruct any specific fix from
`git log --oneline --grep="science\|gap"`.

### Pinned false positive from the 2026-05-05 deep code review

The 22-item review on the post-merge branch landed eleven targeted
fix commits (review-fixes A-M, plus the dead-handoff cleanup); see
`git log --oneline --grep "review-fix"` for the chain.  One item was
a false positive worth documenting so it doesn't resurface:

- **TIER 2 #8 (geomeTRIC `convergence_*` kwargs raise TypeError)**
  was wrong.  PySCF's `geometric_solver.optimize(method, **kwargs)`
  forwards `**kwargs` into `geometric.optimize.OptParams(**kwargs)`,
  which accepts the lowercase keys `convergence_energy` /
  `convergence_grms` / `convergence_gmax` and stores them as the
  capitalized `Convergence_*` attributes.  The contract is pinned
  by introspection (no subprocess, no PySCF dependency) in
  `test_geometric_optparams_accepts_pyscf_optimize_kwargs` --
  that test fails at unit-test time if either side ever renames
  or rejects the keys, so a regression surfaces cleanly instead
  of crashing at user runtime.

### Generated-output style requirements

- **Verbose-comments mode** (default ON) emits inline tuning hints next
  to each parameter plus a troubleshooting block at end of file. Both
  must remain feature-complete through the merge.
- **Section headers** (`# --- Lattice ---`, `#  1. Build the molecule`,
  etc.) are mandatory.
- **Every tunable parameter** appears with its default value visible
  and a comment range (e.g. `# Range 0.001 - 0.5`) rather than hidden
  behind a function call.
- **Post-processing hook placeholders** (commented-out, ready to
  uncomment) belong at the end of every generated script / FDF.

---

## Backend roadmap

### 3DNA (canonical helix builder)

3DNA's `fiber` command produces true B-form / A-form / Z-form helical
geometry — the only thing the existing `rdkit` (folded conformer) and
`amber` (extended chain) backends do not provide.

#### Licensing and distribution constraints

**3DNA is not auto-installable, and molbuilder must not attempt to fetch it.**

- 3DNA is distributed by the Olson lab (Columbia University) through
  http://x3dna.org/ behind a **registration form** that requires the
  user to accept the license. The archive is not on a public mirror and
  cannot be obtained via `pip`, `conda`, `wget`, or any automated
  fetcher driven by molbuilder. Users **must** download it themselves
  by following the instructions on x3dna.org.
- The 3DNA license is **non-commercial-use only**. molbuilder itself is
  MIT-licensed; bundling, redistributing, or auto-mirroring 3DNA would
  drag the molbuilder distribution under 3DNA's restricted terms. We
  do neither, and shipped CI / docs / examples never invoke a fetch.
- The `x3dna-*.tar.gz` and `x3dna-*.zip` patterns in `.gitignore` exist
  for both reasons (a) keep the binary archive out of git on developer
  machines and (b) make it structurally hard for someone to accidentally
  commit a 3DNA archive into a public-facing molbuilder release.
- Documentation (this file, READMEs, error messages) must always tell
  users to **download from x3dna.org per their instructions and accept
  the license** rather than implying any automated install path exists.

#### Backend implementation

**Backend file:** `molbuilder/backends/_threedna.py` (will move under
`builders/backends/` in Phase 2.7), mirroring the shape of `_amber.py`:
shell out to `fiber`, parse the output PDB into `Structure`, run the
backbone-connectivity self-check
(`_common.verify_backbone_connectivity`).

**Detection chain.** First hit wins; `is_available()` returns True iff
any source resolves to a *complete* install (i.e. both `bin/fiber`
executable AND `config/` directory present). The chain is:

1. **In-tree** — glob `<repo_root>/x3dna-v*/`, where `<repo_root>` is
   one level above the molbuilder package. The user can simply unpack
   the 3DNA tarball at the repo root (gitignored — see `.gitignore`)
   and the backend lights up automatically. Easiest path for a dev
   install; useless for a wheel install (site-packages has no
   meaningful "next-to-package" location), so the env-var fallback
   exists.
2. **`$X3DNA` env var** — the canonical 3DNA install convention. Set
   `export X3DNA=$HOME/opt/x3dna-v2.4` and we use it.
3. **`fiber` on `PATH`** — last resort; we derive `X3DNA` root from
   `shutil.which("fiber")` (assumes the standard `$X3DNA/bin/`
   layout). Useful when the user has a system install that doesn't
   bother with the env var.

For **each** candidate root the backend verifies *completeness*:
`bin/fiber` is a regular file with the executable bit set, AND
`config/` exists as a directory (it holds 3DNA's atomic-parameter PDB
templates; without them `fiber` fails at runtime with cryptic
errors). The completeness check filters out the easy
foot-gun where `$X3DNA` points at a half-extracted tarball or a
sibling directory.

When `fiber` is shelled out, the resolved root is injected into the
subprocess environment as `X3DNA` (and prepended to `PATH`)
regardless of the calling shell's setup, so 3DNA's auxiliary scripts
resolve their config files correctly even when the user found the
install via the in-tree or PATH path rather than the env var.

If the entire chain fails, `is_available()` returns False **and**
`BackendUnavailable` is raised on explicit `--backend threedna`
requests, with the canonical error message below.

**Required error message contract.** When the user explicitly requests
`--backend threedna` (or any equivalent in the web UI / Python API)
and the backend is unavailable, the raised `BackendUnavailable`
message must include all of:

- which sources were checked (the three resolution strategies above —
  in-tree glob, `$X3DNA` env var, `fiber` on PATH — and their current
  values, so the user can see exactly what fell through);
- the URL `http://x3dna.org/` and an explicit "register and accept the
  license to download — molbuilder cannot fetch this for you";
- a one-line reminder that 3DNA is non-commercial-use only;
- the names of the two fallback backends (`amber`, `rdkit`).

Example of the required shape (final wording lives in the implementation,
keep this contract in sync):

```
3DNA is not available.  Tried, in order:
  1. in-tree   : no match for /path/to/repo/x3dna-v*
                 (unpack the 3DNA tarball at the repo root and this
                 lights up automatically)
  2. $X3DNA    : (unset)
                 (must point at a directory containing bin/fiber + config/)
  3. fiber on PATH: (not on PATH)

3DNA must be downloaded directly from http://x3dna.org/ after
registering and accepting the license — molbuilder cannot fetch it
for you.  The license is non-commercial-use only; do not redistribute
the archive.

If you don't need a canonical helix, the `amber` (extended chain) and
`rdkit` (folded conformer) backends remain available.
```

**Runtime errors during `fiber` execution** (timeout, non-zero exit,
empty PDB, malformed PDB, missing parameter files at runtime even
though config/ existed at detection time) are caught and re-raised as
`RuntimeError` with the captured stdout/stderr included verbatim.
Mirrors `_amber.py:96-108` in spirit. Do not silently swallow.

**Auto-detect order** in `builders/backends/__init__.py:dispatch` becomes
`threedna > amber > rdkit` (best geometry first). When 3DNA isn't
available the auto path falls through cleanly with no error — only
explicit `--backend threedna` raises.

**CLI / web surface:** existing `--backend` choices (`auto / rdkit / amber`)
extend to include `threedna`. The CLI's click `Choice(...)` and the
web UI's `<select>` options must include the new value. The web UI's
"backend not available" feedback for `threedna` must surface the same
"download from x3dna.org / non-commercial" guidance — not a bare
HTTP 500.

**Tests must cover:** `is_available()` returns False with each
detection-chain step missing (no in-tree dir, env unset, fiber off
PATH) without raising; an env-var path that points at an incomplete
install (no `config/`) is rejected; explicit `--backend threedna`
request when nothing is reachable produces a `BackendUnavailable`
containing the URL, the non-commercial license note, and the named
fallback backends; `auto` falls through silently when 3DNA is
unavailable; when an install IS reachable the build produces a
chemically plausible Structure (P present, expected base residues,
backbone connectivity passing); A-form and B-form coordinates differ
(the form flag actually plumbs through to fiber); RNA build uses U
not T (the `-rna` flag is set).

#### 3DNA installation

3DNA is distributed by the Olson lab (Columbia, x3dna.org).  Two install
shapes work; pick whichever matches how you use molbuilder.

**Option A — in-tree (recommended for dev / editable installs).** Unpack
the tarball at the molbuilder repo root.  The detection chain's first
step globs `<repo_root>/x3dna-v*/` and verifies completeness, so no
shell config or env var is needed:

```bash
cd /path/to/molbuilder              # the repo root, alongside pyproject.toml
tar -xzf x3dna-v2.4-<platform>.tar.gz
ls x3dna-v2.4/bin/fiber             # smoke check
python -c "from molbuilder.backends import available_backends; \
           print(available_backends())"
# expected: {'threedna': True, 'amber': ..., 'rdkit': ...}
```

The `x3dna-v*/` directory is gitignored (see `.gitignore`) — both for
hygiene and to make it structurally hard for someone to accidentally
commit the 3DNA archive into a public-facing molbuilder release.

**Option B — system install with `$X3DNA` env var (canonical).** This
is the install path the 3DNA upstream documents; the second step in
the detection chain picks it up:

```bash
tar -xzf x3dna-v2.4-<platform>.tar.gz -C ~/opt
export X3DNA=$HOME/opt/x3dna-v2.4
export PATH=$X3DNA/bin:$PATH
fiber -h
fiber -seq=ATCG /tmp/probe.pdb && head /tmp/probe.pdb
```

The `X3DNA` environment variable is required by 3DNA's auxiliary
scripts; molbuilder's `_threedna.py` injects it into the subprocess
environment automatically when shelling out, so the env var only needs
to be set in the user's shell when they want to invoke 3DNA tools
directly outside molbuilder.

##### Windows install (project-specific)

3DNA's official binary distribution does **not** include a native-Windows
build. The Linux tarball runs only inside WSL or Cygwin. **Recommended
path: WSL2 (Ubuntu).**

The archive on this machine is `x3dna-v2.4-linux-64bit.tar.gz` at the
molbuilder repo root (gitignored — see `.gitignore`). Concrete install
inside WSL2:

```bash
# 1. From a WSL2 (Ubuntu) shell.  The Windows path Y:\GitHub\quantum_simulation\molbuilder
#    is reachable from WSL as /mnt/y/GitHub/quantum_simulation/molbuilder.
mkdir -p ~/opt
tar -xzf /mnt/y/GitHub/quantum_simulation/molbuilder/x3dna-v2.4-linux-64bit.tar.gz \
        -C ~/opt
ls ~/opt/x3dna-v2.4/bin/fiber          # smoke check that extraction worked

# 2. Persist the env vars (append to ~/.bashrc):
echo 'export X3DNA=$HOME/opt/x3dna-v2.4'    >> ~/.bashrc
echo 'export PATH=$X3DNA/bin:$PATH'         >> ~/.bashrc
source ~/.bashrc

# 3. Verify fiber works
fiber -h                                      # prints usage
fiber -seq=ATCGATCG /tmp/probe.pdb && \
  head -5 /tmp/probe.pdb                      # prints REMARK lines

# 4. Verify molbuilder picks it up (run from inside WSL)
cd /mnt/y/GitHub/quantum_simulation/molbuilder
python -c "from molbuilder.builders.backends import available_backends; print(available_backends())"
# expected (after _threedna.py lands): {'rdkit': True, 'amber': ..., 'threedna': True}
```

Notes specific to running molbuilder from WSL on this host:

- **Run molbuilder from inside WSL,** not from Windows Python — only the
  WSL Python sees `fiber` on PATH and the `X3DNA` env var.
- File paths are interchangeable: WSL sees Windows drives at `/mnt/<letter>/`,
  Windows sees WSL files at `\\wsl$\Ubuntu\home\<user>\...`. Generated
  `.fdf` and `.py` files written from WSL are immediately editable from
  Windows tools.
- If you also want molbuilder's CLI from PowerShell, that's fine for
  build subcommands that don't need 3DNA (`peptide`, `smiles`, `fdf`,
  etc.); just don't pass `--backend threedna` from the Windows side —
  it'll fail `is_available()` and the user gets a clear
  `BackendUnavailable` error.

##### Alternative: Cygwin / MSYS2

The Linux tarball usually extracts and runs under Cygwin. Set the same
env vars in `~/.bashrc` inside the Cygwin shell. Path translation is
handled by Cygwin automatically. Less common than WSL2 these days.

##### Backend behavior when 3DNA isn't installed

`builders/backends/_threedna.py:is_available()` returns False when `fiber` isn't on
PATH or `X3DNA` isn't set. With `--backend auto` (default), molbuilder
falls through to `amber > rdkit` cleanly. With `--backend threedna`
explicit, the user gets a `BackendUnavailable` error citing the missing
PATH / env-var so they know exactly what to fix.

---

## Tool limitations and the H-placement design

Each backend has known quirks; molbuilder compensates so the
`build_dna` / `build_rna` API contract is consistent across them.
This section documents what each tool gets wrong, what we do about
it, and *why* the code is shaped the way it is so the next person
to touch it doesn't unwind the workarounds.

### What each backend produces, raw

| backend | helical shape | H atoms | terminal phosphate | residue names |
|---|---|---|---|---|
| `threedna` (X3DNA fiber) | canonical B/A/Z | **none** (heavy-only) | **always 5'-P** (ignores request) | DA / DT / DG / DC |
| `amber` (AmberTools tleap) | extended chain | included | honors request | DA5 / DT / DG / DC3 (5'/3' suffixes) |
| `rdkit` | folded conformer | included via `Chem.AddHs(mol)` | none (single nucleoside fragments) | molecule-level (no per-residue) |

The X3DNA path is the one that needs the most repair work.

### Hydrogen addition: OpenBabel preferred, RDKit fallback

Implementation in `chemistry.add_hydrogens(struct)`. Detection chain:
**OpenBabel → RDKit → warning**.

#### Why OpenBabel first

- **`OBMol.AddHydrogens()` is geometric.** It places H along sp3 /
  sp2 / sp vectors directly from each parent atom's hybridization
  and existing neighbours. There is no "give up and place at
  parent coordinates" failure mode.
- On standard biomolecules (DA/DT/DG/DC, 20 amino acids) the
  residue-template chemistry is mature and battle-tested (25+ years
  of cheminformatics use; what AutoDock and most MD prep pipelines
  use under the hood).
- It doesn't reorder atoms.
- **Verified on the X3DNA → ATGC test case:** OpenBabel produces
  the canonical `5 O-H + 37 C-H + 8 N-H` breakdown, matching
  Amber-tleap and RDKit-via-SMILES exactly. All Watson-Crick H-bond
  donors (A.N6-H₂, T.N3-H, G.N1-H + G.N2-H₂, C.N4-H₂) are present.

#### Why RDKit is the fallback (and what it gets wrong)

- **Bond-order perception from PDB residue templates is correct.**
  When given a heavy-atom-only PDB with standard residue names,
  `Chem.MolFromPDBBlock` perceives bond orders correctly.
- BUT `Chem.AddHs(mol, addCoords=True)` has a known limitation:
  for sites where the heavy-atom geometry doesn't uniquely
  constrain H placement — typically **exocyclic -NH₂ amines on
  nucleic acid bases** (A.N6, G.N2, C.N4) and **peptide N-terminal
  -NH₃⁺** — the addCoords flag sometimes leaves H atoms **at their
  parent atom's coordinates** (zero-distance "ghost H").
- For a typical ATGC chain, this loses 4 H out of 50 — exactly the
  Watson-Crick H-bond donors. Structurally crippled for any
  H-bonding chemistry.
- The SMILES path doesn't have this issue (`build_peptide` and the
  `rdkit` nucleic backend reach the SMILES path); only PDB-parse
  then AddHs has it. The X3DNA path lands here.
- We keep RDKit as the fallback because it's already a dep, the
  failure mode is bounded (peptide ambiguous H, nucleic exocyclic
  amines), and `_drop_overlapping_hydrogens` cleans up the ghosts
  so downstream validators don't see zero-distance pairs.

#### Why not AmberTools `reduce`

`reduce` is the gold standard for protein protonation (His tautomer
selection, Asn/Gln side-chain flips). For DNA it's not better than
OpenBabel and adds:

- A subprocess + temp-file round trip (vs in-process OpenBabel).
- A different deployment story (it's bundled with AmberTools, but
  invoking it shells out — harder to reason about than a Python
  call).

We have AmberTools as a transitive dep already (the `amber` nucleic
backend uses `tleap`), so `reduce` would not add a dependency. We
still don't use it because keeping H-placement uniform across
peptide and nucleic builds — same function, same code path,
in-process — is more important than the marginal protein-side
correctness `reduce` would add. The peptide builder is currently
satisfied by OpenBabel; if and when we hit a peptide tautomer case
that OpenBabel mishandles, `reduce` becomes a candidate third
engine in the chain.

#### `_drop_overlapping_hydrogens` post-pass

Removes H atoms < 0.05 Å from any other atom. Threshold rationale:
the shortest physical X-H bond (H-F at ~0.92 Å) is far above 0.05 Å,
so a H within 0.05 Å of another atom is unambiguously a placement
artifact.

- **What this catches:** RDKit `addCoords=True` ghost H at
  ambiguous-valence sites (the defining failure mode); rare
  OpenBabel duplicates at tautomeric sites.
- **What this does NOT do:** re-place the ghost H at sensible
  positions. That's the smarter remediation but requires
  hybridization perception (already in `_adjacency`) plus
  open-valence vector computation (new code). Worth doing only if
  RDKit becomes the primary engine; with OpenBabel preferred, the
  drop is a safety net, not a load-bearing path.
- **What this never touches:** heavy atoms. Two heavy atoms within
  0.05 Å are a broken structure that the validator should error on,
  not silently fix.

### X3DNA `fiber` quirks and how we compensate

In `_threedna.py`:

1. **Heavy-atom output → routed through `chemistry.add_hydrogens`**
   at the `nucleic.build_dna`/`build_rna` layer. The
   `_maybe_add_hydrogens` shim short-circuits via the H/heavy ≥ 0.3
   ratio gate, so amber- and rdkit-built structures (which already
   have H) skip the round-trip cleanly.
2. **Mandatory 5'-terminal phosphate → `_strip_5prime_phosphate`.**
   Removes atoms named in `_PHOSPHATE_ATOM_NAMES` (covers both
   modern OPx and legacy OxP naming) from the 5'-terminal residue
   when `terminal in ('OH', '3P')`. The bridging O5' stays as part
   of the sugar; H is added later by `chemistry.add_hydrogens`.
3. **3'-phosphate cannot be added → warn.** fiber's output is
   5'-P / 3'-OH; we can strip the 5', but not synthesize a 3'.
   `terminal in ('PP', '3P')` warns the request will be served as
   5'-P / 3'-OH or 5'-OH / 3'-OH respectively.
4. **Z-form is poly-d(GC) only; RNA is A-form only.** Mismatches
   are warned at dispatch (see `build()`).

### 5'/3' directionality on user input

Bare letters (`"ATGC"`) follow biology convention: 5' on the left, 3'
on the right. `parse_dna_sequence` / `parse_rna_sequence` also accept
optional end-labels:

  * `"5'-ATGC-3'"` — explicit 5'→3', identical to bare.
  * `"3'-CGTA-5'"` — reverse-direction; the parser reverses the
    residue list so the backend (which always builds 5'→3') produces
    a polymer matching the user's stated direction.
  * `"5'-ATGC-5'"` / `"3'-ATGC-3'"` / `"5'-ATGC"` / `"ATGC-3'"` —
    self-contradictory or one-sided; ValueError.

Whitespace, internal dashes, and mixed punctuation between the labels
and the body are tolerated (`"5'  -  ATGC  -  3'"` parses cleanly).

The orientation validator (above) catches the case where the
*structural* 5' end (the residue with no incoming O3'-P bridge) doesn't
match `residue_ids[0]` — this is what protects against a future backend
that lists residues 3'→5' rather than 5'→3'.

### How a regression in any of this would surface

Tests that pin the current behavior (`tests/test_nucleic.py`):

- `test_dna_default_protonation_yields_simulation_ready_h_count`
  — asserts H/heavy ≥ 0.55 across all installed backends. Catches
  the case where the X3DNA path silently falls through to "no H
  added" (e.g., both OpenBabel and RDKit uninstalled, or the
  H/heavy ratio gate misfires).
- `test_dna_atgc_protonation_chemistry_matches_across_backends`
  — pins the canonical anchor-element breakdown
  (5 O-H / 37 C-H / 8 N-H). Catches the RDKit-fallback regression
  where Watson-Crick H atoms get dropped.
- `test_threedna_strips_5prime_phosphate_for_terminal_oh` — pins
  P count = 0 for a single nucleotide, P count = 3 for ATGC.
  Catches a regression in the strip helper or a fiber-output
  format change that defeats the atom-name match.
- `test_dna_add_hydrogens_false_returns_heavy_skeleton` — pins
  that the kwarg is honored (≤ 5 H on the fiber-skeleton path).

If any of these red, the protonation contract has drifted; don't
"fix" by adjusting the test thresholds — re-derive what changed.

---

## File format spec

`.molwatch.log v1` — single source of truth post-merge:
`molbuilder/trajectory_log/format.py`. Both the writer (in PySCF input
generation + the standalone preview helper) and the reader (the parser
at `molbuilder/parsers/molwatch_log.py`) read field names from the same
place. The format is marker-delimited and tolerant of truncation (a
torn final block on a still-running job is dropped on parse).

The on-disk file extension `.molwatch.log` does **not** change in the
rename. It's a user-facing filename that downstream tools and scripts
may match on; only the Python module name changes.

---

## Next steps

_The previous Next-steps list is empty as of 2026-05-11._
The Build-tab dataclass-driven form (the last Principle-#1
anti-pattern) shipped in commit `20f6d49`; the related decision
row is in the log above.  Add items here when new design gaps
surface; otherwise leave this section as the visible-clean
indicator that nothing is pending.

---

## Process rules

- Any change to the principles or decisions in this document requires
  updating it in the same PR as the code change. A drift between this
  doc and the code is a bug.
- Test contracts (the per-component specs) live under `docs/`, organised
  by purpose: `protocols/` (cross-cutting interfaces), `types/` (L1
  data-type contracts), `engines/` (per-engine emitter specs), and
  `tabs/` (per-tab UI specs).  Tests must be derivable from those specs
  without reading the implementation.  See [`docs/README.md`](README.md)
  for the index + the spec-doc rule.
- Code review must explicitly check (a) target-tool correctness for
  generated SIESTA / PySCF outputs, (b) scientific defensibility of
  defaults, and (c) the layering invariant (no L1 → L2 imports, no
  L2 → L3 imports) — not just code quality and tests.
- Every commit on a phase branch must keep `pytest tests/ -q` green.
  No "intermediate broken state" commits; if a refactor would
  temporarily break tests, split it finer.
- Adding a third-party dependency requires a one-line entry in the
  decisions log explaining what wheel it replaces. Default is to not
  add dependencies.
