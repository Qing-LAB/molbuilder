# molbuilder — design and roadmap

This document is the durable design reference for molbuilder. It captures
mission, architectural principles, decisions made, and active roadmap items
that span multiple sessions of work. Per-component test contracts live under
[`docs/spec/`](spec/README.md); this document sits above them.

When in doubt about whether to do something, read this file first. When the
file is wrong (the decision changed, the constraint shifted), update it in
the same PR as the code change.

---

## Mission

molbuilder builds 3-D molecular structures from sequence / SMILES / name
input, generates SIESTA and PySCF input files for those structures, and
provides a live trajectory viewer that monitors the resulting calculations.

The package is a single, internally-coherent toolkit covering the full
pipeline:

```
sequence ──► Structure ──► SIESTA .fdf  ──► siesta ──┐
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

> **Module paths in this table reflect the post-2.7 target layout.**
> Phase 2.7 is rolling out incrementally; these paths still differ
> from the on-disk reality:
>   * builders -- still `molbuilder/peptide.py`,
>     `molbuilder/nucleic.py`, `molbuilder/smiles.py`,
>     `molbuilder/pubchem.py` (top-level, not under `builders/`).
>   * generators -- still `molbuilder/siesta/input.py` and
>     `molbuilder/pyscf/input.py` (the renderers; not under
>     `generators/`).
> The verbs themselves and their consume/yield contracts are accurate
> now.

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
- **Spec docs**: `docs/spec/<config>.md` can be (semi-)generated from
  metadata so they don't drift.

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

Single page, three control tabs: **Style** (representation, radius,
element coloring, background, cell visibility), **Overlays** (atom
indices, force arrows with magnitude threshold, highlight max-force
atom), **Playback** (slider, prev / play / pause / next, speed,
loop). Frame counter "X / N" sits above the slider.

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
is pasted verbatim into the generated script (currently as
text-templating; Phase 4 switches to `inspect.getsource()` of a real
file under `generators/_runtime/molwatch_emitter.py` for IDE-checkability
and unit testability — the "generated artifact has no extra imports"
invariant is preserved either way).

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

---

## Post-merge package layout

> **This is the post-2.7 target layout.**  Phase 2.7 done so far:
>   * `config/` -- `SiestaConfig` and `PySCFConfig` live there
>     (re-exported from the engine packages for back-compat).
>   * `trajectory_log/` -- renamed from `molwatch_log/`; the old
>     name is preserved as a back-compat shim.
>   * `builders/backends/` -- the per-tool nucleic-acid backends
>     moved under a new `builders/` package; `molbuilder.backends`
>     is preserved as a back-compat shim.
> Still pending:
>   * `generators/` -- the SIESTA / PySCF emitters still live at
>     `molbuilder/siesta/input.py` and `molbuilder/pyscf/input.py`.
>   * The build verbs (peptide / nucleic / smiles / pubchem) still
>     live at the top level rather than under `builders/`.
> Re-export shims keep external imports stable while the layout
> moves; until then,
> re-export shims keep external imports stable but the on-disk
> shape is the pre-2.7 one.

```
molbuilder/
  # ----- L1: core types -----
  structure.py             # Structure dataclass + readers / writers
  frame.py                 # Frame dataclass — Structure + per-step physics
  issues.py                # Issue(severity, message, where)
  config/
    __init__.py            # re-exports SiestaConfig, PySCFConfig
    siesta.py              # SiestaConfig (was head of siesta/input.py)
    pyscf.py               # PySCFConfig (was head of pyscf/input.py)
  chemistry.py             # element table, masses, valences
  residues.py              # PDB residue templates
  trajectory_log/          # was molwatch_log/
    __init__.py
    format.py              # writer + on-disk format spec
                           # (the .molwatch.log v1 file extension is unchanged;
                           # only the module name changes)

  # ----- L2: domain verbs -----
  builders/
    __init__.py            # re-exports build_peptide / build_dna / build_rna /
                           # build_from_smiles / build_from_name
    peptide.py
    nucleic.py
    smiles.py
    pubchem.py
    backends/              # used only by builders/
      __init__.py
      _amber.py            # tleap-driven extended chain
      _rdkit.py            # ETKDG embedded conformer
      _threedna.py         # canonical B/A/Z-form helix via fiber
      _common.py
  generators/
    __init__.py            # re-exports render_fdf, render_script
    siesta.py              # was siesta/input.py (generator only; Config moved out)
    pyscf.py               # was pyscf/input.py (generator only; Config moved out)
    _runtime/
      molwatch_emitter.py  # extracted; pasted into generated script via inspect.getsource()
  parsers/
    __init__.py            # registry + detect_parser
    base.py                # TrajectoryParser ABC; parse() -> Trajectory
    molwatch_log.py
    siesta.py
    pyscf.py
  validation.py            # validate_geometry(struct, cfg) -> List[Issue]

  # ----- L3: surfaces -----
  cli.py                   # click-based; one subcommand group per verb
  web/
    __init__.py
    app.py                 # Flask app + Blueprint registration
    blueprints/
      build.py             # /api/build/* routes
      watch.py             # /api/watch/* routes
    templates/index.html   # tabbed UI shell
    static/
      viewer.js            # shared 3Dmol viewer + style controls
      style.css

  __init__.py              # public API: re-exports L1 types + key L2 verbs

tests/
  conftest.py
  test_structure.py
  test_frame.py            # NEW
  test_chemistry.py
  test_residues.py
  test_peptide.py
  test_nucleic.py
  test_smiles_and_siesta.py
  test_pyscf.py
  test_pyscf_spec.py
  test_review_fixes.py
  test_load.py
  test_pdb_ter.py
  test_output_correctness.py
  test_molwatch_preview.py
  test_web.py
  test_validation.py       # NEW
  watch/                   # parser tests; was _inbound_molwatch/tests/
    test_registry.py
    test_molwatch_log_parser.py
    test_siesta_parser.py
    test_pyscf_parser.py
    test_api_load.py
    test_app_concurrency.py
spec/                      # docs/spec/ — per-component test contracts
  README.md
  builders.md
  chemistry.md
  cli.md
  pyscf-script.md
  siesta-fdf.md
  structure.md
  web-api.md
  parsers.md               # NEW
  validation.md            # NEW
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

## Merge plan

| Phase | Outcome | Status |
|---|---|---|
| 1 / commit 3 | `siesta`, `pyscf`, `molwatch_log` promoted to subpackages with re-exporting `__init__.py`; tests green | **done — `e34ede7`** |
| 1 / commit 4 | `_inbound_molwatch/parsers/` moved to `molbuilder/parsers/` (flat layout: one `<format>.py` per parser); parser tests lifted to `tests/watch/`; `_inbound_molwatch/tests/conftest.py` merged into `tests/conftest.py` (it's a `sys.path` hack, not a copy-target) | **done — `1cb9c48`** |
| 1 / commit 5 | Watch web app moved to `molbuilder/web/blueprints/watch.py` (split route group, not a separate app); `molbuilder watch serve` CLI subcommand added; web routes namespaced `/api/build/*` and `/api/watch/*` via Flask Blueprints; `flask` lifted to core dependency; `molwatch` console-script shim added | **done — `29af210`** (build-side namespacing landed later as `3d3d85b`) |
| 1 / commit 6 | `_inbound_molwatch/` deleted; remaining `docs/spec/` files merged into `docs/spec/`; `pyproject.toml` / `requirements.txt` from the subtree dropped | **done — `7214c08`** |
| 2 | `Frame` and minimal `Trajectory(source_format, frames, lattice)` dataclasses added at `molbuilder/frame.py`; parsers' `parse(path)` now returns `Trajectory`; the legacy molwatch v1 dict shape produced by `molbuilder/parsers/__init__.py:trajectory_to_legacy_dict` so `/api/watch/load` keeps the same JSON the JS client expects | **done** |
| 2.5 | 3DNA backend added at `molbuilder/backends/_threedna.py` (will move to `builders/backends/` in Phase 2.7); detection chain `in-tree > $X3DNA > PATH`; registered in dispatch with auto-order `threedna > amber > rdkit`; CLI's `--backend` choice list extended | **done** |
| 2.6 | `Issue` dataclass added (`issues.py`); `validation.py` wired into `render_fdf` / `render_script`; per-field metadata lifted onto `SiestaConfig` and `PySCFConfig` via `dataclasses.field(metadata=...)`; validators read ranges from the metadata. **This is when principles #1 and #6 become load-bearing.** Side effect: the validator surfaced an `AddHs(addCoords=True)` artifact in the peptide builder (some Hs left at heavy-atom anchor positions); fixed in the same phase via a `_drop_overlapping_hydrogens` post-pass. | **done** |
| 2.7 | Layering-compliance commit: `SiestaConfig` / `PySCFConfig` split out into `molbuilder/config/` (re-exports preserve external imports); `molwatch_log/` renamed to `trajectory_log/` (re-exports preserved); `backends/` moved under `builders/`. No behavior change. | **done** (landed as 2.7a + 2.7b + 2.7c) |
| 3 | Web UI redesigned: Build tab + Watch tab; shared 3Dmol viewer, style controls; clean styling pass; "Watch this run" handoff. SSE / WebSocket vs 15s mtime polling resolved at `a3aab23` (server is authoritative; polling stays). | **partially done** — 3a build routes namespaced under `/api/build/*` (`3d3d85b`); 3b shared tab nav + banner above menu (`4b274d5` / `10a0d88` / `562c34d`); 3d Watch-this-run handoff (`86a2d64`); 3c shared 3Dmol viewer **deferred** (refactor, not load-bearing — current per-page viewers work). |
| 4 | `_MolwatchEmitter` extracted to `molbuilder/generators/_runtime/molwatch_emitter.py`; pasted into generated script via `inspect.getsource()`; round-trip emitter→parser unit test added. **Re-tiered:** principle #7 already holds via the inlined-text emitter at `pyscf/input.py:618`; this phase is for IDE-checkability and unit-testability of the emitter, not correctness. Can ship after Phase 5. | not started |
| 5 | argparse → click conversion of `cli.py`; `molbuilder watch parse` / `molbuilder watch tail` JSON-on-stdout subcommands (issue #81); `molbuilder validate` subcommand emitting `Issue` JSON; `-` stdin support on `fdf` / `pyscf` so principle #2 is realized end-to-end; `add_dataclass_options` bridge from field metadata to `click.option` lands here | not started |
| 6 | v0.4 scientific polish — fix the 10 known gaps below; each fix lands as the triple (generator change + validation rule + spec test) so they can't drift; lift `mf.diis_space` / `mf.damp` / `pao_energy_shift` defaults onto the metadata-augmented config fields from Phase 2.6 | not started |

Each commit must keep `pytest tests/ -q` green. No "intermediate broken
state" commits; if a refactor would temporarily break tests, split it
finer.

---

## Scientific correctness

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

### Known SIESTA / PySCF science gaps

Identified during the 2026-05-01 design review. All ten confirmed
present and unfixed in the audit on the same date. Each lands as the
triple (generator change + validation rule + spec test) in Phase 6.

1. ~~**`SpinTotal` keyword in FDF (`generators/siesta.py`, line ~587 in
   pre-split file) is probably not a real SIESTA keyword.** SIESTA uses
   `Spin.Fix true` + `Spin.Total <v>`. Verify against the SIESTA manual
   for the targeted version range and fix the emission. (Currently
   silently ignored by SIESTA's fdf parser on a value mismatch.)~~
   **Fixed:** generator now emits `Spin.Fix true` + `Spin.Total <v>`
   as a paired two-line block.  Test
   `test_gap_1_siesta_emits_spin_total_with_dot` flipped from xfail
   to passing; `test_c2_spin_total_emits_dotted_form_with_fix`
   pins the new form + asserts `SpinTotal ` (legacy bogus token)
   never appears.
2. ~~**`SpinPolarized true` (line ~579 in pre-split file) is the v4-era
   keyword.** SIESTA v5 prefers single-line `Spin polarized`. Either
   feature-detect or document the targeted SIESTA version range.~~
   **Fixed:** generator emits the v5 single-line `Spin polarized`.
   Targeted SIESTA range now documented in the spin block's verbose
   comments (4.1 -- 5.x; v4 back-compat accepted but deprecated in v5+).
   Test `test_gap_2_siesta_emits_v5_spin_block` flipped to passing;
   `test_c2_spin_polarized_emits_v5_form` pins the v5 form + asserts
   `SpinPolarized` (legacy v4 token) never appears.
3. ~~**No SIESTA dispersion-correction emission.** For organic /
   biomolecule work without a vdW-aware functional, plain PBE / B3LYP
   underbinds. Add a commented-out `%block MM.Potentials` (D2/D3
   empirical) template when the chosen XC is non-dispersive.~~
   **Fixed:** SIESTA generator now emits a commented Grimme-D2
   template via `_emit_dispersion_template` when
   `cfg.xc_functional.upper() != "VDW"`.  Verbose mode adds the
   "what under-binds" rationale + the alternative "switch to VDW
   XC" recipe; non-verbose still emits the bare template stub.
   Skipped when the user already chose a vdW-aware XC (otherwise
   the block would double-count).  Tests:
   `test_gap_3_siesta_emits_dispersion_template_for_pbe` (xfail
   flipped) + `test_gap_3_dispersion_template_suppressed_for_vdw_xc`.
4. ~~**`mf.stability_analysis()` is not auto-emitted for UKS / UHF.**
   Open-shell SCFs can converge to broken-symmetry saddles; without a
   stability check the user gets a non-variational answer with no
   warning. Auto-emit when `method` starts with `U`.~~ **Fixed:**
   PySCF generator now emits a `mf.stability_analysis()` call after
   the SCF/optimize stage when `method_class.startswith("U")`. The
   xfail is flipped to a passing test
   (`test_gap_4_pyscf_uks_emits_stability_analysis`); a complementary
   test pins that closed-shell scripts don't carry the call.
5. ~~**`PAO.EnergyShift 0.02 Ry` default is loose.** Production SIESTA
   work typically uses 0.005 – 0.01 Ry. Tighten the default to 0.01 Ry.~~
   **Fixed:** `SiestaConfig.pao_energy_shift` default is now 0.01 Ry
   (was 0.02; SIESTA's own default).  ~2x slower SCF for far better
   PAO-tail convergence -- production-side of "well-converged" per
   the SIESTA manual's 0.001-0.01 recommendation.  Tighten to 0.005
   for phonon / vibrational work; loosen back to 0.02 for screening.
   Test `test_gap_5_siesta_pao_energy_shift_default_is_tight` flipped
   from xfail to passing.
6. **No post-processing block in either generator.** Add a commented-out
   `# --- Post-processing hook ---` placeholder to both with 2-3 common
   follow-ups (SIESTA: `BandLines` / `PDOS`; PySCF: `analyze()` /
   `mulliken_pop()` / `dip_moment()`).
7. **No SIESTA minimum version pinned.** `requirements-runtime.txt`
   doesn't declare a minimum; emitted keywords like `DM.Energy.Tolerance`
   may be silently ignored on old builds. Document the targeted range.
8. **No ECP support for non-def2 bases.** A user with a Pt/Pd structure
   on cc-pVDZ needs a manual `ecp = {...}` block. Lower priority.
9. **`save_optimized_xyz` writes from `mol_eq` (correct), but `mf.e_tot`
   may not match `mol_eq`'s geometry for non-converged opts.** Probably
   a non-issue in practice; flag for awareness.
10. **No `mf.diis_space` / `mf.damp` in `PySCFConfig`.** Hard-SCF
    troubleshooting requires editing the generated script. Mentioned in
    the troubleshooting block; could be exposed as config fields.

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

## Open questions

- Frequency / thermochemistry support in the PySCF script (post-relax
  Hessian + RRHO). Lower priority than the science gap list above.
- Whether `Trajectory` should grow analysis methods (RMSD, principal
  axes, dipole moment time series) versus staying as a thin
  `(source_format, frames, lattice)` wrapper.  Phase 2 landed it as
  the thin shape; revisit if Phase 3's web redesign or new CLI
  subcommands want richer ergonomics.
- Whether a non-trivial CP2K / ORCA generator + parser arrives before
  v1.0. If yes, the `generators/` and `parsers/` flat layouts already
  accommodate it; if no, the abstraction is fine as-is.

---

## Process rules

- Any change to the principles or decisions in this document requires
  updating it in the same PR as the code change. A drift between this
  doc and the code is a bug.
- Test contracts (the per-component specs) live under `docs/spec/`. Tests
  must be derivable from those specs without reading the implementation.
  See [`docs/spec/README.md`](spec/README.md) for the rule.
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
