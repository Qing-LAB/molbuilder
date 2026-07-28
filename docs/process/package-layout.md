# Package layout — where everything lives

**Role:** reference
**Domain:** process
**Companions:** this map is keyed to the doc domains — each area links to the
doc that explains it: [`model/overview.md`](?doc=model/overview.md),
[`science/overview.md`](?doc=science/overview.md),
[`engines/overview.md`](?doc=engines/overview.md),
[`execution/overview.md`](?doc=execution/overview.md),
[`web/overview.md`](?doc=web/overview.md), [`ops/installation.md`](?doc=ops/installation.md).

This is the "where does *X* live" map of the repository. It's organised by the
same **domains** the docs use, not by an abstract layer scheme — so if you know
which doc explains a thing, you know which folder holds it, and vice-versa.

## 1. Top of the repo

| Path | What it is |
|---|---|
| `molbuilder/` | the Python package — **all** app + backend code |
| `tests/` | the test suite (`pyproject.toml` `testpaths=["tests"]`) |
| `docs/` | these docs (domain-split: `model/ science/ engines/ execution/ web/ ops/ process/ archive/`) |
| `old_docs/` | the **frozen** legacy docs — the migration source, deleted at closeout |
| `scripts/` | operator helpers (`install-env.sh`, screenshot capture, a bench script) |
| `tools/` | dev/pytest tooling (`progress_plugin.py`, `testrun.py`, screenshot capture) |
| `projects/` | runtime **project data** a running server reads/writes (`project/topic/structure/…`) |
| `pyproject.toml` | build + packaging + the pytest/lint config |
| `x3dna-v2.4/` | the vendored 3DNA backend (restricted-licence, not pip-shipped) |

## 2. The `molbuilder/` package, by domain

### Data model + science backends → [`model/`](?doc=model/overview.md), [`science/`](?doc=science/overview.md)

The shared "nouns" and the cross-engine chemistry:

- **`structure.py`** — the `Structure` dataclass + XYZ/PDB/PySCF/ASE readers/writers; the common currency between builders and consumers ([`structure.md`](?doc=model/structure.md)).
- **`frame.py`** — `Frame` + `Trajectory` (parser-output types). **`chemistry.py`** — the shared element/adjacency helpers ([`chemistry.md`](?doc=model/chemistry.md)). **`selection.py`** — the atom-selection rule grammar. **`engine_atom_index.py`** — the 0-based ⇄ engine-index translation. **`residues.py`**, **`issues.py`** (`Issue`/`ValidationError`).
- **`config/`** — the engine-parameter dataclasses (`SiestaConfig`/`PySCFConfig`/`SpectraConfig`/`TransportConfig`).
- **`parse/`** — the unified file/text/dir → `ParseResult` layer ([`parse.md`](?doc=model/parse.md)), split into `coords/` (geometry parsers), `engines/` (`.out`/`.log` parsers), `scripts/` (the reserved-block text parsers), `dirs/` (directory composers), `sidecars/` (sidecar **read** side).
- **`sidecars/`** — the sidecar JSON **write** side (a deliberate read/write split from `parse/sidecars/`).
- **`validation/`** — the pre-emission validation package (`chemistry`/`geometry`/`siesta`/`pyscf`/`metadata`/`sidecar`) ([`science/validation.md`](?doc=science/validation.md)). **`pseudos.py`** — SIESTA `.psml` validation ([`pseudopotentials.md`](?doc=science/pseudopotentials.md)).

### Engine emitters → [`engines/`](?doc=engines/overview.md)

- **`siesta/`** — `.fdf` generation ([`siesta.md`](?doc=engines/siesta.md)); **`pyscf/`** — `.py` script generation ([`pyscf.md`](?doc=engines/pyscf.md)); **`spectra/`** — the harmonic-vibration Spectra engine; **`transport/`** — the TranSIESTA transport engine ([`transport.md`](?doc=engines/transport.md)); **`builders/`** (+ `builders/backends/` = amber/rdkit/3dna) — the structure-build backends ([`builders.md`](?doc=engines/builders.md)).
- Write-side glue: **`script_emit.py`** (the reserved-block emitter), **`bundle_writer.py`** (materialise a bundle), **`annotations_fdf.py`**, **`runtime_info.py`**.

### Execution + job harness → [`execution/`](?doc=execution/overview.md)

- **`cli.py`** — the `molbuilder` command (§3 of [`conventions.md`](?doc=process/conventions.md)).
- **`runwrap.py`** (the `.run.sh` wrapper), **`monitor.py`** (background job monitor), **`checkpoint.py`** (git run-checkpoints), **`persist.py`** (the `name@major` versioned-doc IO), **`trajectory_log/`** (the `.molwatch.log` writer).
- **`jobset/`** — the declarative JobSet framework ([`job-system.md`](?doc=execution/job-system.md)); **`bench/`** — the SIESTA benchmark harness.

### Ops → [`ops/`](?doc=ops/installation.md)

- **`envs/`** — conda-env management (`recipes.py`, `install.py`, `doctor.py`, `builds.py`, …). **`diagnostics.py`** (pre-run machine snapshot), **`runtime_config.py`** (the `./molbuilder.json` reader — named to *not* collide with the `config/` engine-param package), **`auth_setup.py`**.

### The web app → [`web/`](?doc=web/overview.md)

`molbuilder/web/` is the Flask UI:

- **Server:** `app.py` (the app factory), `blueprints/` (the route modules — `build`, `modify`, `results`, `spectra`, `transport`, `watch`, `files`, `docs`, `checkpoint`, `selection`, `state_timeline`, `system_load` + `_shared.py`), `auth.py`, `rate_limit.py`, `tabs.py`, `auth_providers/`, `runtime_config` (shared with ops), `projects.py`.
- **Client:** `templates/` (one `.html` per tab + partials) and `static/` — per-tab asset dirs (`modify/`, `structure-optimization/`, `spectra/`, `results/`, `transport/`, `documents/`, `molview/`), `vendor/` (codemirror, dompurify, marked, mermaid, gitgraph), and **`static/lib/`** — the reusable front-end **module packages**, each owned by a web doc:

| `static/lib/…` | Doc |
|---|---|
| `molview/` + `viewer/` | [`molview.md`](?doc=web/molview.md) |
| `vibrationview/` | [`vibrationview.md`](?doc=web/vibrationview.md) |
| `workspace/` | [`workspace.md`](?doc=web/workspace.md) |
| `projects/` | [`projects.md`](?doc=web/projects.md) |
| `inspectors/` | [`presenters.md`](?doc=web/presenters.md) |
| `results/` · `trajectory/` · `spectra/` · `transport/` | [`results.md`](?doc=web/results.md) · [`trajectory.md`](?doc=web/trajectory.md) · [`spectra.md`](?doc=web/spectra.md) · [`tabs.md`](?doc=web/tabs.md) |
| the loose `lib/*.js` primitives | [`runtime.md`](?doc=web/runtime.md), [`form-schema.md`](?doc=web/form-schema.md), [`notifications.md`](?doc=web/notifications.md), [`ui-contract.md`](?doc=web/ui-contract.md) |

## 3. Tests

`tests/` is **flat at the top with a few topic subdirs** — ~275 `test_*.py` files, most in the root, plus `tests/parse/`, `tests/spectra/`, `tests/validation/`, `tests/watch/`. Fixtures live in `tests/data/`; there are two `conftest.py` (`tests/` and `tests/validation/`). The pyramid markers (`unit`/`module`/`interface`/`integration`/`smoke`/`e2e`/`slow`) are declared in `pyproject.toml`, and **import direction is enforced by `tests/test_layering.py`** — the one structural invariant that keeps the layering honest. How to test is [`testing.md`](?doc=process/testing.md).

## 4. Packaging

From `pyproject.toml`: package `molbuilder` (setuptools), the console scripts
`molbuilder = molbuilder.cli:main` (+ a back-compat `molwatch`), all code
discovered under `molbuilder/`, and optional extras (`gpu`, `rdkit`, `web`,
`auth`, `test`, `e2e`, `full`, …). Package-data ships the templates + the static
assets.

## 5. Known packaging drift (recorded follow-ups)

Documentation-only migration — these are **recorded, not fixed**:

- **`[tool.setuptools.package-data]` is stale/incomplete**: it lists a
  `web/static/watch/*.js` glob for a directory that no longer exists, and it
  enumerates only *some* `lib/`/`static/` dirs — a wheel built from it would omit
  `lib/{molview,workspace,viewer,vibrationview,transport,results,spectra}/`,
  several `static/<tab>/` dirs, and all of `static/vendor/`.
- Stale docstrings: `siesta/__init__.py` and `pyscf/__init__.py` still say "input
  generation **and trajectory parsing**" though parsing now lives entirely in
  `parse/engines/`.
- `molbuilder/backends/` is a pure back-compat re-export of `builders.backends`
  — a shim the "no back-compat shims" rule would retire.
