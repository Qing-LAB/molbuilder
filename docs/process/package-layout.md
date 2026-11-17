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
| `docs/` | these docs (domain-split: `model/ science/ engines/ execution/ web/ ops/ process/`, with `plans/` (the one plan), `img/` (screenshots) and `archive/`; the pre-migration tree is archived under `docs/archive/old_docs/`) |
| `scripts/` | operator helpers (`install-env.sh`, screenshot capture, a bench script) |
| `tools/` | dev/pytest tooling (`progress_plugin.py`, `testrun.py`, screenshot capture) |
| `projects/` | runtime **project data** a running server reads/writes (`project/topic/structure/…`) |
| `pyproject.toml` | build + packaging + the pytest/lint config |
| `x3dna*/` | 3DNA, where a person unpacked it — found by the 3DNA builder (`builders/backends/_threedna.py`); git-ignored and never shipped: its licence is non-commercial, and each person downloads it from x3dna.org |

## 2. The `molbuilder/` package, by domain

### Data model + science backends → [`model/`](?doc=model/overview.md), [`science/`](?doc=science/overview.md)

The shared "nouns" and the cross-engine chemistry:

- **`structure.py`** — the `Structure` dataclass + XYZ/PDB/PySCF/ASE readers/writers; the common currency between builders and consumers ([`structure.md`](?doc=model/structure.md)).
- **`frame.py`** — `Frame` + `Trajectory` (parser-output types). **`chemistry.py`** — the shared element/adjacency helpers ([`chemistry.md`](?doc=model/chemistry.md)). **`selection.py`** — the atom-selection rule grammar. **`engine_atom_index.py`** — the 0-based ⇄ engine-index translation. **`residues.py`**, **`issues.py`** (`Issue`/`ValidationError`).
- **`config/`** — the engine-parameter dataclasses (`SiestaConfig`/`PySCFConfig`).
- **`parse/`** — the unified file/text/dir → `ParseResult` layer ([`parse.md`](?doc=model/parse.md)), split into `coords/` (geometry parsers), `engines/` (`.out`/`.log` parsers), `instruments/` (what the run script measured, `parse.md` § 5c), `dirs/` (directory composers), `sidecars/` (sidecar **read** side); the reserved blocks of a generated file are read by `deck_record.py`.
- **`sidecars/`** — the sidecar JSON **write** side (a deliberate read/write split from `parse/sidecars/`).
- **`validation/`** — the pre-emission validation package (`builders`/`chemistry`/`geometry`/`metadata`/`sidecar`/`siesta`/`pyscf`/`spectra`/`stages`/`task`/`transport`) ([`science/validation.md`](?doc=science/validation.md)). **`pseudos.py`** — SIESTA `.psml` validation ([`pseudopotentials.md`](?doc=science/pseudopotentials.md)).

### Engine emitters → [`engines/`](?doc=engines/overview.md)

- **`siesta/`** — `.fdf` generation ([`siesta.md`](?doc=engines/siesta.md)); **`pyscf/`** — `.py` script generation ([`pyscf.md`](?doc=engines/pyscf.md)); **`spectra/`** — the harmonic-vibration Spectra engine; **`transport/`** — the TranSIESTA transport engine ([`transport.md`](?doc=engines/transport.md)); **`builders/`** (+ `builders/backends/` = amber/rdkit/3dna) — the structure-build backends ([`builders.md`](?doc=engines/builders.md)).
- **`optimization/`** — an optimization's export ([`optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md) § 6); its data file, a calculation's runs and their steps, is **`stepsdata.py`**'s ([`model/datafiles.md`](?doc=model/datafiles.md) § 9.5).
- Write-side glue: **`script_emit.py`** (the reserved-block emitter), **`annotations_fdf.py`**, **`runtime_info.py`**.

### Execution + job harness → [`execution/`](?doc=execution/overview.md)

- **`cli.py`** — the `molbuilder` command (§3 of [`conventions.md`](?doc=process/conventions.md)).
- **`runwrap.py`** (the `.run.sh` wrapper), **`monitor.py`** (background job monitor), **`checkpoint.py`** (git run-checkpoints), **`persist.py`** (the `name@major` versioned-doc IO), **`task.py`** (the `task.json` codec — **the one reader both surfaces use**, [`stages.md § 6`](?doc=engines/stages.md)), **`trajectory_log/`** (the `.molwatch.log` writer).
- **`jobset/`** — the declarative JobSet framework ([`job-system.md`](?doc=execution/job-system.md)); **`bench/`** — the SIESTA benchmark harness.
- **`scheduler/`** — what a machine offers and what a job may ask: the record, the kinds (`kinds.toml`), the fit, placement and the directives ([`scheduler.md`](?doc=execution/scheduler.md)).

### Ops → [`ops/`](?doc=ops/installation.md)

- **`envs/`** — conda-env management (`recipes.py`, `install.py`, `doctor.py`, `builds.py`, …). **`diagnostics.py`** (pre-run machine snapshot), **`runtime_config.py`** (the machine-config reader — named to *not* collide with the `config/` engine-param package), **`auth_setup.py`**.

### The web app → [`web/`](?doc=web/overview.md)

`molbuilder/web/` is the Flask UI:

- **Server:** `app.py` (the app factory), `blueprints/` (the route modules — `build`, `modify`, `results`, `spectra`, `transport`, `bench`, `watch`, `files`, `docs`, `checkpoint`, `selection`, `workspace_storage`, `system_load`, `jupyter`, `notify`, `notify_setup` + `_shared.py`), `auth.py`, `admin.py`, `rate_limit.py`, `tabs.py`, `zip_stream.py`, `auth_providers/`; it reads `molbuilder/runtime_config.py` and `molbuilder/projects.py`.
- **Client:** `templates/` (one `.html` per tab + partials) and `static/` — per-tab asset dirs (`modify/`, `structure-optimization/`, `spectra/`, `task-setup/`, `results/`, `transport/`, `this-machine/`, `documents/`, `jupyternb/`), `vendor/` (3Dmol, codemirror, dompurify, gif, gitgraph, katex, marked, mermaid, and the licence texts), and **`static/lib/`** — the reusable front-end **module packages**, each owned by a web doc:

| `static/lib/…` | Doc |
|---|---|
| `molview/` | [`molview.md`](?doc=web/molview.md) |
| `frameset/` | [`structure.md`](?doc=model/structure.md) |
| `scfplot/` | [`trajectory.md`](?doc=web/trajectory.md) |
| `spectrumchart/` | [`spectrumchart.md`](?doc=web/spectrumchart.md) |
| `vibrationview/` | [`vibrationview.md`](?doc=web/vibrationview.md) |
| `workspace/` | [`workspace.md`](?doc=web/workspace.md) |
| `projects/` | [`projects.md`](?doc=web/projects.md) |
| `inspectors/` | [`presenters.md`](?doc=web/presenters.md) |
| `results/` · `trajectory/` · `spectra/` · `transport/` | [`results.md`](?doc=web/results.md) · [`trajectory.md`](?doc=web/trajectory.md) · [`spectra.md`](?doc=web/spectra.md) · [`tabs.md`](?doc=web/tabs.md) |
| the loose `lib/*.js` primitives | [`runtime.md`](?doc=web/runtime.md), [`form-schema.md`](?doc=web/form-schema.md), [`notifications.md`](?doc=web/notifications.md), [`ui-contract.md`](?doc=web/ui-contract.md) |

## 3. Tests

`tests/` is **flat at the top with a few topic subdirs** — most `test_*.py` files in the root, plus `tests/parse/`, `tests/spectra/`, `tests/validation/`, `tests/watch/`, `tests/field/`; the shared doors in `tests/support/`, the case tables in `tests/data/`, an engine's real inputs in `tests/fixtures/`; one `conftest.py`. The markers (`unit`/`module`/`interface`/`integration`/`smoke`/`e2e`/`slow`/`engine`/`needs`/`capture_on_fail`) are declared in `pyproject.toml`, and **import direction is kept by review** (`code-audit.md` § 1c (e)) — the one structural rule that keeps the layering honest. How to test is [`testing.md`](?doc=process/testing.md).

## 4. Packaging

From `pyproject.toml`: package `molbuilder` (setuptools), the console scripts
`molbuilder = molbuilder.cli:main` (one script), all code
discovered under `molbuilder/`, and optional extras (`rdkit`, `web`, `name`,
`auth`, `test`, `e2e`, `all`). Package-data ships the templates, the whole
static tree, and the data files the code reads at run time (`data/`, the
engines' and the scheduler's TOML, `envs/`' lists).

**The rule: an installed copy is the checkout.** Every file git tracks under
`molbuilder/` ships in the wheel — the scripts, styles, templates and data
files an installed copy reads as well as its modules — and every asset a page
the server serves names is one of them. The repo cannot show a break: Flask
serves `web/static/` off disk, so a pattern that misses a directory, or a file
never added to git, works in the checkout and is missing from the install.
`tests/test_wheel_ships_the_front_end.py`
builds the wheel from the tracked files and asks both halves.

The aggregate extra is `all`. The conda host env gets its test tooling from
`install-env.sh install molbuilder --with-dev-tools`, never from the `e2e`
extra: asking pip for an extra installs the *project* to reach it.

