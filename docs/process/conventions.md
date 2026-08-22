# Conventions — how you write and invoke the code

**Role:** reference
**Domain:** process
**Companions:** [`package-layout.md`](?doc=process/package-layout.md) — where the
code lives; [`testing.md`](?doc=process/testing.md) — the guard tests named here;
[`execution/overview.md`](?doc=execution/overview.md), [`ops/installation.md`](?doc=ops/installation.md),
[`web/tabs.md`](?doc=web/tabs.md) — where each CLI command's *behaviour* is documented.

Two things live here: the **conventions the codebase enforces** (lead with those —
each is backed by a test that fails the build), and the **CLI surface** (the shape
of the `molbuilder` command and the design principle behind it). Advisory,
nice-to-have style rules are called out as such so you know which lines are load-
bearing.

## 1. The conventions that are actually enforced

These are gated — a violation fails `pytest` (and therefore the pre-commit hook):

- **Layering — imports only point *down*.** The code is layered, and the two
  **surfaces** (`cli.py` and `web/`) are the top layer: nothing lower may import
  them, and no module may import from a layer above it.
  `tests/test_layering.py` walks every `molbuilder/*.py` and asserts both the
  import direction *and* that every top-level name is classified into a layer (so a
  new module can't silently escape the check).
- **The parse layer stays pure.** The memory-only text parsers do **no I/O**
  (`tests/parse/test_scripts.py::test_text_parsers_do_no_io`), the file parsers do
  **no subprocess / network / threads**, and the parse core carries **no
  engine-specific names** (`tests/parse/test_audit_gaps.py`). These keep the
  parse module reusable and testable without a filesystem.
- **`pyflakes` on every commit.** A pre-commit hook runs pyflakes
  (`.pre-commit-config.yaml`) — it doesn't lint style, only "would this actually
  run" (undefined names, unused imports). Plus a `node -c` syntax check on any
  changed `*.js`.
- **The full test suite gates every commit.** The pre-commit config runs `pytest -m
  "not slow"` — and it deliberately does **not** exclude `e2e` as a class (the
  comment in the config says so outright: gate on the tests that catch production
  bugs); it only `--ignore`s one chromium-heavy file. The migration's own doc rules
  ride the same gate via `tests/test_docs_structure.py`.
- **HTTP negative-body assertions must be status-guarded** — an AST meta-lint over
  the whole test suite (`tests/test_negative_body_assert_lint.py`) so a test can't
  assert "the body doesn't contain X" without first pinning the status code.

Two things worth internalising from the list: there is **no CI** — all of this is
enforced by the **local pre-commit hook** (there is no `.github/workflows/`); and
the code runs from the tree (`pythonpath=["."]`, **no `pip install -e`**).

## 2. The module-provenance header (advisory)

The house style is that a file opens with a short **`MODULE · ROLE · USED-BY`**
header so a reader knows what it is and who depends on it — the reference
implementations are `lib/workspace/dispatcher.js` and `snapshot-io.js`.

**Be honest about its status: this one is *advisory / review-only*.** No test pins
it, and adoption is partial (only a fraction of the front-end modules actually
carry the header today). It's a good habit and the review standard, but it is not
a build gate — don't mistake it for one. (Fixing that — either an AST/text guard
like `test_layering.py`, or softening the "mandatory" wording — is a recorded
follow-up.) Note this is *not* the same as the **docs** provenance header
(`**Role:**`/`**Domain:**`), which *is* enforced, by `test_docs_structure.py`.

### Versioning — what counts as a breaking change

molbuilder is a **1.x project** (`1.1.0` today, pinned in `pyproject.toml` and
`molbuilder/__init__.py`). The version tracks the *promises the code makes about
files it writes*, not the size of the diff:

- **Removing or renaming a promised output file** — an emitted script, a sidecar,
  a bundle member — is a **minor bump** (1.x → 1.x+1) and needs a note in the
  owning domain doc saying what moved and what reads the old name.
- **Adding an optional field or a new file** is a **patch**. Existing readers keep
  working, so nothing breaks.

The rule exists because downstream things — a user's run directory, a parser, a
sidecar reader — key off those names. `.out` → `.pyscf.log`
([`execution/job-contracts.md § 2.2`](?doc=execution/job-contracts.md)) is the
worked example: a one-line rename that silently broke a viewer's dispatch.

## 3. The CLI surface

### The design: a thin shell over the same API the web UI uses

`molbuilder = molbuilder.cli:main` (plus a back-compat `molwatch` that now maps to
`molbuilder serve`). The CLI is a **top-layer surface** — like the web server — and
it calls the **same lower-layer functions the blueprints call**, never a private
copy. For example: the `peptide`/`dna`/`smiles`/`name` commands and the Build
blueprint both dispatch to `build_peptide` / `build_dna` / `build_from_smiles` /
`build_from_name`; `modify` and the Modify blueprint both call
`molbuilder.modify`; `jobset prep` and the Build blueprint both reach
`runwrap.write_run_wrapper` through the same seam. The layering test guarantees
this — `cli` and `web` are the only two top-layer modules, both sitting above one
shared API.

### The command catalogue (an index — behaviour lives in the domain docs)

**13 top-level commands:**

| | | |
|---|---|---|
| `peptide` `dna` `rna` `smiles` `name` | build a structure | ([`web/tabs.md`](?doc=web/tabs.md)) |
| `pyscf` | XYZ/PDB → a PySCF run script | ([`execution/`](?doc=execution/overview.md)) |
| `validate` | geometry (+ optional engine) checks → Issue JSON | ([`science/validation.md`](?doc=science/validation.md)) |
| `modify` | one structure-edit op per call | |
| `xv2xyz` `runtime-info` `monitor` | SIESTA `.XV`→xyz · dump a runtime-info sidecar · watch a job | |
| `auth-setup` `serve` | generate the auth config · run the web UI | ([`ops/deployment.md`](?doc=ops/deployment.md)) |

> **There is no `molbuilder run`** *(decided 2026-08-11, user)*. **Everything
> about running a job goes through `molbuilder jobset …`**: `prep` writes the
> directory and its wrapper, and `launch` runs it — `--mode direct` on a
> workstation, `--mode submit` on a scheduler
> ([`execution/job-system.md`](?doc=execution/job-system.md)). `run` was the
> pre-job-system entry point: it emitted a wrapper for one hand-made deck, which
> only something that knows the target machine may do
> ([`running-a-job.md`](?doc=execution/running-a-job.md) § 2.1). It is deleted,
> not deprecated — there is no second path to keep working.

> **And there is no `molbuilder fdf`** *(decided 2026-08-11, user: "obsolete
> residue from the flat-dir design")*. It is the same shape as `run` one step
> earlier: it wrote a **finished deck** straight from CLI flags, and
> `fdf --jobset` wrote a whole flat bundle — both skipping the description that
> makes a calculation reproducible. **Describing a calculation is
> `molbuilder jobset init`**, which writes the portable package — the
> template, `task.json`, the data files — and `prep` renders the deck from it on
> the machine that will run it
> ([`project-layout.md § 2.1`](?doc=execution/project-layout.md)).
>
> **The emitter is untouched.** `render_fdf` and `convert`
> ([`engines/siesta.md`](?doc=engines/siesta.md) § 2) are the Python API and stay
> exactly as they are — what is deleted is the *top-level verb* that let a person
> reach them without a description. `pyscf` survives for now because PySCF's
> ladder runs inside one emitted script rather than as a job set
> ([`stages.md § 1`](?doc=engines/stages.md)); it goes the same way when that
> path is reworked.

**7 sub-groups:** `envs` (conda-env management → [`ops/installation.md`](?doc=ops/installation.md)),
`bench` (the CPU-vs-GPU benchmark), `jobset` (staged execution →
[`execution/job-system.md`](?doc=execution/job-system.md)), `transport`
(TranSIESTA helpers), `pseudo` (`.psml` screening →
[`science/pseudopotentials.md`](?doc=science/pseudopotentials.md)), `checkpoint`
(git run-checkpoints), `watch` (trajectory parse to JSON/NDJSON).

#### Three orchestration lifecycles, where the design says one

> ⚠ **`jobset`, `bench` and `transport` each run calculations their own way.**
> *Named as a design problem 2026-08-11. Two are the same act with two spellings;
> the third is a case the unified design cannot yet express.*

| group | how it runs work | status |
|---|---|---|
| **`jobset`** | `prep` → `launch`, one job per invocation, per-attempt directories, `run.json` | the design ([`job-system.md`](?doc=execution/job-system.md)) |
| **`bench`** | `probe-scheduler` only | **RESOLVED 2026-08-17** — the four duplicate verbs are gone; benchmarking is `jobset prep bench <stage>`. See below |
| **`transport`** | `bundle` → `bash run-transport.sh`, a driver that **chains** three coupled runs | **the case with no representation** ([`transport.md § 8`](?doc=engines/transport.md)) |

**They are not the same kind of problem, and conflating them would fix neither.**
`bench` duplicates something that exists; `transport` does something the model
cannot say. The first is a merge, the second needs a vocabulary — job-set edges
between genuinely coupled runs, which decision 6 removed for ladders on purpose
and would have to come back differently.

##### `bench` and `jobset` — RESOLVED, and the record was inverted

> **This section described four acts with two spellings each and said *"the
> `jobset` column is the one that does not exist yet"*. That was true when it
> was written and is the exact inverse of today** (measured 2026-08-17): the
> `bench` command's four verbs were deleted in the 2026-08-12 fold, and all
> three `jobset` forms work.

| the act | today | the second spelling it replaced |
|---|---|---|
| build it | `jobset prep bench <stage>` | ~~`bench generate` + `bench prep`~~ |
| run it | `jobset launch bench <stage>` | ~~`bench siesta-gpu`~~ |
| read it | `jobset summarize bench <stage>` | ~~`bench summarize`~~ |
| use the answer | `jobset prep run <stage>` — the verdict is **offered** and waits | ~~`bench prep-run`~~ |

**`molbuilder bench` is gone entirely** (2026-08-17, user: *all verbs unified
under `jobset`*). Its last inhabitant was `probe-scheduler`, which reads
`sinfo`/`sacctmgr` and proposes a scheduler config block — **never a benchmark
verb at all**, and keeping the group alive for it left a name that described
nothing it contained. It is now `molbuilder jobset probe`.

**Why it resolved this way, and it is the reason predicted here:**
[`project-layout.md § 2.3.1a`](?doc=execution/project-layout.md) — *benchmarking
is `prep`, specialised.* A normal prep resolves one configuration and renders one
deck; a benchmark prep resolves a **grid** and renders one deck per point.
Machine detection, activation and the directory build are the same framework
doing the same thing, so there was never a second system to integrate — only a
general part to lift out of where the need first appeared.

**`bench` is a positional, alongside `run`.** `jobset prep <run|bench> [STAGE]`
— which settles `generator.md` § 9's open question G3. And the STAGE is
required for a benchmark, because a sweep belongs to one stage rather than to
the calculation ([`generator.md § 4.3a`](?doc=execution/generator.md)).

### The CLI-wide conventions

- **Exit codes:** `--help` → 0; a usage / unknown-command / bad-argument error → 2;
  a domain error → 1. A few commands deliberately use 2 for a *domain* problem the
  user must fix before anything runs (e.g. a missing pseudopotential at
  `jobset prep`, `validate --exit-on-error`) — the number tells a script "stop,
  this won't work."
- **stdout vs stderr:** structure/data output goes to **stdout**, the human summary
  to **stderr**, and `-` means stdin/stdout — so `molbuilder … | molbuilder …`
  pipes cleanly.
- **`--help` is generated from the config dataclasses.** The engine flags are
  derived from `SiestaConfig`/`PySCFConfig` field metadata, so a `choices=` field
  becomes a `click.Choice` that fails at parse time (exit 2) — the CLI and the web
  form expose the *same* options from one source. That is also what makes
  `jobset init` possible without a second flag list: the template it writes is
  generated from the same metadata ([`template.md § 5`](?doc=engines/template.md)).

## 4. Where the guards live (test map)

- `test_layering.py` — the import-direction + full-classification gate.
- `parse/test_scripts.py`, `parse/test_audit_gaps.py` — the parse-layer purity gates.
- `test_negative_body_assert_lint.py` — the status-guarded-assert meta-lint.
- `test_docs_structure.py` — the docs migration/structure rules.
- `test_cli.py` (+ `test_cli_run.py`, `test_cli_runtime_info.py`,
  `test_cli_siesta_stages.py`, `test_cli_tls.py`) — every subcommand's `--help`,
  routing, and the dataclass→flag bridge.

> **Migration note.** The legacy `cli.md` was stale in several places, corrected
> here against code: `validate` uses `--engine` (not `--config`), defaults to
> geometry-only, emits a JSON *object*, and exits non-zero only with
> `--exit-on-error`; the CLI *does* have dedicated tests (five files, not "none");
> and enforcement is pre-commit, not CI. The full command families
> (`envs`/`jobset`/`transport`/`checkpoint`/…) that the legacy table
> omitted are included above.
