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

These are held by a check — a test, the milestone review, or review where the
item says so:

- **Layering — imports only point *down*.** The code is layered, and the two
  **surfaces** (`cli.py` and `web/`) are the top layer: nothing lower may import
  them, and no module may import from a layer above it. **Review keeps it, not a
  test** (`process/code-audit.md` § 1c (e); user, 2026-09-27) — the scan of every
  file's imports that did until then earns no place (`testing.md`), and what an
  upward import breaks shows up by itself: a cycle that fails every import of the
  module, or a shipped monitor file that fails beside the job
  (`tests/test_monitor_watches_a_live_run_e2e.py`).
- **The parse layer stays pure.** The block readers take a string and do **no
  I/O** (`tests/test_script_emit.py::test_the_block_readers_do_no_io`), the file
  parsers do **no subprocess / network / threads**, and the parse core carries
  **no engine-specific names** (`tests/parse/test_audit_gaps.py`). These keep the
  parse module reusable and testable without a filesystem.
- **The package is `pyflakes`-clean, checked at every milestone's review**
  (`plans/plan.md` § 0a): `python -m pyflakes $(git ls-files 'molbuilder/*.py'
  ':!:molbuilder/data/*')` answers nothing. It doesn't lint style, only "would
  this actually run" (undefined names, unused imports). pyflakes reads no
  `# noqa`, so an import kept for what importing does says so in code
  (`importlib.import_module`), never in a comment. `molbuilder/data/` is left
  out: nothing there is imported — the notebook server's config is executed by
  Jupyter, which gives it its `c`. The tests, `tools/` and `scripts/` are not
  clean yet (plan § 2, PYF-T). `.pre-commit-config.yaml` runs pyflakes and a
  `node -c` syntax check on every commit where a person installs the hook;
  nothing here depends on that.
- **The tests a change touched gate its commit** — run through
  `tools/testrun.py`, chosen by what the change touched, e2e included; the whole
  suite only when that set cannot bound what could break
  ([`testing.md` § 7](?doc=process/testing.md)). **No doc rule rides that
  gate**: `tests/test_docs_structure.py` was retired in `082ba979`, with the 60
  others that asserted where things live rather than what they do. The rules
  below are review standards, and the diff is what catches drift in them.
- **HTTP negative-body assertions must be status-guarded** — an AST meta-lint over
  the whole test suite (`tests/test_negative_body_assert_lint.py`) so a test can't
  assert "the body doesn't contain X" without first pinning the status code.

Two things worth internalising from the list: there is **no CI** — all of this is
held locally, by review and the tests run before a commit (there is no
`.github/workflows/`); and the code runs from the tree (`pythonpath=["."]`,
**no `pip install -e`**).

## 2. The module-provenance header (advisory)

The house style is that a file opens with a short **`MODULE · ROLE · USED-BY`**
header so a reader knows what it is and who depends on it — the reference
implementations are `lib/workspace/dispatcher.js` and `snapshot-io.js`.

**Be honest about its status: this one is *advisory / review-only*.** No test pins
it, and adoption is partial (only a fraction of the front-end modules actually
carry the header today). It's a good habit and the review standard, but it is not
a build gate — don't mistake it for one. (Fixing that is softening the
"mandatory" wording: a guard that reads the source text would be a scan of the
package's own source, which earns no place — `testing.md`.) Note this is *not* the same as the **docs** provenance header
(`**Role:**`/`**Domain:**`) — which is **also advisory now**. It was enforced by
`test_docs_structure.py` until `082ba979` retired that file for asserting the
shape of the repository rather than a result: when such a test went red the docs
were not wrong, the layout had moved. So the contrast this paragraph draws is
gone, and both headers stand on review.

### Versioning — when the number moves *(the user, 2026-10-11; disciplined 2026-10-13)*

molbuilder's version has **one home**, `__version__` in `molbuilder/__init__.py`;
`pyproject.toml` reads it (`[tool.setuptools.dynamic]`). Each release is a git
tag, `v<version>` — `v2.0.0` on 2026-10-11, the first since `v1.1.0`
(2026-05-11). The number says **what kind of change** a version is *(the user,
2026-10-13: "middle number means new functionality, major number means major
framework change")*:

| the number | moves when | example |
|---|---|---|
| **first** (**x**.0.0) | **a major framework change** — the user declares it | 2.0.0 |
| **middle** (2.**x**.0) | **new functionality** — something a person can do that they could not before | the groups: every stage states its group, prep prepares every group at once, a group's first launch gathers what it builds on (2.7.0) |
| **last** (2.7.**x**) | **everything else that changes what molbuilder does or writes** — a fix, a consolidation, a corrected default or check, a changed warning, wording, comments, help — **including a change to what a file says** | the transport k-point consolidation and its wording (2.8.0 and 2.9.0, had this rule held) |

**The discipline:**

* **One move per push.** The version moves once for what a push to `main`
  holds, by the largest kind of change in it; work committed between two pushes
  shares that number. A version is never moved twice in one push, and a push
  that changes only documents moves nothing.
* **When in doubt, the last number.** New functionality is a new thing a
  person can do — a new kind of calculation, a new verb, a new surface — not
  a better way of doing what they already could. The first number is the
  user's to declare, never inferred.
* **Every version has its line in the [changelog](?doc=process/changelog.md)**
  — the version, the date, what changed and in which file molbuilder writes —
  written in the commit that moves the number, a last-number move as much as a
  middle one: what a file says is still recorded, it just no longer decides the
  number.

**Every deck and run script records the version that wrote it, and its git
commit** (`generator-version 2.9.0 (git …)`,
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 3.2), so a
check that meets a file from another molbuilder names both versions instead
of calling the file different (*"prepared by molbuilder 2.8.0; this is
2.9.0"*), and the commit tells two builds of one version apart; a file written
before 2.0.0 records only its git commit and reads as *before 2.0.0*. Only one
comparison orders versions — a template written before 2.1.0, whose zeros
meant *not set* (`template.version_key`, `ZERO_MEANING_CHANGED_AT`); every
other check asks only whether two versions are the same.

The changelog exists because downstream things — a user's run directory, a
parser, the gather that compares a run's deck with the one prep writes now —
key off what a file says. `.out` → `.pyscf.log`
([`execution/job-contracts.md § 2.2`](?doc=execution/job-contracts.md)) is the
worked example of a change nobody recorded: a one-line rename that silently
broke a viewer's dispatch.

> **2.1.0 – 2.9.0 were numbered too loosely** *(2026-10-11 → 13)*. The rule
> that stood then moved the middle number for **any** change to what molbuilder
> writes into a file, and it was applied per commit: nine middle-number moves in
> three days. Several were no new functionality — a validation note reworded
> (2.2.0), a consolidation of the transport k-point grids (2.8.0), a deck's
> comments reworded (2.9.0) — and two moved in one push (2.7.0 and 2.8.0,
> commit `85164f2e`): last-number changes by the rule above. The numbers already given
> stand — files record them, and `jobset migrate` and the 2.1.0 comparison key
> off them — and every version from 2.9.0 on is numbered by the rule above
> *(the user, 2026-10-13: "we will discipline from now on")*.

## 3. The CLI surface

### The design: a thin shell over the same API the web UI uses

`molbuilder = molbuilder.cli:main` — **one** console script; the back-compat
`molwatch` entry was deleted 2026-09-17 and `pyproject.toml` carries its
gravestone. The CLI is a **top-layer surface** — like the web server — and
it calls the **same lower-layer functions the blueprints call**, never a private
copy. For example: the `peptide`/`dna`/`smiles`/`name` commands and the Build
blueprint both dispatch to `build_peptide` / `build_dna` / `build_from_smiles` /
`build_from_name`; `modify` and the Modify blueprint both call
`molbuilder.modify`; `jobset prep` and the Build blueprint both reach
`runwrap.write_run_wrapper` through the same seam. The layering keeps this —
`cli` and `web` are the only two top-layer modules, both sitting above one
shared API (review, `code-audit.md` § 1c (e)).

### The command catalogue (an index — behaviour lives in the domain docs)

**19 top-level commands** *(re-derived from `cli.commands` 2026-09-17; this
table said 13, listed `pyscf` — deleted that day — and omitted `notify-token`,
`checkpoint`, `envs`, `jobset`, `jupyter`, `pseudo` and `watch`):*

| | | |
|---|---|---|
| `peptide` `dna` `rna` `smiles` `name` | build a structure | ([`web/tabs.md`](?doc=web/tabs.md)) |
| `modify` | one structure-edit op per call | |
| `validate` | geometry (+ optional engine) checks → Issue JSON | ([`science/validation.md`](?doc=science/validation.md)) |
| `jobset` | **every calculation, every engine** — init / prep / launch / status / summarize | ([`execution/job-system.md`](?doc=execution/job-system.md)) |
| `envs` `pseudo` `checkpoint` `watch` | the four remaining sub-groups | |
| `xv2xyz` `runtime-info` `monitor` | SIESTA `.XV`→the structure **pair** (`--from-run` also takes what the run declared — its labels, held atoms and axes — from its own deck, and what it says about itself, its `info.calculation` and `info.relaxation`) · dump a runtime-info sidecar · watch a job | |
| `auth-setup` `notify-token` `serve` `jupyter` | the auth config · a run-report key · the web UI · its notebook | ([`ops/deployment.md`](?doc=ops/deployment.md)) |

**NO ENGINE HAS A VERB EITHER** *(2026-09-17)*. `molbuilder pyscf` was the last
one — `fdf`'s surviving twin, taking a structure plus every engine field as a
flag and writing a finished deck. It kept its flags on the stated grounds that
*"its ladder runs inside one emitted script"*, and `cli.py` refuted that beside
it: a ladder is N decks and that command wrote one, which is why
`--stage-strategy` was taken off it on 2026-08-18. Both engines' single-shot
`convert()` workers went with their verbs.

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
> **The engine describes the deck; `prep` writes it.** `siesta/input.py::spec_for`
> and `pyscf/input.py::spec_for` return a `script_emit.DeckSpec`, and
> `jobset prep` renders, writes and checks it through `script_emit.prepare_deck`
> ([`engines/siesta.md`](?doc=engines/siesta.md) § 2).

**6 sub-groups** *(re-derived 2026-09-17, not decremented — this row said
seven and named `bench`, deleted 2026-08-17, and `transport`, deleted
2026-09-17):* `envs` (conda-env management →
[`ops/installation.md`](?doc=ops/installation.md)), `jobset` (staged execution →
[`execution/job-system.md`](?doc=execution/job-system.md)), `pseudo` (`.psml`
screening → [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md)),
`checkpoint` (git run-checkpoints), `watch` (trajectory parse to JSON/NDJSON),
`serve` / `jupyter` (the web UI and its notebook — verb groups since 2026-08-28).

**NO CALCULATION KIND HAS A VERB.** There is no `spectra`, no `optimization`,
no `vibration` — and since 2026-09-17 no `transport`. A kind is described in
`task.json` and run through `jobset`, which is decision 7 applied to the last
group that broke it: *"everything is a job set … a second way in is a second way
to lose your results."*

#### One orchestration lifecycle (the 2026-08-11 problem, closed)

> **`jobset`, `bench` and `transport` each ran calculations their own way** —
> named as a design problem 2026-08-11, closed in two moves:
> `bench` folded 2026-08-17 (benchmarking is `jobset prep bench <stage>` — the
> four duplicate verbs died), and `transport` migrated 2026-08-29 (the
> composite: one citation → five stages through the ordinary jobset verbs;
> the `bundle` verb and its bash chain driver deleted,
> [`archive/2026-09-01-transport-design.md`](?doc=archive/2026-09-01-transport-design.md)).  No edges
> came back: a bias scan is ONE submission whose walker chains the points'
> `.TSDE` inside a single job, so the job set still carries no dependency
> vocabulary.

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
| use the answer | `jobset prep task --group <N>` — the verdict is **offered** and waits | ~~`bench prep-run`~~ |

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

- `test_monitor_watches_a_live_run_e2e.py` — the monitor's shipped files run
  beside a run made on the road, with molbuilder absent (the layering itself is review's).
- `test_pyscf_bundle_runs_alone.py` — every module the PySCF script imports loads
  from `mb_pyscf.pyz` with molbuilder absent, and writes a pair the package's codec
  reads back.
- `parse/test_scripts.py`, `parse/test_audit_gaps.py` — the parse-layer purity gates.
- `test_negative_body_assert_lint.py` — the status-guarded-assert meta-lint.
- `test_cli.py` (+ `test_cli_runtime_info.py`, `test_cli_tls.py`) — every
  subcommand's `--help`, routing, and the dataclass→flag bridge.

> **Three names left this list on 2026-09-12, having been deleted on
> 2026-09-10** in the sweep that retired 18 files asserting the shape of the
> repository rather than a result: `test_docs_structure.py` (the docs
> migration/structure rules), `test_cli_run.py` and
> `test_cli_siesta_stages.py`. A test map naming a file that is not there is
> worse than a short one — it is the reason somebody believes a rule is guarded.
> **The docs structure rules in § 1–2 of this page are now held by review
> alone**, which is the deliberate consequence of that sweep and not an
> oversight; see [`testing.md`](?doc=process/testing.md) § *a stated contract
> also fails usefully*.

> **Migration note.** The legacy `cli.md` was stale in several places, corrected
> here against code: `validate` uses `--engine` (not `--config`), defaults to
> geometry-only, emits a JSON *object*, and exits non-zero only with
> `--exit-on-error`; the CLI *does* have dedicated tests (five files, not "none");
> and nothing ran in CI. The full command families
> (`envs`/`jobset`/`transport`/`checkpoint`/…) that the legacy table
> omitted are included above.
