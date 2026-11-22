# Conventions — how you write and invoke the code

**Role:** reference
**Domain:** process
**Companions:** [`package-layout.md`](?doc=process/package-layout.md) — where the
code lives; [`testing.md`](?doc=process/testing.md) — the guard tests named here;
[`execution/overview.md`](?doc=execution/overview.md), [`ops/installation.md`](?doc=ops/installation.md),
[`web/tabs.md`](?doc=web/tabs.md) — where each CLI command's *behaviour* is documented.

Two things live here: the **conventions the codebase holds** (lead with those —
each held by a test, the milestone review or review, as its item says), and the **CLI surface** (the shape
of the `molbuilder` command and the design principle behind it). Advisory,
nice-to-have style rules are called out as such so you know which lines are load-
bearing.

## 1. The conventions that are actually enforced

These are held by a check — a test, the milestone review, or review where the
item says so:

- **Layering — imports only point *down*.** The code is layered, and the two
  **surfaces** (`cli.py` and `web/`) are the top layer: nothing lower may import
  them, and no module may import from a layer above it. **Review keeps it, not a
  test** (`process/code-audit.md` § 1c (e); user, 2026-09-27) — a scan of every
  file's imports earns no place (`testing.md` § 3b, *What does NOT earn a
  place*), and what an
  upward import breaks shows up by itself: a cycle that fails every import of the
  module, or a shipped monitor file that fails beside the job
  (`tests/test_monitor_watches_a_live_run_e2e.py`).
- **The parse layer stays pure.** The block readers take a string and do **no
  I/O**, the file parsers do **no subprocess / network / threads**, and the
  parse core carries **no engine-specific names** (`model/parse.md` § 7), held
  by review. These keep the parse module reusable and testable without a
  filesystem.
- **The package is `pyflakes`-clean, checked at every milestone's review**
  (`plans/plan.md`, *How a release closes*, step 2): `"${CONDA_PREFIX:?}/bin/python" -m pyflakes $(git ls-files 'molbuilder/*.py'
  ':!:molbuilder/data/*')` answers nothing. It doesn't lint style, only "would
  this actually run" (undefined names, unused imports). pyflakes reads no
  `# noqa`, so an import kept for what importing does says so in code
  (`importlib.import_module`), never in a comment. `molbuilder/data/` is left
  out: nothing there is imported — the notebook server's config is executed by
  Jupyter, which gives it its `c`. The tests, `tools/` and `scripts/` are
  clean too: `"${CONDA_PREFIX:?}/bin/python" -m pyflakes $(git ls-files '*.py' ':!:molbuilder/*'
  ':!:tests/data/*')` answers nothing. `.pre-commit-config.yaml` runs pyflakes and a
  `node -c` syntax check on every commit where a person installs the hook;
  nothing here depends on that.
- **The tests a change touched gate its commit** — run through
  `tools/testrun.py`, chosen by what the change touched, e2e included; the whole
  suite only when that set cannot bound what could break
  ([`testing.md` § 7](?doc=process/testing.md)). **No doc rule rides that
  gate**: a test that asserts where things live rather than what they do
  earns no place (`testing.md` § 3b). The rules below are review standards,
  and the diff is what catches drift in them.
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
implementations are `lib/workspace/dispatcher.js` and `lib/validation-findings.js`.

**Be honest about its status: this one is *advisory / review-only*.** No test pins
it, and adoption is partial (only a fraction of the front-end modules actually
carry the header today). It's a good habit and the review standard, but it is not
a build gate — don't mistake it for one. A guard that reads the source text
would be a scan of the package's own source, which earns no place
(`testing.md` § 3b). The **docs** provenance header (`**Role:**`/`**Domain:**`)
is a different thing and advisory too: both stand on review — a test of a
document's shape goes red when the layout moves, not when the document is
wrong.

### Versioning — when the number moves *(the user, 2026-10-11; disciplined 2026-10-13)*

molbuilder's version has **one home**, `__version__` in `molbuilder/__init__.py`;
`pyproject.toml` reads it (`[tool.setuptools.dynamic]`). Each release is a git
tag, `v<version>` — `v2.0.0` on 2026-10-11, the first since `v1.1.0`
(2026-05-11); the releases 2.7.0 to 2.56.1 carry none. The number says **what kind of change** a version is *(the user,
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
before 2.0.0 records only its git commit and reads as *before 2.0.0*. Two
places order versions: a template written before 2.1.0, whose zeros meant
*not set* (`template.version_key`, `ZERO_MEANING_CHANGED_AT`), and
`jobset migrate`, which keeps a rewritten file under the earliest rule it was
written before (`migrate._kept_as`); every other check asks only whether two
versions are the same.

The changelog exists because downstream things — a user's run directory, a
parser, the gather that compares a run's deck with the one prep writes now —
key off what a file says. `.out` → `.pyscf.log`
([`execution/job-contracts.md § 2.2`](?doc=execution/job-contracts.md)) is the
worked example of a change nobody recorded: a one-line rename that silently
broke a viewer's dispatch.

A version once given is never renumbered: files record it, and `jobset
migrate` and the 2.1.0 comparison key off it — so the versions numbered
before the rule above stand as given *(the user, 2026-10-13: "we will
discipline from now on")*.

## 3. The CLI surface

### The design: a thin shell over the same API the web UI uses

`molbuilder = molbuilder.cli:main` — **one** console script, one spelling
for one act. The CLI is a **top-layer surface** — like the web server — and
it calls the **same lower-layer functions the blueprints call**, never a private
copy. For example: the `peptide`/`dna`/`smiles`/`name` commands and the Build
blueprint both dispatch to `build_peptide` / `build_dna` / `build_from_smiles` /
`build_from_name`; `modify` and the Modify blueprint both call
`molbuilder.modify`; `jobset prep` and the Build blueprint both reach
`runwrap.write_run_wrapper` through the same seam. The layering keeps this —
`cli` and `web` are the two top-layer surfaces (L3 with `__main__`, which only
calls `cli`), both sitting above one
shared API (review, `code-audit.md` § 1c (e)).

### The command catalogue (an index — behaviour lives in the domain docs)

The top-level commands, verbs and groups, as `molbuilder/cli.py` registers
them (`jobset`'s own verbs as `molbuilder/jobset/_cli.py` registers them):

| | | |
|---|---|---|
| `peptide` `dna` `rna` `smiles` `name` | build a structure | ([`web/tabs.md`](?doc=web/tabs.md)) |
| `modify` | one structure-edit op per call | |
| `validate` | geometry (+ optional engine) checks → Issue JSON | ([`science/validation.md`](?doc=science/validation.md)) |
| `atom-index` | which index a file's atoms are listed by, and both indices of each | |
| `describe` | write a calculation's description and its job request — not a job verb ([`execution/job-request.md`](?doc=execution/job-request.md) § 9) | |
| `jobset` | **every calculation, every engine** — prep / launch / status / summarize / migrate / export / scratch / machines / probe | ([`execution/job-system.md`](?doc=execution/job-system.md)) |
| `jobs` | the jobs launched from this machine: what each is doing, and stop one | ([`execution/job-system.md`](?doc=execution/job-system.md) § 6.0a) |
| `frameset` | a vibration mode as one multi-frame structure pair a transport calculation cites | |
| `envs` `pseudo` `checkpoint` `watch` | conda envs · pseudopotential screening · run checkpoints · trajectory parse | |
| `xv2xyz` `runtime-info` `monitor` | SIESTA `.XV`→the structure **pair** (`--from-run` also takes what the run declared — its labels, held atoms and axes — from its own deck, and what it says about itself, its `info.calculation` and `info.relaxation`) · dump a runtime-info sidecar · watch a job | |
| `auth-setup` `notify-token` `serve` `jupyter` | the auth config · a run-report key · the web UI · its notebook | ([`ops/deployment.md`](?doc=ops/deployment.md)) |

**No engine has a verb either.** A ladder is N decks, written by `jobset prep`
from a calculation's description; a verb that wrote one deck from flags would
be a second way in.

> **There is no `molbuilder run`** *(decided 2026-08-11, user)*. **Everything
> about running a job goes through `molbuilder jobset …`**: `prep` writes the
> directory and its wrapper, and `launch` runs it — `--mode direct` on a
> workstation, `--mode submit` on a scheduler
> ([`execution/job-system.md`](?doc=execution/job-system.md)). Emitting a run
> script is for what knows the target machine
> ([`running-a-job.md`](?doc=execution/running-a-job.md) § 2.1), which is
> `prep`; there is one path.

> **And there is no `molbuilder fdf`** *(decided 2026-08-11, user: "obsolete
> residue from the flat-dir design")*. A deck written straight from flags
> skips the description that makes a calculation reproducible. **Describing a calculation is
> `molbuilder describe`**, which writes the portable package — the
> template, `task.json`, `job-request.json`, the data files — and `prep` renders the deck from it on
> the machine that will run it
> ([`project-layout.md § 2.1`](?doc=execution/project-layout.md)).
>
> **The engine describes the deck; `prep` writes it.** `siesta/input.py::spec_for`
> and `pyscf/input.py::spec_for` return a `script_emit.DeckSpec`, and
> `jobset prep` renders, writes and checks it through `script_emit.prepare_deck`
> ([`engines/siesta.md`](?doc=engines/siesta.md) § 2).

**The groups:** `envs` (conda-env management →
[`ops/installation.md`](?doc=ops/installation.md)), `jobset` (staged execution →
[`execution/job-system.md`](?doc=execution/job-system.md)), `jobs` (the jobs
launched from this machine), `pseudo` (`.psml`
screening → [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md)),
`frameset` (a vibration mode as a frame set),
`checkpoint` (git run-checkpoints), `watch` (trajectory parse to JSON/NDJSON),
`serve` / `jupyter` (the web UI and its notebook).

**NO CALCULATION KIND HAS A VERB.** There is no `spectra`, no `optimization`,
no `vibration`, no `transport`. A kind is described in
`task.json` and run through `jobset`, which is decision 7 applied to the last
group that broke it: *"everything is a job set … a second way in is a second way
to lose your results."*

#### One orchestration lifecycle

**Benchmarking is `jobset`'s too** *(the user, 2026-08-17: all verbs unified
under `jobset`)*: `jobset prep bench <stage>` builds the trials, `jobset launch
bench <stage>` runs them, `jobset summarize bench <stage>` reads them and
prints the execution block to copy, and `jobset prep task --group <N>` reads
what you wrote ([`job-system.md`](?doc=execution/job-system.md) § 7); a
machine's record — its cores, GPUs, scheduler and queues — is written by
`jobset probe`. A bias scan is one submission whose walker chains the points
inside one job, so the job set carries no dependency vocabulary.

**Why one framework:**
[`project-layout.md § 2.3.1a`](?doc=execution/project-layout.md) — *benchmarking
is `prep`, specialised.* A normal prep resolves one configuration and renders one
deck; a benchmark prep resolves a **grid** and renders one deck per point.
Machine detection, activation and the directory build are the same framework
doing the same thing, so there was never a second system to integrate — only a
general part to lift out of where the need first appeared.

**`bench` is a positional, alongside `task`.** `jobset prep <task|bench> [STAGE]`.
The STAGE is required for a benchmark, because a sweep belongs to one stage
rather than to the calculation ([`generator.md`](?doc=execution/generator.md) § 4.3a).

### The CLI-wide conventions

- **Exit codes:** `--help` → 0; a usage / unknown-command / bad-argument error → 2;
  a domain error → 1. A few commands deliberately use 2 for a *domain* problem the
  user must fix before anything runs (`validate --exit-on-error`) — the number
  tells a script "stop, this won't work"; a refusal at `jobset prep`, a missing
  pseudopotential among them, exits 1.
- **A help names only what runs.** Every command a group's `--help` tells a
  person to run is marked ` ``name`` ` — in that group or from the root — or
  written as a `molbuilder …` line of an example, and is a command of the
  tree; a help that sends someone to a verb that is gone is worse than none
  (`test_cli.py`, every group of the tree). Single backticks are for the rest
  of the code a help quotes — a flag, a file, a key.
- **stdout vs stderr:** structure/data output goes to **stdout**, the human summary
  to **stderr**, and `-` means stdin/stdout — so `molbuilder … | molbuilder …`
  pipes cleanly.
- **No flag is made from a config field.** A calculation's settings are its
  template's items, from the one catalogue every surface reads
  (`molbuilder/data/catalogue.template.toml`,
  [`template.md`](?doc=engines/template.md) § 4.3); `describe`'s options —
  the structure, the engine, the calculation, the stage ladder, and what a
  transport calculation cites and sweeps — are declared by hand in
  `jobset/_cli.py`.

## 4. Where the guards live (test map)

- `test_monitor_watches_a_live_run_e2e.py` — the monitor's shipped files run
  beside a run made on the road, with molbuilder absent (the layering itself is review's).
- `test_pyscf_bundle_runs_alone.py` — every module the PySCF script imports loads
  from `mb_pyscf.pyz` with molbuilder absent, and writes a pair the package's codec
  reads back.
- `test_negative_body_assert_lint.py` — the status-guarded-assert meta-lint.
- `test_cli.py` (+ `test_cli_runtime_info.py`, `test_cli_tls.py`) — every
  subcommand's `--help`, and a build verb building what was typed (the real
  builders).

A test map names only files that exist; one naming a missing file is the
reason somebody believes a rule is guarded. The docs structure rules of § 1–2
are held by review alone ([`testing.md`](?doc=process/testing.md) § 3b,
*Every test states its GOAL and its CONTRACT*).
