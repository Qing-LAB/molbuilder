# molbuilder — project profile (contract-discipline)

- code globs: `molbuilder/*.py`, `molbuilder/**/*.py`, `molbuilder/**/*.js`, `molbuilder/**/*.css`, `molbuilder/**/*.html`, `molbuilder/**/*.toml`, `molbuilder/**/*.sh`, `tests/*.py`, `tests/**/*.py`, `tests/data/*.toml`, `tools/*.py`, `scripts/*.sh`
- destructive commands: refuse
- busy lock: `.test-progress/*.lock`
- python: conda env molbuilder
- package: molbuilder
- env: MOLBUILDER_CONFIG_DIR={run}/config, MOLBUILDER_PROJECTS={run}/projects
- plan: `docs/plans/plan.md`

## Where the rules live

- The chapters under `docs/` are the rules; `docs/toc.json` is their index,
  regenerated only by `GET /api/docs/toc`, never by hand.
- `docs/execution/architecture.md` names each module's floor and the door
  (the one function) other code asks for each fact; "one home" means a fact
  is stated at its door and asked everywhere else.
- `docs/execution/job-contracts.md` owns the file catalogue (`runfiles`), run
  files, warm files and scratch; `docs/execution/checkpointing.md` the saves;
  `docs/execution/job-system.md` and `job-request.md` the jobs; `docs/model/`
  the data files and structures; `docs/web/` the pages; `docs/engines/` each
  engine and calculation kind.
- `docs/archive/` is history, not rules.

## Layers

Two layerings, both kept by review (`docs/execution/architecture.md` § 1):

- **import depth** — L1, L2, L3: an L1 module imports nothing above it, L2
  may import L1, and only the two surfaces, `cli` and `web` (L3), may import
  both (`docs/architecture.md` § 3; `docs/process/code-audit.md` § 1c (e));
- **floors** — 1 to 7, the surfaces last (`docs/execution/architecture.md`
  § 2, § 2.1's table; rule A7, with the exceptions it names).

An import pointing up either layering is a finding wherever it sits — at a
module's top or inside a function — apart from A7's named exceptions.

## Code that runs elsewhere

Modules copied beside a job and imported there without molbuilder: the
members of each bundle in `runwrap.BUNDLES` (`runwrap.members`) -- the
monitor's, a SIESTA vibration's finish and the PySCF script's imports --
all gathered by one closure (`runwrap.bundle_members`) from each module's `TRAVELS_WITH`. Each imports the other members both ways
(`try: from .x import ...` / `except ImportError: from x import ...`) and
declares `TRAVELS_WITH` for what it imports. Check: build the bundle in
scratch and import every member with the job env's own Python, by its path,
`-I` and no molbuilder on the path; `tests/test_monitor_notify_policy.py::test_each_bundle_loads_on_the_packages_its_row_allows_alone`
does every bundle on its row's packages alone.

## What runs in an env is the env's own

`docs/ops/installation.md` M6 (the user, 2026-11-21: *"make sure that no bare
python or out-of-env python is ever tried. period."*; *"apply the same rule to
ctest, make, mpirun and tleap too"*). Wherever the touched code or text starts
a Python or an env's program, or tells a person to: inside an env, the door
(`envs.builds.dispatch_into_env`: no active env handed to the manager,
`env_python_argv`) names it, a routed tool runs by its path in its env
(`envs.run_tool`), a shell line names `"${CONDA_PREFIX:?}/bin/<program>"`, a
probe names `<prefix>/bin/<program>`, a job's Python is `$_mb_py`, its program
the first on PATH in a folder of `$CONDA_PREFIX` and its MPI
`$CONDA_PREFIX/bin/mpirun`, else it stops (`runwrap._program_launcher_block`);
a build step names the env's tools by `{env_prefix}/bin/...`; molbuilder's
own children are `sys.executable`. A bare `python`/`python3`/`jupyter` or an
env's program left to `PATH`, `shutil.which(...)` for an env's program, a
fallback to the host's, a `#!/usr/bin/env python` line, or a `pip install`
remedy is a finding at every review — activation does not put an env's `bin`
first. Check: `tests/test_envs_enters_the_env_through_the_manager.py`'s M6
rows (decoys first on `PATH` in a shell with an env active, the real manager)
and `tests/test_launch_door_gate.py`'s job rows.

## Shared answers to check first

- `molbuilder/runfiles.py` — every file name, role, field and glob (`family`,
  `transient_families`); `warmfiles` — warm lists and the engines' scratch
  (`scratch_families`, `never_carried`) — read by the checkpoint, the
  generated `.gitignore`, the walk's shell lines, downloads and copies.
- `template` / `resolve` — catalogue items, the run card, the grid.
- `identity` — labels, stage tokens, the label cap.
- The generated run scripts and decks (`runwrap`, `script_emit`): a change
  there reaches every job a person prepares.

## Running the project's code in isolation

The settings above give the comparison tool the `molbuilder` conda
environment's python, the package at the repository root, and configuration
and projects folders of its own per side (home and XDG are isolated by the
tool itself).

## Tests

- Always `tools/testrun.py` under the `molbuilder` environment, from the
  repository root — never raw pytest: `run <files or nodes>` for a targeted
  set (by node id; `-k` is refused); default workers.
- The whole fast suite is `run none2e` (about seven minutes); it is never a
  default release step — only when a targeted set cannot bound the change,
  said why first. The slow browser and engine tests run once, last, and only
  the named files (`run tests/<file>_e2e.py ...`), never the whole `run e2e`
  batch unless the person asks. A batch holds `.test-progress/<batch>.lock`.
- `tests/support/` (the road helpers) is imported only under pytest, never by
  a bare script. A test writes only to its temp folders.
- One test on the old release: `compare_head.py --ref <tag> --unpack-to
  <scratch>/past`, copy the new test file and any case-table row it reads
  (`tests/data/*.toml`) into `<scratch>/past`, then from `<scratch>/past` run
  `<python> tools/testrun.py run tests/<file>::<test>`.

## Pages

- Node-run page tests (`tests/test_*_js.py`, through `tests/_node_esm.py`)
  run a page's module against the app's own routes; the browser tests are the
  `_e2e` files. A reviewer runs none of them in the repository (a run holds
  the batch lock and writes `.test-progress/`); a node page test only in an
  `--unpack-to` copy. After a page script changes, the main session runs the
  e2e test that opens its page, after the review — `node --check` misses
  runtime errors.
- MolView and VibrationView are sealed apart: neither is judged by the
  other's code, and a finding across the seal is dropped.
- Widths and layout: as `docs/web/` states for each page.

## Review

- Two rounds of new reviewers (round 2 never the round-1 ones), then two new
  independent reviewers each reading every touched module and chapter section
  whole; every round reads the involved files whole, including the round after
  the last fixes. A discipline reader at every release with code.
- The triage question for every finding: does it break molbuilder's own
  workflow? If not, it is dropped.
- Findings grouped by owning door; a design change goes to the person before it
  is fixed.

## Not a finding here

- Nothing stops a calculation the person set: a scientific concern is said
  and the run goes as set; an engine's own stop is said as information.
- The run script's two zones are the person's to type in; nothing is checked
  or reported about what they write there.
- "The person could change X by hand" is no defect; foreign input gets a
  warning, not a refusal.
- No work goes into reading files older releases wrote beyond saying they are
  not read.
- A calculation is run as a jobset run (`init → prep → launch`), never by a
  hand-written script; a test or review never starts one any other way.

## Off limits

`~/.config/molbuilder`, `~/.local/state/molbuilder`, `projects/`, and the
environments and their install files (changed only with the person's yes). No
server is started; the dev server is the person's.

## Releases

- Version: `molbuilder/__init__.py` `__version__`; changelog:
  `docs/process/changelog.md` (one row per release, newest first); plan:
  `docs/plans/plan.md`.
- Review records: `docs/archive/audits/<date>-<name>/`, listed in
  `docs/archive/README.md`.
- Commits only after the person's yes; the author and the other release
  mechanics are the person's to state.
