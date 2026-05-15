# Job layout — directory + filename protocol

> **This document is the sole source of truth for the on-disk shape of
> a molbuilder-driven SIESTA / PySCF run.**  Both the generators (which
> WRITE the layout — Build tab, `molbuilder fdf`, `molbuilder pyscf`)
> and the watcher (which READS it — Watch tab, `molbuilder watch parse
> / tail`) follow this contract.  Drift between the two is a bug; if a
> generator changes a filename or a parser changes a discovery rule,
> update this spec in the same commit.

The protocol exists so that:

1.  The Watch tab can be pointed at a **run directory** instead of a
    specific output file and figure out the right thing to load.
2.  A user staging a calculation across `coarse → medium → tight`
    runs (see the Watch-tab embedded "Staged relaxation workflow"
    guide) can keep all artefacts in one place without ad-hoc naming.
3.  Cluster-level scripts can `scp -r my-job/` between machines and
    everything still works because no path or filename is hard-coded
    outside the directory.

---

## The two rules

### Rule 1 — One job per directory

Every job lives in its own directory.  The directory may contain
**zero or more inputs** (one input per stage of a staged relaxation;
see § Multi-stage runs below) plus the engine's outputs and restart
files.  A single directory must not contain inputs for *different*
jobs (different molecules, different `SystemLabel`s) — that's two
directories.

This is mechanically enforced by SIESTA: the restart files
(`<basename>.XV`, `<basename>.DM`, `<basename>.CG`) are unprefixed
within the directory, so a second job's `SystemLabel` would clobber
them.  PySCF inherits the same convention via the inlined
`_MolwatchEmitter` and the geomeTRIC trajectory.

### Rule 2 — All files share one **basename**

Every file the generator produces and the watcher reads carries the
same basename:

- **SIESTA**: the basename **is** the FDF's `SystemLabel`.
- **PySCF**: the basename **is** the script's `job_name` field.

The basename must be a single token: letters, digits, hyphens,
underscores; no spaces, no dots, no slashes.  Generators reject
anything else with a clear error.

Across staged runs, the basename **stays identical** (only the
parameters change).  This is what lets SIESTA pick up
`<basename>.XV` and `<basename>.DM` from the previous stage on the
next invocation.

---

## File catalogue

For a job with basename `my-job`:

| File | Written by | Read by | Purpose |
|---|---|---|---|
| `my-job.fdf`            | molbuilder Build tab / `molbuilder fdf`   | SIESTA | input deck |
| `my-job.py`             | molbuilder Build tab / `molbuilder pyscf` | python  | input script (PySCF) |
| `my-job.run.sh`         | `molbuilder run my-job.fdf` / `molbuilder run my-job.py` | user's shell / SLURM / Cron | shell wrapper that activates the right conda env and executes the script -- see § "Run wrapper" below |
| `my-job.molwatch.log`   | both generators (initial preview) + the inlined PySCF emitter (live frames) + SIESTA preview-writer | Watch tab parser, `molbuilder watch parse`, `molbuilder watch tail` | **canonical trajectory source** — preferred by every reader |
| `my-job.out`            | the user's shell redirect (SIESTA) | Watch tab fallback parser | engine stdout — recommended redirect target |
| `my-job.log`            | the generated PySCF script | Watch tab fallback parser | textual SCF log (PySCF) |
| `my-job_geom_optim.xyz` | the generated PySCF script | geomeTRIC parser fallback | trajectory frames (PySCF) |
| `my-job.STRUCT_OUT`     | SIESTA | end-user / next stage | final relaxed coordinates |
| `my-job.ANI`            | SIESTA | external trajectory tools | per-step trajectory |
| `my-job.XV`             | SIESTA | next stage (with `MD.UseSaveXV`) | latest coordinates + velocities |
| `my-job.DM`             | SIESTA | next stage (with `DM.UseSaveDM`) | density matrix |
| `my-job.CG`             | SIESTA | next stage (with `MD.UseSaveCG`) | conjugate-gradient state |
| `my-job.chk`            | the generated PySCF script | next-stage continuation | SCF checkpoint |

The single `.molwatch.log` is **the** canonical trajectory source.  It
is written before the engine even starts (the initial-geometry
preview at step 0) so that pointing the Watch tab at the directory
works *immediately after* generating the input, before
`siesta` / `python my-job.py` has finished even one SCF cycle.

---

## How Watch resolves a directory

When the Watch tab loads a path that points at a directory, the
discovery chain runs in this order; first hit wins:

1. **`*.molwatch.log`**.  If multiple are present (staged run; see
   § Multi-stage runs), pick the most recently modified.
2. **`*.fdf`**.  Parse `SystemLabel`; look for
   `<system_label>.molwatch.log`, then `<system_label>.out`.
3. **`*.py`** containing the molbuilder marker comment.  Parse the
   `job_name` literal; look for `<job_name>.molwatch.log`, then
   `<job_name>.log`, then `<job_name>_geom_optim.xyz`.
4. **`run.out`** / **`siesta.log`** / **`*.out`**.  Try the SIESTA
   parser's content sniff; fall through if nothing matches.
5. **`*_geom_optim.xyz`**.  Try the PySCF parser.

If no candidate file is found, return a clear error listing every
filename the discovery chain tried so the user can see exactly what
was missing.

A non-directory path (a regular file) is loaded directly, as today —
the discovery chain only triggers when the path resolves to a
directory.  Existing scripts and tests that pass a file path keep
working unchanged.

---

## How Build / Modify enforces the layout

**Generators** (Build tab, `molbuilder fdf`, `molbuilder pyscf`):

- Validate the basename at the form / CLI boundary: reject anything
  that isn't `[A-Za-z0-9_-]+`.
- Write the initial `<basename>.molwatch.log` at file-emission time,
  before the engine runs.
- Emit a "Run with:" line in the FDF's verbose-comments block,
  e.g.

    ```fdf
    # Run with:
    #     mpirun -np 4 siesta < my-job.fdf > my-job.out
    ```

  The user's redirect target is the recommended one.  SIESTA can't
  enforce this — the user always controls the shell — but the Watch
  tab's discovery chain will still find the trajectory via
  `<basename>.molwatch.log` even if the user redirects elsewhere.

**The Modify tab "Send to Build" handoff** carries the basename
through `sessionStorage["builder-structure"]`; Build's restore-state
path picks it up unchanged.

---

## Multi-stage runs

For staged relaxation (the canonical workflow on the Watch tab's
"Staged relaxation workflow" panel), a single directory hosts the
three FDFs and three logs:

```
my-job/
├── my-job-stage1.fdf       ← coarse
├── my-job-stage2.fdf       ← medium
├── my-job-stage3.fdf       ← tight
├── my-job.molwatch.log     ← latest stage's frames; preferred trajectory source
├── my-job.out              ← latest stage's stdout (one redirect at a time)
├── my-job.XV               ← restart: from the latest stage
├── my-job.DM               ← restart: from the latest stage
├── my-job.CG               ← restart: from the latest stage
└── my-job.STRUCT_OUT       ← final relaxed coordinates after stage 3
```

The basename stays `my-job` across all three stages; only the FDF
filenames carry the `-stageN` suffix.  This means:

- The continuation flags (`MD.UseSaveXV`, `DM.UseSaveDM`,
  `MD.UseSaveCG`) work between stages.
- Watch tab discovery picks up the **latest** `.molwatch.log` as the
  primary view (since each stage rewrites the same file).

**Multi-stage trajectory merging.**  When the directory contains
**more than one** `*.molwatch.log` file, the Watch loader parses
all of them in mtime order (oldest first), concatenates the
trajectories into a single merged view, and tags each source as a
"stage" with metadata (start frame, end frame, source filename).
The energy and force plots draw dashed vertical lines at each stage
boundary with the source filename as a label.  Live polling pins to
the **newest** log file (the one currently being written by the
active SIESTA / PySCF run); older stages are static.

**Auto-suffixed log filenames per stage.**  When the Build tab's
Relaxation-stage selector is set to a non-Custom value, both the
SIESTA and PySCF generators emit a stage-suffixed log filename:

  * SIESTA: the preview write at FDF-emission time goes to
    `<basename>-stage<N>.molwatch.log` (the FDF's "Run with:" hint
    block also advertises the stage-suffixed FDF + stdout names).
  * PySCF: the inlined `MolwatchEmitter(...)` constructor receives
    `JOB + "-stage<N>.molwatch.log"` instead of the bare
    `JOB + ".molwatch.log"`.

The basename / `SystemLabel` / `JOB` itself stays unsuffixed across
stages so SIESTA's `.XV` / `.DM` / `.CG` restart files transfer
cleanly.  Only the *log* filename grows the suffix.  A run directory
ends up with the catalogue:

```
my-job/
├── my-job-stage1.fdf            ├── my-job-stage1.molwatch.log
├── my-job-stage2.fdf            ├── my-job-stage2.molwatch.log
├── my-job-stage3.fdf            ├── my-job-stage3.molwatch.log
├── my-job.XV / .DM / .CG        ├── my-job.STRUCT_OUT (final)
└── my-job-stage1.out / -stage2.out / -stage3.out (engine stdout)
```

Pointing the Watch tab at `my-job/` resolves to all three
`.molwatch.log` files in mtime order and renders one merged
trajectory with stage boundary markers.

---

## Project-tree organisation

A single job-layout-v1 directory sits at the bottom of a three-level
organisational hierarchy under the (gitignored) `projects/` root:

```
projects/
└── <project>/                  # e.g. "Au-thiol-junctions"
    └── <topic>/                # canonical vocabulary (see below)
        └── <structure>/        # ← the job-layout-v1 directory
            ├── <basename>.fdf  (or .py)
            ├── <basename>.run.sh
            ├── <basename>.molwatch.log
            └── ... (all other files share the same basename)
```

The hierarchy is **organisational only**.  The innermost `<structure>/`
is exactly the flat one-job-per-directory shape rules 1 and 2 of this
protocol describe — no subdirectories inside it; SIESTA's restart
files (`.XV` / `.DM` / `.CG`) and the rest of the catalogue live at
the same level next to the input deck.

Canonical topic vocabulary (six values, hard-coded in
`molbuilder.projects.CANONICAL_TOPICS`):

| Topic | Used for |
|---|---|
| `optimization` | Geometry relaxation |
| `frequency`    | Hessian / vibrational frequencies + RRHO thermo |
| `spectrum`     | Raman / IR (and later UV-Vis) at an optimised geom |
| `transport`    | NEGF / TBtrans device calculations |
| `single-point` | Energy at a fixed geometry |
| `scan`         | Potential-energy-surface scans |

Topic names not in this set are rejected with `InvalidName` at path-
construction time.  This is deliberate: an open vocabulary would
fragment the workflow tree across users and break the "compare the
same analysis across structures" intuition that motivated the
topic-first ordering.

Naming rules for each segment of the tree (`<project>`, `<topic>`,
`<structure>`):

* Must match `[A-Za-z0-9_-]+`.  Same character set rule 2 above
  imposes on the basename, for the same reason — SIESTA's
  basename-based file discovery would break otherwise.
* `<topic>` additionally must be one of the canonical six.

`molbuilder.projects` exposes:

* `validate_name(name, kind=...)`, `validate_topic(name)`
* `project_dir(p)`, `topic_dir(p, t)`, `structure_dir(p, t, s)`
* `ensure_structure_dir(p, t, s)` -- mkdir -p
* `list_projects()`, `list_topics(p)`, `list_structures(p, t)`
* `find_geom_candidates(project=...)` -- scan the tree for files
  matching `*_optimized.xyz` / `*.STRUCT_OUT` / `*.xyz` / `*.pdb`,
  sorted by mtime descending.  This is the "intelligence" backing
  the Build-tab "starting geometry" picker in the (future) web UI.

## Run wrapper (`<basename>.run.sh`)

`molbuilder run my-job.fdf` (or `.py`) emits a sibling
`<basename>.run.sh` that activates the routed conda env and executes
the tool.  Generated wrappers look like:

```bash
#!/usr/bin/env bash
#
# molbuilder run-wrapper -- SIESTA-MPI run on 4 ranks
# Script: my-job.fdf
# Target env: molbuilder-siesta
#
set -euo pipefail
cd "$(dirname "$0")"

exec conda run -n molbuilder-siesta --no-capture-output \
    mpirun -np 4 siesta my-job.fdf > my-job.out
```

Routing:

* `.fdf`  → `molbuilder-siesta` env, `mpirun -np N siesta ... > .out`
  (or single-process if `--np` is omitted or < 2).
* `.py`   → `molbuilder-pySCF` env, `python ...`.  No stdout
  redirect — the inlined `_MolwatchEmitter` writes its own
  `.molwatch.log` / `.log` files.

The wrapper is **plain bash**: a user can read it, edit it, paste
chunks into a SLURM script, or run it directly:

```
bash my-job.run.sh           # foreground
nohup ./my-job.run.sh &      # background, detached
```

molbuilder does NOT manage the resulting process.  Monitoring is via
the existing Watch tab pointed at the run directory; the watcher's
discovery chain (above) finds the `.molwatch.log` as soon as the
engine writes its first frame.

The wrapper is regenerated freshly each time `molbuilder run` runs
(it's per-invocation output, not state).  Edits between regenerations
are lost — keep custom flags in a sibling wrapper if you need them
preserved.

## What the protocol does NOT cover

- **Pseudopotential files** (`<Element>.psml`) sit alongside the FDF
  in the same directory; their names are dictated by the chemical
  element, not the basename.  They're shared across jobs (a Au
  pseudopotential is the same Au pseudopotential everywhere); the
  `--psml-lib` CLI flag copies them into the run directory at
  generate time, but the layout doesn't *require* this.
- **Output post-processing files** (Mulliken charges, BandLines
  output, PDOS files) follow SIESTA's own naming conventions
  (`<basename>.MullikenPop`, `<basename>.bands`, etc.).  They use
  the protocol's basename automatically because they're SIESTA's own
  output.
- **Pickle / cache files for analysis tools** the user runs after
  the job finishes are out of scope.

---

## Cross-engine reference

| Concept | SIESTA | PySCF | Watch tab |
|---|---|---|---|
| Job basename source | `SystemLabel` in FDF | `job_name` in script | discovered from input file |
| Input filename | `<basename>.fdf` (or `<basename>-stage1.fdf` etc.) | `<basename>.py` | recognised by extension |
| Initial-geometry preview | `<basename>.molwatch.log` (written by generator before engine starts) | same | loaded immediately |
| Live trajectory | `<basename>.molwatch.log` (SIESTA's preview is overwritten as the engine runs; in molbuilder-generated FDF, the engine appends frames via the parser-on-stdout path) | `<basename>.molwatch.log` (inlined `_MolwatchEmitter` appends per geomeTRIC step) | re-parses on mtime change |
| Restart inputs | `<basename>.XV` / `.DM` / `.CG` | `<basename>.chk` | n/a |
| Final geometry | `<basename>.STRUCT_OUT` | written by the script's `save_optimized_xyz` | n/a |

---

## Versioning

This spec is `job-layout v1`.  Future protocol changes (e.g. adding
sub-directories for per-stage isolated workspaces, or moving
restart files into a `.restart/` hidden folder) get a new version
number; the discovery chain in the Watch loader retains v1 fallbacks
until at least one minor release after v2 lands.
