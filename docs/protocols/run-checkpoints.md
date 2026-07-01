# Run-checkpoints — git-based working-dir state management

**Status**: design — pre-implementation. No code referenced here exists yet. Authoritative for the implementation that closes tasks #33 and the related sidebar work.

**Related design surfaces**:
- [`projects-sidebar.md`](projects-sidebar.md) — the sidebar architecture this design plugs into.
- [`script-execution.md`](script-execution.md) — the run-wrapper contract (the Phase 2/3 hooks land in `runwrap.py`).
- [`web-api.md`](web-api.md) — endpoint envelope shapes.
- [`workspace-contract.md`](workspace-contract.md) — the persistence model this design **does not** alter.

---

## 0. How to read this doc

| If you are … | Start here |
|---|---|
| Approving the design | § 1 (mission), § 2 (principles), § 11 (open questions) |
| Implementing the Python module | § 3 (architecture), § 4 (lifecycle), § 7 (module API), § 9 (.gitignore policy) |
| Implementing the HTTP routes | § 8 (HTTP contract) + [`web-api.md`](web-api.md) for envelope rules |
| Implementing the sidebar UI | § 6 (UI integration), § 6.4 (graph viewer), § 6.5 (sensor) |
| Wiring the wrapper hooks | § 4.2–4.4 (lifecycle phases) + [`script-execution.md`](script-execution.md) |
| Writing tests | § 12 (testing strategy) |
| Reviewing a PR against this surface | § 13 (anti-patterns) |

---

## 1. Mission

> When the user wants to save a state of a working directory — `.fdf`, `.out`, `.molwatch.log`, `.DM`, `.TSHS`, sidecars, user notes — that state must be **recoverable, diffable, and branchable** with a single explicit action.

The current model (rolling overwrites, manual file-shuffling) loses scientific work the moment a parameter sweep is attempted. The user already understands `git tag`, `git branch`, `git checkout` — the design re-uses that mental model rather than inventing a parallel snapshot vocabulary.

**Scope**:
- A working dir under `projects/` becomes a git repository **once the user explicitly initialises it**.
- The user explicitly commits, tags, branches, and restores. There is **no** automatic activity.
- Big binary state is archived **outside** the git object database but **keyed by** the git SHA.
- The Projects Sidebar surfaces the commit history, branches, tags, and binary archives as first-class navigation.
- The repository is **scoped to the single working directory** (where the `.fdf` / `.py` / `.run.sh` live). No parent or sibling directory is touched. This is the **lowest-directory** rule — it keeps each independent run (and each SLURM job) self-contained.

**Non-goals**:
- This is **not** a backup tool. We do not push to a remote unless the user configures one.
- This is **not** an audit trail. No wrapper hook commits behind the user's back. If the user runs the wrapper without checkpointing first, the prior state is the prior state — there is no auto-recovery.
- This is **not** a replacement for the workspace persistence contract ([`workspace-contract.md`](workspace-contract.md)). The workspace remains the sole source of truth for browser-side UI state; this design owns disk-side state.
- This does **not** integrate with the outer molbuilder git repo. The `.git` directories created here are nested git repos; the outer molbuilder repo's `.gitignore` already excludes `projects/*`.
- This is **single-user**. No multi-user / NFS / concurrent-access semantics. One person owns the working dir.

---

## 2. Principles

**P1. Git is the source of truth for everything textual.** `.fdf`, `.out`, `.molwatch.log`, sidecars (`.molstruct.json`, `.transport.json`), configs, READMEs. No parallel "history" mechanism exists for these.

**P2. Binaries are archived by SHA, not committed.** `.DM`, `.HSX`, `.TSHS`, `.TBT.AVTRANS_*` go to `.binsnapshots/<sha>/` inside the working dir. Git tracks the existence of the archive (the directory's `.gitkeep`), not the contents.

**P3. Every commit after the first is explicit and user-initiated.** The user chooses when to checkpoint; that choice is the only thing that creates a non-bootstrap commit. Running the wrapper without checkpointing first is a deliberate user choice — the prior state is the prior state, and there is no auto-recovery.

**P3a. One bootstrap exception.** The very first wrapper invocation in a virgin directory (no `.git/`) auto-initialises the repo AND auto-creates the first commit containing every existing file. This is the **only** time the wrapper writes to git. After that, the wrapper is silent on the git side. Rationale: without this, every new working dir would need a manual "init" before the user could even checkpoint — high friction for low cost. After this, the user explicitly chooses every subsequent commit.

**P4. The wrapper never auto-commits after the bootstrap.** No pre-run hook, no post-run hook, no background activity. The single git operation in the wrapper's lifetime is the bootstrap in P3a. This is what guarantees SLURM compatibility for **runs** — compute nodes execute the same SIESTA / PySCF command they would without git, plus optionally a `git status` line that records the starting state (informational, never fails the run).

**P4a. Users extend git activity through the USER-CUSTOM block.** Every generated script carries an empty `USER-CUSTOM` block ([`script-contract.md`](script-contract.md), [`script_emit.py::emit_user_custom_placeholder`](../../molbuilder/script_emit.py)). Users who want pre-run tagging, conditional branching, or post-run commits drop those commands into the USER-CUSTOM block. Molbuilder ships a **snippet library** of recommended fragments (§ 4.7) the user can paste in. The wrapper does not pre-fill the block with git commands — the user opts in by pasting.

**P5. The repository scope is the single working directory.** `git init` happens **only** in the lowest directory — the one containing `.fdf` / `.py` / `.run.sh`. Sibling directories and parent directories are not touched. Each independent run (and each SLURM job) is self-contained.

**P6. Tagging is for humans, branching is for experiments.** Tags (e.g. `stage3-converged`) are semantic milestones the user creates. Branches (e.g. `stage4-tzp`) carry experimental parameter sweeps. (How this composes with the staged-execution job-set — each stage dir is one of these per-working-dir repos, carry-forward vs git-branch as the two lineage axes — is in [`staged-execution.md`](staged-execution.md) § 6.2.)

**P7. The sidebar is the primary navigation UI; the CLI is the secondary.** The CLI exists for completeness and SSH workflows; the sidebar is where most users will operate.

**P8. Git is provided by molbuilder's conda envs, not the system.** Every env that ships a wrapper (`molbuilder-host`, `molbuilder-siesta`, `molbuilder-siesta-gpu`, `molbuilder-pySCF`, `molbuilder-MDtools`) declares `git` as a `conda_packages` entry. After `conda activate <env>`, the env's `bin/git` is at the front of PATH and takes precedence over any system git. Rationale: HPC sites have inconsistent git versions (some ship 1.8, some 2.40); reproducibility requires the same git for everyone; and the conda env is the only thing we control. The bootstrap preflight (`molbuilder envs bootstrap`) verifies `git --version` returns from a path under the env's prefix, not from `/usr/bin` or `/bin`.

**P9. Single user.** This design assumes one human owns the working directory. No concurrent-access reconciliation, no NFS contention story, no merge UI.

---

## 3. Architecture

### 3.1 Modules

| Layer | Module | Owns |
|---|---|---|
| **L1**: data model | `molbuilder/checkpoint.py` (new) | `Checkpoint`, `Branch`, `Tag` dataclasses; pure git-state representation. |
| **L2**: orchestration | `molbuilder/checkpoint.py::Repo` | Init (Path A), user-invoked checkpoint, binary archive write/read. Wraps `subprocess.run(["git", ...])`. |
| **L2**: CLI | `molbuilder/cli.py::cmd_snapshot_group` (new) | `molbuilder snapshot {init,checkpoint,list,tag,branch,diff,restore,prune,migrate-manifest}` |
| **L2**: wrapper bootstrap prologue | `molbuilder/runwrap.py` (small addition) | Emits the `if [ ! -d .git ]; then ...` block at the top of every generated `.run.sh`. Idempotent shell-side check. |
| **L2**: snippet library | `molbuilder/snippets/git/*.sh` (new) | User-pasteable git fragments (pre-run-checkpoint, post-run-checkpoint, tag-on-converged, branch-on-experiment). |
| **L3**: HTTP routes | `molbuilder/web/blueprints/checkpoint.py` (new) | Read-only endpoints + checkpoint/tag/branch/restore POST endpoints + GET snippet library. |
| **L3**: sidebar UI | `molbuilder/web/static/lib/projects/checkpoint.js` (new) | The graph viewer, sensor badge, action menus, vertical-split layout management. |
| **L3**: graph rendering library | `molbuilder/web/static/vendor/gitgraph/gitgraph.umd.min.js` (vendored) | `@gitgraph/js` (MIT). SVG git-graph rendering. Loaded via `<script>` tag like the existing CodeMirror vendor. |
| **L3**: form UI | `molbuilder/web/static/lib/build/snippet-menu.js` (new) | "Insert snippet ▸ Git checkpoint …" menu integrated with the USER-CUSTOM CodeMirror editor. |

**The wrapper prologue is the ONE place runwrap.py touches git**, and it does so at script-generation time (emits the bash test) — not at Python execution time. `runwrap.py` itself never invokes git via `subprocess`.

### 3.2 Surface separation

Same separation principle as [`projects-sidebar.md`](projects-sidebar.md) § 3: the Python module is content-agnostic git plumbing; the JS owns the visualisation; the HTTP layer is a thin contract that does not embed view logic.

### 3.3 Where git binaries come from

**Every** molbuilder env declares `git` as a `conda_packages` entry. Concretely, `envs/recipes.py` adds `git` to:

- `molbuilder-host` (CLI + Flask + sidebar UI — Path A init and the HTTP routes use this)
- `molbuilder-siesta` (precompiled CPU SIESTA — Path B bootstrap prologue uses this)
- `molbuilder-siesta-gpu` (source-built GPU SIESTA — same)
- `molbuilder-pySCF` (PySCF — same)
- `molbuilder-MDtools` (AmberTools workflows that might checkpoint between tleap calls)
- `molbuilder-tests` (Playwright env — tests need git to set up fixture repos)

This guarantees that after `conda activate <any-env>`, `git` is on PATH at the env's prefix. Compute nodes do not need a system git, do not need network access, and do not need any site-specific module-load step. The bootstrap preflight (`molbuilder envs bootstrap`) verifies:

1. `git --version` returns successfully.
2. `which git` resolves under one of molbuilder's env prefixes, NOT `/usr/bin/git` or `/bin/git`. If the system git is shadowing the env git (PATH ordering bug), the bootstrap fails loudly with the fix instruction (`hash -r` or env-recreation).
3. The version is ≥ 2.20 (when `git restore` was introduced — load-bearing for Phase 5).

Conda-forge's `git` package is ~30 MB compressed; the cost of adding it to every env is bounded.

---

## 4. Lifecycle

All phases except Phase 0 are user-initiated. The wrapper is **never** in the call path of any phase.

### 4.1 Phase 0 — Bootstrap (once per host)

**Trigger**: `molbuilder envs bootstrap`.
**Action**:
1. Check `git --version` ≥ 2.20.
2. If absent: emit one-line error `"molbuilder requires git ≥ 2.20; install via your distro's package manager."` and exit non-zero.

### 4.2 Phase 1 — Working-dir init

Two equivalent paths into the same end state, both writing the same files in the same order.

#### Path A — Explicit (CLI / sidebar button)

**Trigger**: `molbuilder snapshot init` (CLI) or the sidebar "Initialise run history" button (UI). Used when the user wants the repo set up **before** any run — e.g. they want to inspect a fresh dir under git first.
**Action**: equivalent to:
```
git init -q
git config user.email "molbuilder@<hostname>"
git config user.name  "molbuilder"
git config commit.gpgsign false
write .gitignore         (§ 9.1)
write .binsnapshots/.gitkeep
git add .                                         # everything not in .gitignore
git commit -q -m "molbuilder: initial state of <dir>"
```

#### Path B — Bootstrap from the wrapper (first run only)

**Trigger**: the generated `.run.sh` is executed in a dir without `.git/`.
**Action**: identical to Path A above — the wrapper emits the same sequence as a short shell prologue at the top of the script (before activate / before SIESTA), wrapped in `if [ ! -d .git ]; then ... fi`. On subsequent runs the prologue is a single `[ ! -d .git ]` check that short-circuits to false; the wrapper does nothing git-related.

**SLURM note**: the bootstrap prologue runs on the compute node. This is fine because (i) HPC clusters universally ship `git` on PATH; (ii) the bootstrap touches only files inside the working dir — no network, no `~/.gitconfig` writes (we use `git config` without `--global`); (iii) on subsequent runs the prologue is a single test and a noop.

#### Common refusal rule

If the directory has subdirectories that are themselves working dirs (contain `.fdf` / `.py` / `.run.sh`), init refuses with one message naming them — per P5, each lowest-directory is its own repo, and a higher-level init would violate the scoping rule. In the wrapper-bootstrap path (B) the refusal becomes a clear shell-side error that aborts the run before SIESTA launches.

#### Optional `git status` echo on every run

The wrapper prologue, AFTER the bootstrap check, optionally emits a one-line `git status --short | wc -l` capture into the run log:
```
echo "# git status at run start: $(git status --short | wc -l) uncommitted files" >> "${_runlog}"
```
This is informational — records what state the run started from, never alters behaviour, never fails the run. Off by default; enabled via `cfg.script_generation.echo_git_status = true` in `molbuilder.json`.

### 4.3 Phase 2 — Checkpoint (user-driven)

**Trigger**: `molbuilder snapshot checkpoint [-m MESSAGE]` (CLI) or the sidebar "Checkpoint now" button (UI).
**Action**:
1. `git add .` — picks up everything not in `.gitignore`.
2. If nothing to commit: emit `"working tree clean; nothing to checkpoint"` and exit successfully (no error).
3. Otherwise: `git commit -m "<MESSAGE>"` (if message provided) or `git commit -m "checkpoint <ISO_TS>"` (default).
4. `NEW_SHA = git rev-parse HEAD`.
5. Identify big binaries present (`.DM`, `.HSX`, `.TSHS`, `.TBT.AVTRANS_*`).
6. `mkdir -p .binsnapshots/$NEW_SHA && cp -a <binaries> .binsnapshots/$NEW_SHA/`.
7. Write `.binsnapshots/$NEW_SHA/MANIFEST` per § 10.2 (canonical 3-column format `<sha256> <bytes> <name>`, sorted by filename, atomic rename).

That is the entire commit lifecycle. There is no Phase 3.

### 4.4 Phase 3 — Semantic tagging (user-driven)

`molbuilder snapshot tag <label> [--message TEXT]` or sidebar context menu. Equivalent to `git tag -a <label> HEAD -m TEXT`. Tags are always annotated (carry a message) so the audit trail is human-readable.

### 4.5 Phase 4 — Experimental branching (user-driven)

`molbuilder snapshot branch <name>` or sidebar UI. Equivalent to `git checkout -b <name>`. The user's subsequent checkpoints land on that branch.

### 4.6 Phase 5 — Inspection / restore

| User action | Underlying git op | Binary handling |
|---|---|---|
| List | `git log --graph --oneline --decorate --all` | — |
| Diff text | `git diff <a>..<b> -- '*.fdf' '*.out' '*.molwatch.log'` | — |
| Restore to a checkpoint | **verify** the ref's binary archive, then `git restore --source=<ref> .`, then `cp .binsnapshots/<sha>/* .` | verify-first, then text, then binaries |
| Prune unused binaries | identify SHAs unreferenced by any tag/branch/HEAD; `rm -rf .binsnapshots/<sha>` | — |

**Restore is ATOMIC and safe by contract (§ 10.3 — data-safety critical):**

1. **Refuses on a dirty *text* tree** — `git status --porcelain` non-empty (P3: the user decides whether to discard or checkpoint first; the system does not pick).
2. **Refuses on dirty *binaries*** — big binaries are gitignored, so `git status` can't see them; restore separately compares each working big-binary's sha256 to HEAD's archived copy and **refuses** if any differ (they would be silently overwritten). Message: *"uncommitted binary changes would be overwritten by restore: `<names>`. Checkpoint or move them aside first…"* Skipped for a text-only restore (`--no-binaries`).
3. **Verifies the archive BEFORE mutating anything** — the ref's archived binaries are checked (existence + size + sha256) *before* `git restore` runs, so a corrupt/incomplete archive aborts the WHOLE restore (text stays put too). Never a half-restored tree. Message: *"…integrity check failed for `<name>`… refusing to restore."*
4. **Warns loudly on a missing archive** — if the ref has NO archive but the project clearly uses big binaries (working binaries present, or other checkpoints have archives), restore succeeds text-only and prints: *"WARNING: checkpoint `<ref>` has NO binary archive, but this project uses big binaries — if `<ref>` had .DM/.HSX/.TSHS they were NOT restored (the archive may be incomplete, e.g. a checkpoint interrupted between commit and archive). Verify the result; re-checkpoint to heal."* This is the honest bound: because gitignored binaries aren't recorded in the commit, a lost archive can't be *proven*, only flagged — never silent.

**Save-side fidelity (§ 4.3):** `_archive_binaries` hashes the source, copies, re-hashes the archived copy, and requires they match — a silent copy corruption (disk error) fails the checkpoint loudly rather than recording the corrupt bytes' own sha as truth.

**Known limitation (documented, not a silent trap):** a **binary-only change** (a big binary edited with no accompanying text change) is treated as a clean tree by `checkpoint()` and is not captured as a new checkpoint while HEAD already has an archive. Change a text file alongside, or move the binary aside, if you must snapshot a binary-only edit. (Restore's dirty-binary refusal above still protects such an edit from being overwritten.)

### 4.7 USER-CUSTOM snippet library

Per P4a, users extend git activity by pasting snippets into the `USER-CUSTOM` block of their generated script. Molbuilder ships a small library of fragments under `molbuilder/snippets/git/` and surfaces them in the form UI as **"Insert snippet ▸ Git checkpoint pre-run / post-run / tag / branch"** menu items.

Initial set:

| Snippet name | What it does | When to paste it |
|---|---|---|
| `pre-run-checkpoint.sh` | `git add . && git commit -m "auto: before <stage>-<run-id> on $(date -I)"` — captures the pre-run state at SCRIPT runtime | When the user wants every run to start from a clean checkpoint without remembering to click "Checkpoint now." Trade-off: noisy commit log; commits land even if the user didn't intend to. |
| `post-run-checkpoint.sh` | `git add . && git commit -m "auto: after <stage>-<run-id>, exit=$?, Etot=$(grep Total $_out_file | tail -1)"` | When the user wants the post-run state automatically recorded with the SIESTA result inline in the commit message. |
| `tag-on-converged.sh` | parses `.out`; if `>> End of run` present AND constrained max-F < tolerance, `git tag -a converged-$STAGE -m "..."` | For users running the same stage repeatedly and wanting only converged runs to leave a permanent tag. |
| `branch-on-experiment.sh` | parses `cfg.stage_strategy`; if "experiment" prefix, switches to `experiment-<label>` branch before running | For deliberate parameter sweeps where each variant should land on its own branch. |

Snippets are user-editable templates with `{{ placeholder }}` substitution (resolved by the same templating the wrapper uses for `$STAGE_NAME`, `$RUN_ID`, etc.). The Insert-snippet menu uses the existing CodeMirror surface; the user sees the rendered shell text and can edit before saving.

**Snippets live in the USER-CUSTOM block, which is preserved verbatim across regenerations** ([`script_emit.py::merge_user_custom_from_target`](../../molbuilder/script_emit.py)). Re-running `molbuilder fdf` does not clobber the user's pasted git commands.

**Snippets are user-authored content, not part of the wrapper contract.** Molbuilder does not test snippet *behaviour*; it tests only that the templating + placeholder substitution + USER-CUSTOM round-trip is correct. If a user's snippet is buggy and aborts the run, that's the user's bug to fix.

---

## 5. Data model

### 5.1 In Python

```
@dataclass
class Checkpoint:
    sha:         str         # 40-char git SHA
    short_sha:   str         # 7-char
    summary:     str         # one-line commit message
    body:        str         # rest of commit message
    author_at:   datetime    # ISO timestamp
    parents:     List[str]   # SHAs of parent commits
    refs:        List[str]   # tags + branches pointing here
    has_archive: bool        # True iff .binsnapshots/<sha>/ exists + non-empty
    archive_bytes: Optional[int]
    files_added:    int
    files_modified: int
    files_deleted:  int

@dataclass
class Branch:
    name:    str
    head:    str             # SHA the branch tip points at
    current: bool            # is HEAD on this branch?
    
@dataclass
class Tag:
    name:    str
    sha:     str             # SHA the tag points at (after deref of annotated tag)
    message: Optional[str]   # tag message for annotated tags
    is_pre_run: bool         # name starts with "pre-"
    is_semantic: bool        # user-supplied (= not auto-created)
```

### 5.2 Repo state snapshot

```
@dataclass
class RepoState:
    path:          str       # absolute path to working dir
    initialized:   bool      # has .git?
    head:          Optional[str]   # SHA at HEAD; None if no commits yet
    current_branch: Optional[str]  # None if detached
    dirty:         bool      # uncommitted changes since HEAD
    untracked:     int       # count of untracked files
    archive_total_bytes: int
    archive_used_shas: List[str]   # SHAs that still have an archive
```

This is the structure the sidebar sensor reads on **directory-enter
and manual refresh** (§ 6.2) — there is no background polling.

`archive_total_bytes` / `archive_used_shas` are **not** populated by
the refresh-path `state()` call (they default to `0` / `[]`); computing
them requires a walk of `.binsnapshots/`, which is reserved for the
list/detail surfaces (§ 6.3, § 6.6) that actually display archive
size. Keeping `state()` cheap matters because it now runs on every
directory-enter, not just a timer.

---

## 6. UI integration — Projects Sidebar

### 6.1 Where it lives

The Projects Sidebar (see [`projects-sidebar.md`](projects-sidebar.md)) **splits vertically** the moment the selected directory is a **run directory that contains a `.git/`**. Two panes inside the same sidebar column:

```
┌─────────────────────────┐
│ Projects file tree      │   ← top pane (existing surface, unchanged)
│ ├ BDT                   │
│ │ ├ optimization        │
│ │ │ ├ TJ-BDT-Au111 ●    │   ← selected; has .git
│ │ │ └ BDT-only          │
│                         │
├─────────────────────────┤   ← draggable horizontal divider
│ Run history             │   ← bottom pane (new)
│ [List ▾] [Graph ▴]      │
│                         │
│   ● d3f1a92 HEAD, main  │
│   │                     │
│   ● 8b22e07 stage3-conv │
│   │                     │
│   ● 2c19fa1 initial     │
└─────────────────────────┘
```

The split appears when **both** hold:
- The selected `current_dir` is a **run directory** — i.e. it sits at
  **projects rel-depth 3**, in the canonical layout
  `projects/PROJECT_NAME/CATEGORY/RUNNING_DIR_NAME`, AND
- That directory contains `.git/` (= has been initialised; confirmed
  with a single `git rev-parse --is-inside-work-tree`).

**Why rel-depth 3 is the only place a viewer appears.** The projects
tree is a fixed three-level layout (mirrors the sidebar depth model in
[`projects-sidebar.md`](projects-sidebar.md) § 4.1 + `list.js`):

| Path | rel-depth | Origin |
|---|---|---|
| `projects/` | 0 | resolved root (§ 4.2 of projects-sidebar) |
| `projects/PROJECT_NAME/` | 1 | created by the sidebar **New project** button |
| `projects/PROJECT/CATEGORY/` | 2 | auto-scaffolded canonical topic (`molbuilder.projects.CANONICAL_TOPICS`) |
| `projects/PROJECT/CATEGORY/RUNNING_DIR/` | **3** | the run dir — where `.fdf` / `.py` / `.run.sh` + `.git/` live |

A checkpoint repo is, by the lowest-directory rule (§ 1), always the
run dir. So the viewer is gated on rel-depth 3 explicitly: at depth 0,
1, or 2 (and at any depth ≠ 3, even if a stray `.git/` exists there),
the bottom pane stays blank. The flat-storage topics `structure` /
`pseudopotential` hold files, not run dirs, so the `.git/` check
already excludes them; the depth-3 gate is the structural backstop.

When the user selects anything that is not a depth-3 run dir with
`.git/` — a file, a shallower dir, a depth-3 dir not yet initialised —
the bottom pane collapses and the file tree fills the column. Sidebar
sensor pill (§ 6.2) lives in the bottom pane's header even when
collapsed.

**Divider behaviour**: a 4-px horizontal drag handle between the two panes mirrors the existing right-edge resize handle on `.projects-sidebar` ([`projects-sidebar.md`](projects-sidebar.md) § "Drag-resize handle"). User-resized height persists in `sessionStorage` (per [`workspace-contract.md`](workspace-contract.md) § 4.1, key `ws.ui.checkpoint.height_px`). Default split: top pane gets 60 % of the column height; bottom pane gets 40 %. Double-click the handle to reset to default.

**Hidden state**: a small chevron in the bottom-pane header collapses just the bottom pane (file tree fills column) without changing the selection. Collapse state persists per session, also under `ws.ui.checkpoint.collapsed`.

### 6.2 Sensor (always-visible badge)

A small status pill in the sidebar header for the current project directory:

| Visual | Meaning |
|---|---|
| 🟢 `clean · stage3-converged` | Repo present, working tree clean, last tag shown. |
| 🟡 `N changes` | Uncommitted changes since HEAD. Tooltip: "auto-committed on next run." |
| 🔵 `running` | A wrapper was executing in this dir (file lock present) **as of the last refresh** — see § 6.5. |
| ⚪ `no checkpoints` | Directory has no `.git/` — not yet a tracked working dir. |
| 🔴 `error: <one-line>` | Git command failed; pill is clickable → shows last error. |

**Refresh model — explicit, never polled.** The sensor reads
`/api/checkpoint/state` (cheap: `git status --porcelain` +
`git rev-parse HEAD`, no archive walk) on exactly two triggers:

1. **Directory-enter** — when the cursor moves to a depth-3 run dir
   with `.git/` (the `onChange` from the projects-sidebar cursor
   mutator, [`projects-sidebar.md`](projects-sidebar.md) § 4.1). One
   fetch per enter. Leaving the dir tears the viewer down.
2. **Manual Refresh** — a Refresh control (↻) in the bottom-pane
   header re-fetches state + the history list on demand. The user
   hits it after any operation that changes repo state (checkpoint,
   tag, branch, restore, prune) and after a wrapper run completes.

There is **no `setInterval`**, no visibility-driven resume/suspend,
and no background timer. This is a deliberate departure from the
earlier polling design (resolved 2026-06-26): polling charged every
open sidebar a `git status` every few seconds for state that only
changes on explicit user action; explicit refresh removes that
standing browser + subprocess cost entirely. The cost is that
state which changes *outside* the UI (a wrapper starting/finishing in
the background) is not reflected until the next enter or manual
refresh — accepted, because checkpoint capture is already
user-explicit (§ 6.5).

### 6.3 Run-history list view (default)

A scrollable list of checkpoints, most recent first:

```
○ d3f1a92  stage3-run1: converged, Etot=-1743290.22 eV         15:42  HEAD, main
│
○ 8b22e07  stage3-run0: converged, Etot=-1743290.22 eV         13:11  stage3-converged
│
○ 2c19fa1  molbuilder: initial state of TJ-BDT-Au111            09:14
```

Per row:
- Click → expand inline (files changed, archive size, full commit message).
- Right-click context menu → **Tag here**, **Branch here**, **Restore to this checkpoint**, **Diff vs. HEAD**, **Diff vs. working tree**.
- Tag chips (`stage3-converged`) are clickable: filter the list to commits with tags only.
- Branch chip (`main`) is clickable: switches to that branch's view.

### 6.4 Graph viewer (default view)

The bottom pane defaults to the graph view. A `[List ▾] [Graph ▴]` toggle in the header lets the user fall back to the list view (§ 6.3) if they prefer textual.

**Library: `@gitgraph/js`** (MIT, [nicoespeon/gitgraph.js](https://github.com/nicoespeon/gitgraph.js)). Vendored to `web/static/vendor/gitgraph/gitgraph.umd.min.js` (~30 KB), loaded via `<script>` tag on first mount — same pattern as the existing CodeMirror vendor under `web/static/vendor/codemirror/`. No bundler step.

**Why this library**:
- Purpose-built for git-graph rendering. API is `gitgraph.branch(name).commit({subject, hash, onClick})`.
- Built-in support for branches as parallel lanes, merge arcs, tag rendering, custom dot styling, click handlers.
- SVG output, themeable via custom templates → integrates with the sidebar's CSS-variable palette ([`projects-sidebar.md`](projects-sidebar.md) § 10).
- 30 KB minified — small enough to vendor without thinking; small enough that lazy-loading is unnecessary (it's always mounted once the pane is visible).

**Rendered shape** (illustrative; the real output is SVG):

```
● d3f1a92  stage3-run1                                        [HEAD, main]
│
│ ● e441b2c  stage4-tzp-run0: converged, Etot=-1743311.05    [stage4-tzp]
│ │
│/
●  8b22e07  stage3-run0                                       [stage3-converged]
│
●  2c19fa1  initial state
```

**Interactions**:
- Vertical axis is time (newest at top, matching git log default).
- Each commit is an SVG dot in its branch's lane; merge points draw an arc.
- Tags render as labelled chips next to their commit.
- HEAD is highlighted via a custom dot stroke colour from the existing sidebar palette.
- Left-click a commit → selects it; the detail panel (§ 6.6) appears below or to the right of the graph (whichever fits the current sidebar width).
- Right-click a commit → context menu (same actions as list view: Tag here, Branch here, Restore to here, Diff vs. HEAD, Diff vs. working tree).
- Tag chips are clickable: same as right-click → Restore to this tag.

**Out of scope**:
- Drag-and-drop operations (e.g. drag commit onto another to rebase). Too easy to fat-finger a destructive `git reset`. All operations are explicit menu items.
- Lazy-loading older commits. Default `limit=50` from the HTTP route covers most working dirs; if a user accumulates more, they can `git log` from the CLI.

**Style binding**: `@gitgraph/js`'s `templateExtend` API takes a base template and overrides colours / fonts / spacing. We pass:
```js
const template = templateExtend(TemplateName.Metro, {
    colors:  [getCssVar('--projects-sidebar-accent'),
              getCssVar('--projects-sidebar-accent-2'),
              ...],
    branch:  { spacing: 24, lineWidth: 2 },
    commit:  { spacing: 36, dot: { size: 10 } },
});
```
so a future palette change in `projects-sidebar.css` propagates without code change.

### 6.5 Live activity overlay

When a wrapper was running in this dir **as of the last refresh**
(detected from the existing run-lock or PID file — not from any git
state, since the wrapper does not touch git), an extra row at the top
of the list:

```
⟳  wrapper running (as of last refresh)      17:03      [Refresh ↻]
```

Because there is no polling (§ 6.2), this row reflects the lock state
captured at the last directory-enter or manual refresh — it does
**not** live-update or auto-disappear when the wrapper exits. The user
hits Refresh to re-read the lock and clear the row once the run is
done. This is acceptable precisely because the overlay is advisory:
it **does not** create a commit, and the user must explicitly
checkpoint if they want the post-run state captured.

### 6.6 Detail panel (right of the list/graph)

Selecting a commit reveals:
- Full commit message + body.
- File list with the existing **diff-vs-prev** action button per file (routes to the standard diff viewer per [`projects-sidebar.md`](projects-sidebar.md) § preview modal — which already uses CodeMirror).
- Archive presence: "binaries archived (43 MB)" or "no binaries archived" with a **Restore archive** button when present.
- Tag/branch chips, click-to-jump.

### 6.7 Empty state

When the directory has no `.git/`:

```
No checkpoint history yet.

[ Initialise run history ]   ⓘ
```

Click → `molbuilder snapshot init` against this directory. Tooltip explains "this tracks your fdf, .out, and sidecars so every parameter change is recoverable."

---

## 7. Python module API

```python
# molbuilder/checkpoint.py

class Repo:
    def __init__(self, path: str) -> None: ...
    @property
    def initialized(self) -> bool: ...
    def init(self) -> None:
        """Phase 1.  Initialise the working dir as a git repo.  Refuses if
        the dir contains nested working dirs (per P5)."""
    def state(self) -> RepoState:
        """Cheap (= sensor-polling) snapshot of current repo state."""
    def checkpoint(self, message: Optional[str] = None) -> Optional[Checkpoint]:
        """Phase 2 — user-invoked.  ``git add .`` + commit + archive new
        big binaries by SHA.  Returns the new Checkpoint, or ``None`` if
        nothing was changed since HEAD."""
    def list_checkpoints(self, *, since: Optional[str] = None,
                                 limit: int = 50) -> List[Checkpoint]: ...
    def list_branches(self) -> List[Branch]: ...
    def list_tags(self) -> List[Tag]: ...
    def diff(self, ref_a: str, ref_b: str,
                  pathspec: List[str] = None) -> str:
        """Unified-diff text.  pathspec defaults to text-only globs."""
    def tag(self, label: str, *, message: str, at: str = "HEAD") -> Tag:
        """Phase 3 — user-driven semantic tag.  Always annotated."""
    def branch(self, name: str, *, at: str = "HEAD",
                                   checkout: bool = True) -> Branch:
        """Phase 4 — user-driven branch."""
    def restore(self, ref: str, *, include_binaries: bool = True) -> None:
        """Phase 5 — git restore + binary archive copy.  RAISES if working
        tree is dirty (loud refusal; user must checkpoint or discard
        first)."""
    def prune_archives(self, *, keep_refs_only: bool = True) -> List[str]:
        """Remove .binsnapshots/<sha>/ for SHAs unreferenced by any
        tag/branch/HEAD.  Returns list of removed SHAs.  Dry-run when
        ``keep_refs_only=False``."""

class CheckpointError(Exception): pass
class GitNotInstalledError(CheckpointError): pass
class DirtyWorkingTreeError(CheckpointError): pass
class NoSuchRefError(CheckpointError): pass
class NestedRepoRefusedError(CheckpointError): pass
```

Each method delegates to `subprocess.run(["git", ...], cwd=self.path)`. No shell strings; argv lists throughout. Errors propagate verbatim from git.

**Not in the API**: `pre_run`, `post_run`, or any other wrapper-hook method. The wrapper is git-agnostic per P4.

---

## 8. HTTP contract

All routes return the standard envelope per [`web-api.md`](web-api.md) § 1.6.

### 8.1 Read routes

| Verb / Path | Body | Response (success) |
|---|---|---|
| `GET /api/checkpoint/state?path=PATH` | — | `{"state": RepoState}` |
| `GET /api/checkpoint/list?path=PATH&limit=50` | — | `{"checkpoints": [Checkpoint, ...], "branches": [...], "tags": [...]}` |
| `GET /api/checkpoint/diff?path=PATH&a=REF&b=REF&pathspec=*.fdf,*.out` | — | `{"diff": "<unified diff text>"}` |

### 8.2 Write routes (require explicit user action in UI)

| Verb / Path | Body | Notes |
|---|---|---|
| `POST /api/checkpoint/init` | `{"path": PATH}` | Phase 1. Refuses with `ok:false` when nested working dirs are present. |
| `POST /api/checkpoint/commit` | `{"path", "message"}` | Phase 2 — user-clicked "Checkpoint now". Returns `{"checkpoint": Checkpoint}` on success, or `{"ok": true, "checkpoint": null, "note": "nothing to checkpoint"}` when the tree is clean. |
| `POST /api/checkpoint/tag` | `{"path", "label", "message", "at"}` | Phase 3. Annotated tag (message required). |
| `POST /api/checkpoint/branch` | `{"path", "name", "at", "checkout"}` | Phase 4. |
| `POST /api/checkpoint/restore` | `{"path", "ref", "include_binaries"}` | Phase 5. Refuses on dirty working tree (HTTP 200 + `ok:false` per § 1.6 bucket B — this is a scientific-advisory case, not a protocol error). |
| `POST /api/checkpoint/prune` | `{"path", "dry_run"}` | Phase 5. |

### 8.3 Scientific advisories surfaced via this surface

- "Working tree is dirty; restore would clobber your changes." → HTTP 200, `ok:false`, errors_only carries the rule. UI shows a "Commit or discard first" prompt.
- "Restore target has no binary archive; the .DM/.TSHS for this checkpoint were pruned." → HTTP 200, `ok:false`. UI offers "restore text only" override.

---

## 9. .gitignore policy

### 9.1 Default `.gitignore` content (written at Phase 1)

```
# molbuilder run-checkpoints contract: this dir is auto-managed by git.
# Large binary state is archived separately in .binsnapshots/<sha>/.
*.DM
*.HSX
*.TSHS
*.TBT.AVTRANS_*
*.TBT.CC
*.TBT.DOS
*.ion.nc
*.ion.xml
.binsnapshots/
fdf.*.log              # SIESTA's rotating fdf-parse log; noisy + redundant with .out
WORK_*                 # SIESTA scratch
INPUT_TMP.*
```

### 9.2 Why each entry

| Pattern | Why ignored |
|---|---|
| `*.DM`, `*.HSX`, `*.TSHS` | Big binaries; archived by SHA in `.binsnapshots/`. |
| `*.TBT.AVTRANS_*`, `*.TBT.CC`, `*.TBT.DOS` | Transport output; can be regenerated from `.TSHS`. |
| `*.ion.nc`, `*.ion.xml` | Pseudopotential CACHE files; deterministic from `.psml`. |
| `fdf.*.log` | SIESTA's parsing log — verbose and rotating. |
| `WORK_*`, `INPUT_TMP.*` | SIESTA scratch files; meaningless mid-run captures. |

### 9.3 Files EXPLICITLY tracked (none are in `.gitignore`)

`*.fdf`, `*.psml`, `*.out`, `*.molwatch.log`, `*.molstruct.json`, `*.transport.json`, `*.runtime_info.json`, `*.parse.log`, `*.CG`, `*.XV`, `*.EIG`, `*.FA`, `*.FORCE_STRESS`, `*.bib`, `*.run.sh`, `*.md`.

Note `.XV` and `.CG` are tracked: they're small (< 100 KB on this 444-atom system) and load-bearing for warm-restart inspection.

---

## 10. Binary archive layout (canonical, locked 2026-06-25)

**One format only.** No alternative formats are accepted, on read or on write. If on-disk data does not match this spec, the parser raises `CheckpointError` with a clear message — no silent skip, no fallback. Legacy on-disk archives in a different format are migrated explicitly via `molbuilder snapshot migrate-manifest <ref>` (§ 10.4); they are not transparently accepted by the canonical parser.

The lockdown reason: any "accept both formats" code path is an unbounded design that drifts the moment a third variant shows up. Spec the format; code matches the spec; legacy is migrated, not forked.

### 10.1 Directory layout

```
<working_dir>/
├── siesta-foo.DM                              ← live, gitignored
├── siesta-foo.TSHS                            ← live, gitignored
└── .binsnapshots/
    ├── .gitkeep                               ← keeps the dir tracked
    ├── 8b22e07c…<40-char SHA>…/               ← one per checkpoint
    │   ├── MANIFEST
    │   ├── siesta-foo.DM
    │   └── siesta-foo.TSHS
    └── d3f1a92e…/
        ├── MANIFEST
        └── …
```

Rules:
- **SHA directory name MUST be the full 40-character lowercase hexadecimal commit SHA.** No short SHA. No alternative tag-name or branch-name aliases.
- Each SHA directory MUST contain exactly: a `MANIFEST` file (format § 10.2) **plus** one file per archived binary. No other files; no nested directories.
- The MANIFEST file MUST NOT list itself.

### 10.2 MANIFEST format

```
<sha256_hex>  <size_bytes>  <filename>\n
```

Per line:
- `sha256_hex`: exactly 64 lowercase hex characters.
- Two ASCII spaces (`\x20\x20`).
- `size_bytes`: decimal non-negative integer, no leading zeros, no thousands separator, no unit suffix.
- Two ASCII spaces.
- `filename`: bare basename relative to the SHA directory (no `/`, no `..`, no leading dot-prefix outside the canonical layout). ASCII-only. No tabs, no embedded whitespace.
- One `\n` line terminator (no CRLF). File MUST end with a final `\n`.

Whole-file rules:
- One entry per archived file. No comments. No blank lines. No header line.
- Order is alphabetical by filename (deterministic — diff-friendly across machines).
- UTF-8 BOM rejected.

A MANIFEST that fails any of these rules is malformed. The parser raises `CheckpointError(f"malformed MANIFEST in {archive_path}: <specific reason>")`. No silent skip, no field-count fallback.

### 10.3 Read + write contract

- **Write** (called from `Repo.checkpoint()` and `Repo.init()`): always emits the format above, generated from a `sha256sum` of each file's full bytes + `Path.stat().st_size`. Atomic write: write to `MANIFEST.tmp`, rename to `MANIFEST` only after the body is fully flushed.
- **Read** (called from `Repo.restore()` and `Repo.list_checkpoints()`): single canonical parser. The 2-column `sha256sum` default output is **not** accepted here — see § 10.4 for the migration path.
- **Verify on restore**: each archived file's sha256 is recomputed and matched against the MANIFEST's value before it overwrites a working-tree file. Any mismatch raises and aborts the restore before any byte hits the working tree.

### 10.4 Legacy migration

Some `.binsnapshots/<sha>/MANIFEST` files were hand-rolled with bash `sha256sum > MANIFEST`, producing a 2-column format (`<sha256> <filename>`). These are **legacy** and not supported by the canonical parser. They are migrated to the canonical 3-column format via a one-shot CLI:

```
molbuilder snapshot migrate-manifest <ref>
```

Behaviour:
1. Resolves `<ref>` to a 40-char commit SHA via `git rev-parse <ref>^{commit}`.
2. Reads the existing `MANIFEST` in `.binsnapshots/<sha>/`. If it already parses as canonical → exit 0 with "already canonical."
3. If it's the 2-column legacy form: parses sha256 + filename, then `stat`s each file for the bytes column, AND **re-hashes each file's content and verifies against the recorded sha256**. Any mismatch aborts with the bad file named.
4. Writes the canonical 3-column MANIFEST atomically (`MANIFEST.tmp` → rename).
5. Any other format → error, no auto-fix.

The migration touches only the MANIFEST file; the archived binaries themselves are never modified.

### 10.5 Test coverage required

The following tests must exist alongside the format (gate the lockdown):
- Canonical write produces a MANIFEST that round-trips through the canonical parser.
- Canonical read accepts a hand-authored canonical MANIFEST and rejects every malformed variant: 2-column, 4-column, missing `\n`, CRLF, BOM, uppercase hex, decimal SHA, negative size, unsorted lines.
- `migrate-manifest` converts a 2-column MANIFEST to canonical and the result round-trips.
- `migrate-manifest` aborts and leaves the original MANIFEST untouched when an archived file's sha256 fails verification.
- `restore` against a legacy 2-column MANIFEST raises `CheckpointError` pointing at the migration command (no silent fallback).

---

## 11. Design decisions (resolved 2026-06-25)

The 8 questions in this section were answered by the user on 2026-06-25. Recorded here as the spec.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Git identity = local `molbuilder@<hostname>`** | Set via `git config` (no `--global`). Works without `~/.gitconfig`. Single-user dirs make user-identity attribution redundant. |
| 2 | **Branch deletion hidden behind "Advanced" disclosure + double-confirm dialog** naming the branch and its last-commit short SHA. | Branch deletion is a footgun if accidental; explicit second confirm closes the class. |
| 3 | **Restore = `git restore --source=<ref> .`** (overwrites working tree, HEAD stays on current branch). The same menu also exposes a separate **"Branch from here"** item for users who want to start a new branch at the checkpoint. | Matches "make my files look like that snapshot." Avoids the detached-HEAD-confusion of `git checkout <tag>`. |
| 4 | **Archive hash-verify on restore only.** Not on every sidebar poll. | Hashing 43 MB binaries every 2–5 s is unsustainable; restore is the only moment correctness matters. |
| 5 | **Outer-repo nesting = detect + warn, proceed.** Phase 1 runs `git rev-parse --show-superproject-working-tree`; non-empty result emits one warning line + a doc link, then continues. | Cheap insurance; silent surprise weeks later is the bad outcome. |
| 6 | **Default checkpoint message = `"checkpoint <ISO_TS>"`** when the user submits an empty message. | Low friction encourages frequent checkpointing; tags add meaning after the fact. |
| 7 | **No background polling — explicit refresh only** (revised 2026-06-26; supersedes the original 5 s cadence). The sensor + viewer refresh on (a) directory-enter into a depth-3 run dir with `.git/`, and (b) a manual Refresh control. | A `git status` every few seconds per open sidebar is standing cost for state that only changes on explicit user action; explicit refresh removes the browser + subprocess load entirely. State changing outside the UI (background wrapper start/stop) lags until the next refresh — accepted, because capture is already user-explicit. |
| 8 | **No `molbuilder snapshot status` CLI subcommand.** Users run `git status` directly inside the working dir. | Don't duplicate git. The user-facing surfaces (sidebar sensor + sidebar list view) cover the GUI case; plain `git status` covers the CLI case. CLI surface shrinks accordingly. |

The CLI surface declared in § 14 (item 2) drops `status` accordingly and adds `migrate-manifest` (introduced by § 10.4): `molbuilder snapshot {init, checkpoint, list, tag, branch, diff, restore, prune, migrate-manifest}`.

---

## 12. Testing strategy

| Layer | Test |
|---|---|
| L1 | `RepoState` / `Checkpoint` / `Branch` / `Tag` shape pinned by `test_checkpoint_types.py`. |
| L2 | End-to-end on a tmp_path: init, two explicit checkpoints, tag, branch off, checkpoint on branch, restore. Pinned by `test_checkpoint_lifecycle.py`. |
| L2 | Wrapper isolation: rendering + running a wrapper against a checkpointed dir produces NO commits and NO archive activity. Pinned by `test_checkpoint_wrapper_isolation.py` (load-bearing for P4). |
| L2 | Nested-repo refusal: Phase 1 init refuses on a dir whose children contain `.fdf` / `.py` / `.run.sh`. Pinned by `test_checkpoint_nested_refusal.py` (load-bearing for P5). |
| L2 | Binary archive: a .DM is correctly archived + restored bit-for-bit (SHA256 round-trip). Pinned by `test_checkpoint_binary_archive.py`. |
| L2 | Empty checkpoint: clicking "Checkpoint now" on a clean tree returns `checkpoint=None` and does not create a commit. Pinned by `test_checkpoint_clean_tree.py`. |
| L3 | HTTP routes: each endpoint returns the documented envelope; restore on dirty tree returns `ok:false`. Pinned by `test_checkpoint_routes.py`. |
| L3 | Sidebar sensor refresh contract: `/api/checkpoint/state` returns the cheap shape and does **not** walk `.binsnapshots/` (archive fields stay at `0`/`[]` even when archives exist on disk); a git failure returns a structured `{ok:false, error}` envelope the sensor renders as a 🔴 error pill (not an unhandled crash); and `checkpoint.js` carries no `setInterval` (refresh is explicit — directory-enter + manual Refresh — never polled). Pinned by `test_checkpoint_sensor_js.py`. |
| L3 | Graph viewer: nodes render in DAG order; tag chips clickable; branch lanes correct for a fork-merge pattern. Pinned by Playwright `test_checkpoint_graph_e2e.py`. |

---

## 13. Anti-patterns we refuse

- **No `--no-verify`-style "skip git" flag on the wrapper.** Violates P3.
- **No global state in the JS module** — every method takes the current project path explicitly, mirroring the projects-sidebar contract.
- **No "smart" merge attempt** — if a restore would create a conflict, we refuse with a one-line message pointing at the offending file. Conflict resolution is the user's call; we are not a merge tool.
- **No filesystem walking outside the working dir.** The Repo object never reads above its `path`.
- **No HTTP write endpoint that takes shell strings.** Every write route accepts structured fields (ref names, paths) and constructs the git argv internally.

---

## 14. Implementation plan

| # | Item | Effort | Depends on |
|---|---|---|---|
| 1 | `molbuilder/checkpoint.py` — L1 types + L2 `Repo` (init / checkpoint / state / list / tag / branch / restore / prune / diff) | ~4 hours | — |
| 2 | `molbuilder/cli.py::snapshot` CLI group: `init`, `checkpoint`, `list`, `tag`, `branch`, `diff`, `restore`, `prune` (per § 11 decision 8 — no `status` subcommand; users run `git status` directly) | ~2 hours | 1 |
| 3 | `web/blueprints/checkpoint.py` — HTTP routes per § 8 | ~2 hours | 1 |
| 4 | Vendor `@gitgraph/js` to `static/vendor/gitgraph/` (download + LICENSE + integrity SHA) | ~30 min | — |
| 5 | `static/lib/projects/checkpoint.js` — sensor badge + vertical-split layout management + list view + detail panel + action menus | ~3 hours | 3 |
| 6 | Graph viewer mount (default view in bottom pane), template-extend tied to projects-sidebar.css palette | ~3 hours | 4, 5 |
| 7 | `envs/recipes.py` — add `git` to **every** env's `conda_packages`: `molbuilder-host`, `molbuilder-siesta`, `molbuilder-siesta-gpu`, `molbuilder-pySCF`, `molbuilder-MDtools`, `molbuilder-tests`. No matter which env the wrapper activates, `git` is on PATH. | ~30 min | — |
| 8 | Bootstrap preflight (`git --version` + path under env prefix check; refuses to advance if git is missing or shadowed by system git pointing outside the env) | ~30 min | 7 |
| 9 | Doc update — `design.md` index + decision-log entry on landing | ~30 min | all |
| 10 | Test suite (§ 12) | ~3 hours | 1–6 |
| **Total** | | **~1 day + 1.5 days** | |

≈ **2.5 days end-to-end** for the full feature. Can be broken into two PRs: (a) Python module + CLI + tests (~1 day), (b) HTTP routes + sidebar UI (sensor + vertical-split layout + graph viewer with `@gitgraph/js` + tests, ~1.5 days).

**Explicitly NOT in the plan**: any modification to `runwrap.py`. The wrapper stays git-agnostic per P4.

---

## 15. References

- [`projects-sidebar.md`](projects-sidebar.md) — sidebar architecture this design extends.
- [`web-api.md`](web-api.md) § 1.6 — HTTP envelope rules used for all checkpoint endpoints.
- [`script-execution.md`](script-execution.md) — the runwrap contract this design adds pre/post hooks to.
- [`workspace-contract.md`](workspace-contract.md) § 4.1 — sole-persistence-key rule the sidebar collapsed state respects.
- Pro Git book § 3 (branches), § 7.6 (rewriting history), § 7.10 (refs) — the underlying git operations.
- `git(1)` man pages: `git-init`, `git-add`, `git-commit`, `git-tag`, `git-branch`, `git-restore`, `git-log`, `git-diff`, `git-rev-parse`.
- [`@gitgraph/js`](https://github.com/nicoespeon/gitgraph.js) — the SVG git-graph rendering library vendored in `web/static/vendor/gitgraph/`. MIT license. Source kept verbatim alongside the LICENSE file per the existing CodeMirror vendor pattern.

---

*Pre-implementation design. Approved-as-of dates and changes land in [`design.md`](../design.md) decision log. Tasks tracked: #33 (renamed to "git-based run-checkpoints + sidebar viewer"), #34 (stage 4 preset + advisories — independent surface that consumes the checkpoint module but is not blocked on it).*
