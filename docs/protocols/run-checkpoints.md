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

> Every state of a working directory — `.fdf`, `.out`, `.molwatch.log`, `.DM`, `.TSHS`, sidecars, user notes — must be **recoverable, diffable, and branchable** without the user thinking about it.

The current model (rolling overwrites, manual file-shuffling) loses scientific work the moment a parameter sweep is attempted. The user already understands `git tag`, `git branch`, `git checkout` — the design re-uses that mental model rather than inventing a parallel snapshot vocabulary.

**Scope**:
- Every working dir under `projects/` is a git repository.
- Every wrapper invocation produces at least one commit.
- Big binary state is archived **outside** the git object database but **keyed by** the git SHA.
- The Projects Sidebar surfaces the commit history, branches, tags, and binary archives as first-class navigation.

**Non-goals**:
- This is **not** a backup tool. We do not push to a remote unless the user configures one.
- This is **not** a replacement for the workspace persistence contract ([`workspace-contract.md`](workspace-contract.md)). The workspace remains the sole source of truth for browser-side UI state; this design owns disk-side state.
- This does **not** integrate with the outer molbuilder git repo. The `.git` directories created here are nested git repos; the outer molbuilder repo's `.gitignore` already excludes `projects/*`.

---

## 2. Principles

**P1. Git is the source of truth for everything textual.** `.fdf`, `.out`, `.molwatch.log`, sidecars (`.molstruct.json`, `.transport.json`), configs, READMEs. No parallel "history" mechanism exists for these.

**P2. Binaries are archived by SHA, not committed.** `.DM`, `.HSX`, `.TSHS`, `.TBT.AVTRANS_*` go to `.binsnapshots/<sha>/` inside the working dir. Git tracks the existence of the archive (the directory's `.gitkeep`), not the contents.

**P3. Every wrapper invocation produces exactly one commit (success or failure).** The audit trail has no gaps.

**P4. Auto-commit is silent and atomic.** The user doesn't see `git commit` messages flying past; the wrapper does it and moves on.

**P5. Tagging is for humans, branching is for experiments.** Tags (e.g. `stage3-converged`) are semantic milestones the user creates. Branches (e.g. `stage4-tzp`) carry experimental parameter sweeps. The system never auto-creates tags; only auto-creates commits.

**P6. The sidebar is the primary navigation UI. The CLI is the secondary.** The CLI exists for completeness and SSH workflows; the sidebar is where most users will operate.

**P7. Git availability is a hard requirement.** Bootstrap fails if `git` is not on PATH. This is enforced once, at install time, not deferred to per-run failures.

**P8. The design is opt-in per project dir, not per run.** Once a working dir has a `.git`, every wrapper invocation participates. There is no "skip git for this run" flag — that would defeat P3.

---

## 3. Architecture

### 3.1 Modules

| Layer | Module | Owns |
|---|---|---|
| **L1**: data model | `molbuilder/checkpoint.py` (new) | `Checkpoint`, `Branch`, `Tag` dataclasses; pure git-state representation. |
| **L2**: orchestration | `molbuilder/checkpoint.py::Repo` | Init, pre-run hook, post-run hook, binary archive write/read. Wraps `subprocess.run(["git", ...])`. |
| **L2**: wrapper integration | `molbuilder/runwrap.py` (modify) | Calls `Repo.pre_run()` before activate, `Repo.post_run()` after exit. |
| **L2**: CLI | `molbuilder/cli.py::cmd_snapshot_group` (new) | `molbuilder snapshot {init,list,tag,diff,restore,branch,prune}` |
| **L3**: HTTP routes | `molbuilder/web/blueprints/checkpoint.py` (new) | Read-only endpoints + tag/branch/restore POST endpoints. |
| **L3**: sidebar UI | `molbuilder/web/static/lib/projects/checkpoint.js` (new) | The graph viewer, sensor badge, action menus. |

### 3.2 Surface separation

Same separation principle as [`projects-sidebar.md`](projects-sidebar.md) § 3: the Python module is content-agnostic git plumbing; the JS owns the visualisation; the HTTP layer is a thin contract that does not embed view logic.

### 3.3 Where git binaries come from

The host conda env (`molbuilder-host` per [`README_install.md`](../README_install.md)) gains `git` in its `conda_packages` tuple. The bootstrap step's preflight (`molbuilder envs bootstrap`) verifies `git --version` returns ≥ 2.20 (when `git restore` landed) and fails loudly otherwise with the install instruction.

---

## 4. Lifecycle

### 4.1 Phase 0 — Bootstrap (once per host)

**Trigger**: `molbuilder envs bootstrap`.
**Action**:
1. Check `git --version` ≥ 2.20.
2. If absent: emit one-line error `"molbuilder requires git ≥ 2.20; install via your distro's package manager."` and exit non-zero.

### 4.2 Phase 1 — Working-dir init (auto, first wrapper run)

**Trigger**: `runwrap.write_run_wrapper` invoked against a directory without `.git/`.
**Action**: equivalent to:
```
git init -q
git config user.email "molbuilder@<hostname>"
git config user.name  "molbuilder"
git config commit.gpgsign false
write .gitignore         (§ 9.1)
write .binsnapshots/.gitkeep
git add .gitignore .binsnapshots/.gitkeep *.fdf *.psml *.molstruct.json
git commit -q -m "molbuilder: initial state of <dir>"
```
**Result**: a one-commit repo with the configs visible; binaries not yet present.

### 4.3 Phase 2 — Pre-run checkpoint (every wrapper invocation)

**Trigger**: `runwrap.write_run_wrapper` invoked.
**Action**:
1. If `git diff-index --quiet HEAD --` fails (= dirty), auto-commit:
   `"molbuilder: uncommitted state from previous session"`.
2. `PRE_SHA = git rev-parse HEAD`.
3. Identify big binaries present (`.DM`, `.HSX`, `.TSHS`, `.TBT.AVTRANS_*`).
4. `mkdir -p .binsnapshots/$PRE_SHA && cp -an <binaries> .binsnapshots/$PRE_SHA/`.
5. `git tag -m "before <stage>-<run-id>" pre-<stage>-<run-id>`.
**Failure mode**: any git command failure raises `CheckpointError`; the wrapper refuses to launch. This is intentional — running without a checkpoint defeats P3.

### 4.4 Phase 3 — Post-run commit (wrapper exit, ALL paths)

**Trigger**: `runwrap`-generated script's `trap EXIT` handler.
**Action**:
1. Capture exit code, parse `.out` for `Etot` + `Max force` if SIESTA.
2. `git add .` (new `.out`, `.molwatch.log`, updated `.DM` is excluded by `.gitignore`).
3. `git commit -q -m "<stage>-run<N>: <status>, Etot=<E> eV, max_F=<F> eV/Å"`.
4. Move new big binaries to `.binsnapshots/$NEW_SHA/`.

If the wrapper is killed mid-run (`SIGTERM`/`SIGKILL`), no commit happens here; the **next** Phase 2 catches the uncommitted state (so we never miss data, but the commit shows up under the NEXT run's banner). This is a deliberate trade-off — making the post-run commit run reliably from a trap is fragile.

### 4.5 Phase 4 — Semantic tagging (user-driven)

`molbuilder snapshot tag <label> [--message TEXT]` or sidebar context menu. Equivalent to `git tag -a <label> HEAD -m TEXT`. No auto-tagging — semantic labels are the user's vocabulary.

### 4.6 Phase 5 — Experimental branching (user-driven)

`molbuilder snapshot branch <name>` or sidebar UI. Equivalent to `git checkout -b <name>`. The wrapper runs against whatever branch HEAD is currently on; subsequent commits land on that branch.

### 4.7 Phase 6 — Inspection / restore

| User action | Underlying git op | Binary handling |
|---|---|---|
| List | `git log --graph --oneline --decorate --all` | — |
| Diff text | `git diff <a>..<b> -- '*.fdf' '*.out' '*.molwatch.log'` | — |
| Restore to a tag | `git restore --source=<tag> .` + `cp .binsnapshots/<sha>/* .` | both, sequenced |
| Prune unused binaries | identify SHAs unreferenced by any tag/branch/HEAD; `rm -rf .binsnapshots/<sha>` | — |

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

This is the structure the sidebar sensor reads from on every poll.

---

## 6. UI integration — Projects Sidebar

### 6.1 Where it lives

Inside the existing Projects Sidebar (see [`projects-sidebar.md`](projects-sidebar.md)), a new collapsible panel **"Run history"** mounts below the file tree for the currently-selected project. The panel is hidden when:
- The selected node is a file (only meaningful for directories).
- The selected directory has no `.git/`.
- The user has explicitly collapsed it (state in `sessionStorage` per [`workspace-contract.md`](workspace-contract.md) § 4.1, keyed under `ws.ui.checkpoint.collapsed`).

### 6.2 Sensor (always-visible badge)

A small status pill in the sidebar header for the current project directory:

| Visual | Meaning |
|---|---|
| 🟢 `clean · stage3-converged` | Repo present, working tree clean, last tag shown. |
| 🟡 `N changes` | Uncommitted changes since HEAD. Tooltip: "auto-committed on next run." |
| 🔵 `running` | A wrapper is currently executing in this dir (file lock present). |
| ⚪ `no checkpoints` | Directory has no `.git/` — not yet a tracked working dir. |
| 🔴 `error: <one-line>` | Git command failed; pill is clickable → shows last error. |

Polling: 2 s when the sidebar is visible and the project is selected; suspended otherwise. The endpoint is cheap (single `git status --porcelain` + `git rev-parse HEAD`) so polling cost is bounded.

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

### 6.4 Graph viewer (alternative view)

Toggle button in the panel header: **"List" / "Graph"**.

The graph view renders a vertical commit DAG with branches as parallel lanes (the gitk / GitLens / @gitgraph metaphor):

```
* d3f1a92  stage3-run1                                        [HEAD]
│
│ * e441b2c  stage4-tzp-run0: converged, Etot=-1743311.05    [stage4-tzp]
│ │
│/
*  8b22e07  stage3-run0                                       [stage3-converged]
│
*  2c19fa1  initial state
```

- Vertical axis is time.
- Each commit is a node; lanes are branches; merge points draw the natural fork.
- Click a node = select it (highlights, shows details panel on the right).
- Right-click a node = same context menu as the list view.
- Drag a node onto another? **Out of scope** — too easy to fat-finger a `git reset`. Operations are explicit menu items only.
- Library: vendored `@gitgraph/js` (MIT, ~30 KB) or a hand-rolled SVG renderer. Decision deferred to implementation; the rendered shape is fixed by this spec either way.

### 6.5 Live activity overlay

When a wrapper is running (sensor shows 🔵), an extra row at the top of the list:

```
⟳  pre-stage4-tzp-run0           17:03      [running for 3 min]
```

Pulses subtly. Disappears the moment the post-run commit lands.

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
        """Run Phase 1.  Idempotent; no-op if .git exists."""
    def state(self) -> RepoState:
        """Cheap (= sensor-polling) snapshot of current repo state."""
    def list_checkpoints(self, *, since: Optional[str] = None,
                                 limit: int = 50) -> List[Checkpoint]: ...
    def list_branches(self) -> List[Branch]: ...
    def list_tags(self) -> List[Tag]: ...
    def diff(self, ref_a: str, ref_b: str,
                  pathspec: List[str] = None) -> str:
        """Unified-diff text.  pathspec defaults to text-only globs."""
    def tag(self, label: str, *, message: str = "", at: str = "HEAD") -> Tag:
        """Phase 4 — user-driven semantic tag."""
    def branch(self, name: str, *, at: str = "HEAD",
                                   checkout: bool = True) -> Branch:
        """Phase 5 — user-driven branch."""
    def restore(self, ref: str, *, include_binaries: bool = True) -> None:
        """Phase 6 — git restore + binary archive copy.  RAISES if working
        tree is dirty (loud refusal; user must commit or discard first)."""
    def prune_archives(self, *, keep_refs_only: bool = True) -> List[str]:
        """Remove .binsnapshots/<sha>/ for SHAs unreferenced by any
        tag/branch/HEAD.  Returns list of removed SHAs.  Dry-run unless
        ``keep_refs_only=True``."""
    # Wrapper hooks.
    def pre_run(self, *, stage: str, run_id: str) -> str:
        """Phase 2.  Returns the pre-run SHA."""
    def post_run(self, *, stage: str, run_id: str,
                          exit_code: int,
                          out_file: Optional[str] = None) -> str:
        """Phase 3.  Returns the post-run SHA.  Parses .out for Etot / max_F
        when out_file is provided and category is siesta."""

class CheckpointError(Exception): pass
class GitNotInstalledError(CheckpointError): pass
class DirtyWorkingTreeError(CheckpointError): pass
class NoSuchRefError(CheckpointError): pass
```

Each method delegates to `subprocess.run(["git", ...], cwd=self.path)`. No shell strings; argv lists throughout. Errors propagate verbatim from git.

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
| `POST /api/checkpoint/init` | `{"path": PATH}` | Phase 1 manual trigger (empty-state button). |
| `POST /api/checkpoint/tag` | `{"path", "label", "message", "at"}` | Phase 4. |
| `POST /api/checkpoint/branch` | `{"path", "name", "at", "checkout"}` | Phase 5. |
| `POST /api/checkpoint/restore` | `{"path", "ref", "include_binaries"}` | Phase 6. Refuses on dirty working tree (HTTP 200 + `ok:false` per § 1.6 bucket B — this is a scientific-advisory case, not a protocol error). |
| `POST /api/checkpoint/prune` | `{"path", "dry_run"}` | Phase 6. |

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

## 10. Binary archive layout

```
<working_dir>/
├── siesta-foo.DM                              ← current state, gitignored
├── siesta-foo.TSHS                            ← current state, gitignored
└── .binsnapshots/
    ├── .gitkeep
    ├── 8b22e07c.../                           ← SHA-keyed per checkpoint
    │   ├── siesta-foo.DM                      ← 43 MB
    │   ├── siesta-foo.TSHS                    ← 32 MB
    │   └── MANIFEST                           ← list + sha256
    └── d3f1a92e.../
        └── ...
```

Each archive carries a `MANIFEST` file with `<file>  <sha256>  <bytes>` per line. The MANIFEST is regenerated at archive time; mismatch on restore triggers a hard refusal with the bad file's name.

---

## 11. Open questions for review

1. **Auto-commit threshold for "dirty" detection** — should we commit on every untracked file, or only when a meaningful one (`.out`, `.fdf`, sidecar) appears? My instinct: every file. Catches user-side experimentation that would otherwise disappear.

2. **Git user identity** — local (`molbuilder@<hostname>`) vs. inherit from the user's global git config. I propose local-only by default to avoid surprising HPC users whose `~/.gitconfig` may not exist.

3. **Annotated vs. lightweight tags** — semantic tags should be annotated (carry a message). Pre-run tags can be lightweight (auto-clutter; no human reads the message). Confirm?

4. **Branch deletion via UI** — should the sidebar offer branch deletion? Footgun risk vs. clutter. My recommendation: hide deletion behind a small "advanced" disclosure; double-confirm with a dialog naming the branch + last-commit SHA.

5. **Restore semantics** — `git restore --source=<ref> .` overwrites; `git checkout <tag>` detaches HEAD. Which does the UI "Restore to here" button use? I propose `git restore` plus an option in the menu to "Branch from here" if the user wants to keep working from that point.

6. **Archive integrity check** — should every UI load of the run-history panel hash-verify the archives, or only on restore? I propose only on restore (hashing 43 MB per archive on every poll is wasteful).

7. **Multi-user / shared filesystem** — if two users share a `projects/BDT/` over NFS, does the git lock contention break things? Probably no worse than today's file-overwrite contention, but worth surfacing.

8. **Outer-repo nesting** — for the special case where a user has their own outer git tracking `projects/` (against our convention), our `.git` becomes a nested submodule-like artefact. We can detect this (`git rev-parse --show-superproject-working-tree`) and warn during Phase 1.

---

## 12. Testing strategy

| Layer | Test |
|---|---|
| L1 | `RepoState` / `Checkpoint` / `Branch` / `Tag` shape pinned by `test_checkpoint_types.py`. |
| L2 | End-to-end on a tmp_path: init, two pre-run + post-run cycles, tag, branch off, restore. Pinned by `test_checkpoint_lifecycle.py`. |
| L2 | Wrapper integration: a render → run → exit cycle should produce exactly two new commits + one tag. Pinned by `test_runwrap_checkpoint_hooks.py`. |
| L2 | Binary archive: a .DM is correctly archived + restored bit-for-bit (SHA256 round-trip). Pinned by `test_checkpoint_binary_archive.py`. |
| L3 | HTTP routes: each endpoint returns the documented envelope; restore on dirty tree returns `ok:false`. Pinned by `test_checkpoint_routes.py`. |
| L3 | Sidebar sensor: polling shape, suspends when sidebar hidden, retries on transient git error. Pinned by `test_checkpoint_sensor_js.py`. |
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
| 1 | `molbuilder/checkpoint.py` — L1 types + L2 `Repo` | ~4 hours | — |
| 2 | `runwrap.py` pre/post hooks | ~1 hour | 1 |
| 3 | `molbuilder/cli.py::snapshot` CLI group | ~2 hours | 1 |
| 4 | `web/blueprints/checkpoint.py` — HTTP routes | ~2 hours | 1 |
| 5 | `static/lib/projects/checkpoint.js` — sensor + list view | ~3 hours | 4 |
| 6 | Graph viewer (list view's "Graph" toggle) | ~4 hours | 5 |
| 7 | `envs/recipes.py` — add `git` to host env | ~30 min | — |
| 8 | Bootstrap preflight (`git --version`) | ~30 min | 7 |
| 9 | Doc update — `design.md` index + decision-log entry | ~30 min | all |
| 10 | Test suite | ~3 hours | 1–6 |
| **Total** | | **~half day + 1.5 days** | |

≈ **2 days end-to-end** for the full feature. Can be broken into two PRs: (a) Python module + CLI + wrapper hooks + tests (~1 day), (b) HTTP routes + sidebar UI + tests (~1 day).

---

## 15. References

- [`projects-sidebar.md`](projects-sidebar.md) — sidebar architecture this design extends.
- [`web-api.md`](web-api.md) § 1.6 — HTTP envelope rules used for all checkpoint endpoints.
- [`script-execution.md`](script-execution.md) — the runwrap contract this design adds pre/post hooks to.
- [`workspace-contract.md`](workspace-contract.md) § 4.1 — sole-persistence-key rule the sidebar collapsed state respects.
- Pro Git book § 3 (branches), § 7.6 (rewriting history), § 7.10 (refs) — the underlying git operations.
- `git(1)` man pages: `git-init`, `git-add`, `git-commit`, `git-tag`, `git-branch`, `git-restore`, `git-log`, `git-diff`, `git-rev-parse`.

---

*Pre-implementation design. Approved-as-of dates and changes land in [`design.md`](../design.md) decision log. Tasks tracked: #33 (renamed to "git-based run-checkpoints + sidebar viewer"), #34 (stage 4 preset + advisories — independent surface that consumes the checkpoint module but is not blocked on it).*
