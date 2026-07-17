# Spec — Selection model (Projects sidebar + tab contract)

> **Scope of this doc**: the *selection cursor* contract — what
> `current_dir` / `current_file` mean, how tabs inquire/subscribe, and
> the full `/api/files/*` endpoint list.
>
> For the sidebar as a whole — architecture, public API surface, data
> shapes, CSS class catalogue, lock model, visual states, gaps — see
> the canonical reference at
> [`docs/protocols/projects-sidebar.md`](projects-sidebar.md).

**Module(s)**: sidebar JS is split across small ES modules:

  * `static/lib/projects-sidebar.js`     -- entry point (imports + bootstrap)
  * `static/lib/projects/api.js`         -- HTTP wrappers (no DOM, no state)
  * `static/lib/projects/state.js`       -- sessionStorage + Inquire API + writeFile/saveToWorkspace
  * `static/lib/projects/list.js`        -- breadcrumb + entry list + per-entry ⋯ kebab menu + openDir
  * `static/lib/projects/mutation-bar.js` -- header action bar (New project / New folder / Upload buttons)
  * `static/lib/projects/dialogs.js`     -- modal dialogs (rename, move/copy dest picker, upload, confirm)
  * `static/lib/projects/preview.js`     -- file-preview modal (CodeMirror 5; Find + Download; view-only above 1 MB)

Loaded via `<script type="module">` -- no bundler.  Template:
`molbuilder/web/templates/_projects_sidebar.html`.  Backend:
`molbuilder/web/blueprints/files.py`.
**Tests**: `tests/test_web_files.py`.

This document is the canonical contract for how the Projects sidebar
and the per-tab UI agree on what file or directory the user is
currently working with.  The model is intentionally minimal: the
sidebar is a passive state holder + file-manipulation widget; tabs
inquire when they need information.

## 1. Roles

| Component | Role | Knows about |
|---|---|---|
| **Projects sidebar** | File browser; publishes selection state; file-system operations (mkdir, future rename/upload/delete) | The `projects/` filesystem tree only |
| **Tabs** (Build, Modify, Spectra, Watch) | Each owns its own UX for loading inputs + generating outputs; pulls selection from the sidebar when needed | Their own file types + their own UI; pull `window.molbuilder.projects.*` |
| **Backend `/api/files/*`** | Browse + manipulate the `projects/` tree under path-validation | The `projects/` root only |

The sidebar does **not** know which tab consumes which file type.
Cross-tab navigation is a normal user gesture; each tab's own UI
exposes "Load from current selection" when its file types match.

## 2. Selection state

Two slots in `sessionStorage`.  Always coherent.

| Key | Semantics | Set when |
|---|---|---|
| `molbuilder.current_dir`  | Absolute path of the directory the sidebar is currently displaying.  Always inside the picker's root (`projects/`). | Sidebar navigates into a directory; or a file is clicked (then = the file's parent). |
| `molbuilder.current_file` | Absolute path of the currently-selected file.  Empty string when only browsing. | Sidebar selects a file.  Cleared when the user navigates into a new directory. |

**No other selection state.**  No multi-file selection in v1.  No
typed slots ("input_structure" vs "compare_file" etc.) -- one cursor
pair is enough for every v1 workflow.

## 3. The Inquire API

The sidebar exposes a small synchronous read API on a global object:

```js
window.molbuilder.projects.getCurrentDir()          // -> string  (always set)
window.molbuilder.projects.getCurrentFile()         // -> string  ("" when no file)
window.molbuilder.projects.onChange(callback)       // -> unsubscribe_fn
window.molbuilder.projects.readCurrentFile()        // async -> {path, text} | null
window.molbuilder.projects.relativeToProjects(path) // -> string  (display helper)
window.molbuilder.projects.refresh()                // async -> Promise<void>
window.molbuilder.projects.saveToWorkspace(text, filename, opts)
                                                    // async -> null | {ok, path, relPath} | {ok:false, error}
```

``saveToWorkspace`` is the single source of truth for the
"generate-and-save" flow every tab needs.  Internally posts to
``/api/files/write`` against ``<current_dir>/<filename>`` and
auto-refreshes the sidebar on success.

  * Returns ``null`` silently when ``current_dir`` is unset OR at
    the picker root -- callers fall back to a local Download / Copy
    button without showing an error.
  * Returns ``{ok:true, path, relPath}`` on a successful write;
    ``relPath`` is ``path`` shortened to ``projects/...`` for status.
  * Returns ``{ok:false, error}`` on backend failure (409 conflict,
    400 bad path, 403 perm denied).  Backend messages already say
    what to do; callers display ``error`` verbatim.

  ``opts.overwrite`` (default ``false``) opts in to clobbering an
  existing file.  ``opts.expected_mtime`` (number) enables the
  concurrent-edit detection used by the future edit-and-save flow.

* `onChange(cb)` fires `cb({dir, file})` on every selection change
  AND once immediately on registration (so subscribers can initialise
  from current state without a separate `getCurrent*` call).
* `readCurrentFile()` is a convenience for the common "I want the
  text" case -- fetches `/api/files/read?path=<file>` and unwraps.
* `refresh()` re-lists the current directory.  Call from a tab after
  it creates a new file so the user sees it in the sidebar without
  re-clicking.

## 4. Tab contract

Each tab that wants to consume sidebar selections:

> **Updated 2026-06-07 (Phase B.5):** auto-load on `onChange` is
> the LEGACY pattern.  The current model is single-click =
> preview (`onChange`), double-click = commit (`onCommit`).  Tabs
> with editable state subscribe to `onCommit` and gate their
> "use this file" action through a dirty-state warning modal.
> See [`projects-sidebar.md`](projects-sidebar.md) § 6 + § 9
> for the universal model and [`tabs/architecture.md`](../tabs/architecture.md)
> § 9.2 for the per-tab status (B.5.3
> wired Build + Spectra to onCommit; B.5.2 wired Molbuilder
> directly).

1. **Subscribes via `onCommit`** (current model) for tabs whose
   state would be lost by an auto-load — Molbuilder workspace,
   Build form, Spectra form.  Single-click stays preview;
   double-click commits.
2. **Subscribes via `onChange`** (read-only inspector pattern)
   for tabs that have nothing to lose — `/results` auto-mounts
   the inspector on single click because nothing is editable.
3. **Workspace indicator (optional)**: displays the current dir near
   the tab's Generate button so the user knows where new output will
   land.  Updates live via the same `onChange` subscription.

Reference implementations in:
* `molbuilder/web/templates/modify.html` (Molbuilder tab)
* `molbuilder/web/templates/index.html` (Structure-optimization)
* `molbuilder/web/templates/spectra.html` (Spectrum-calculation)
* `molbuilder/web/templates/results.html` (Results)

## 5. Sidebar interaction rules

| User action | Sidebar effect | Selection state change |
|---|---|---|
| Click a directory entry | Drills in: displays that dir's children | `current_dir` = new dir; `current_file` = "" |
| Click a file entry | Marks it selected (visual highlight) | `current_dir` = file's parent; `current_file` = file's abs path |
| Click a breadcrumb segment | Jumps to that ancestor dir | Same as "click a directory" |
| Click `New folder`, submit form with valid name | Creates the dir via `POST /api/files/mkdir` and navigates into it | `current_dir` = new subdir; `current_file` = "" |

The sidebar never:

* navigates the user to a different tab
* triggers a tab's loader
* auto-fires anything on directory or file selection (other than
  updating its own state)

## 6. File-manipulation endpoints

| Endpoint | Body | Validation | Status |
|---|---|---|---|
| `POST /api/projects/create` | `{name}` | name = `^[A-Za-z0-9_-]+$`; strict-create (409 if exists); atomic + READMEs | **Shipped** |
| `POST /api/files/mkdir`     | `{parent, name}` | parent inside an allowed root; name validated against the depth-aware rule | **Shipped** |
| `GET  /api/files/read`      | `path=...&max_bytes=...` | text content (size-capped); used by the file-preview modal | **Shipped** |
| `POST /api/files/rename`    | `{path, new_name}` | same depth-aware naming + conflict rules as mkdir | Deferred (not stubbed) |
| `POST /api/files/upload`    | multipart `{file, target_dir}` | (a) destination depth ≥ 1; (b) inside an allowed root; (c) filename regex allows dots for extensions; (d) inside `user/` depth 2+ free-form; (e) name conflict at destination = 409 | **Stub** (returns 501 with explanatory message; UI surface fully wired) |
| `POST /api/files/write`     | `{path, text, expected_mtime}` | path inside allowed root; mtime-based conflict detection (409 on mismatch); UTF-8 only | **Stub** (501; the file-preview modal's Save button is disabled with `title="coming soon"`) |
| `DELETE /api/files/delete`  | `{path, recursive}` | path inside allowed root; cannot delete picker root or a canonical-topic dir at depth 1; recursive flag required for non-empty dirs | **Stub** (501; UI shows per-entry × on hover at eligible depths + JS confirm dialog before sending the request) |

`mkdir`'s name validation depends on the parent's depth inside the
picker's root:

* depth 0 (parent = `projects/`)              → ``validate_name`` (`^[A-Za-z0-9_-]+$`)
* depth 1 (parent = `projects/<proj>/`)       → ``validate_topic`` (must be in `CANONICAL_TOPICS`; see § *Canonical topics*)
* depth 2 (parent = `projects/<proj>/<topic>/`) → ``validate_name`` (structure name; same regex)
* depth 3+ (inside a `<structure>/` or any `user/` descendant) → ``validate_name`` (ad-hoc subdir; the user is the decider)

### Canonical topics

``molbuilder.projects.CANONICAL_TOPICS`` (the names accepted at depth 1):

| Topic | Flavour | Purpose |
|---|---|---|
| ``structure`` | storage | curated input structures (`.xyz` / `.pdb` / `.cif`); flat dir |
| ``pseudopotential`` | storage | SIESTA pseudos for this project; flat dir |
| ``optimization`` | run-topic | geometry relaxation runs |
| ``frequency`` | run-topic | Hessian / harmonic frequencies / thermochemistry |
| ``spectrum`` | run-topic | Raman / IR (see Spectra tab spec) |
| ``transport`` | run-topic | electron transport (TBTrans / NEGF) |
| ``single-point`` | run-topic | single-energy + property calcs at fixed geometry |
| ``scan`` | run-topic | potential-energy-surface scans along a coordinate |
| ``user`` | **free-form** | no rules below this dir; the user decides the structure |

`user/` is the escape hatch from the strict depth-1 vocabulary.  Use
it for notes, ad-hoc experiments, anything that doesn't fit the
canonical analyses.  Inside `user/` (depth 2+), `validate_name`'s
regex is the only constraint -- arbitrary subdir names are accepted.

### Sidebar create-button visibility

The sidebar exposes two foldable creation sections.  The order is
fixed (project first, subdir second) and visibility is depth-aware:

| User is at | `New project` | `New folder` |
|---|---|---|
| `projects/` (depth 0) | visible | **hidden** -- keeps the root clean; only project dirs live there |
| depth 1+ | visible (always; you can start a new project from anywhere) | visible |

The visibility toggle is driven by `projects-sidebar.js`'s
`updateMkdirContext` (`section.hidden = atRoot`).

### Project bootstrap output

`POST /api/projects/create` (the `New project` flow) writes:

```
projects/<name>/
├── README.md               <- project-level layout reminder
├── structure/
│   └── README.md
├── pseudopotential/
│   └── README.md
├── optimization/
│   └── README.md
├── frequency/
│   └── README.md
├── spectrum/
│   └── README.md
├── transport/
│   └── README.md
├── single-point/
│   └── README.md
├── scan/
│   └── README.md
└── user/
    └── README.md
```

Each README is short (~5 lines), explains the dir's purpose, and
points the reader at the relevant spec.  The bootstrap is atomic --
partial-failure rolls the project dir back via ``shutil.rmtree``.

Strict conflict semantics: the project dir must NOT already exist.
409 is returned with a clear message ("project 'foo' already exists
at /.../projects/foo.  Pick a different name; use 'New folder'
from inside the existing project to add to it.")

## 7. Why this design

| Choice | Reason |
|---|---|
| Pull (Inquire) not push | Sidebar stays content-agnostic. Adding a new tab doesn't touch the sidebar. Each tab independently testable. Mirrors JupyterLab / VS Code where the file tree is pure navigation; actions live in the editors. |
| One cursor pair | Covers every v1 use case (load existing, set workspace, browse, derive-from). Multi-slot (`input_structure`, `compare_file`, ...) is a future extension if a multi-input workflow earns it. |
| File-manipulation buttons in sidebar | These operate **on the projects tree itself**, not on tab state.  `New folder` belongs with the file browser; "Load this Spectra file" doesn't. |
| Single `projects/` root | `projects/` is molbuilder's source of truth for run state. Files outside (laptop downloads) must be moved/copied in; the explorer scope mirrors molbuilder's scope. |
| No auto-load on tab arrival | An auto-load races the user's clicks (they might be about to paste, or open a different file).  Explicit "Load from current selection" is one click; the user owns when it fires. |

## 8. Anti-patterns (don't reintroduce)

| Pattern | Why retired |
|---|---|
| `OPEN_TARGETS` dict in sidebar JS mapping extensions to tabs | Hardcoded per-tab knowledge in the sidebar.  Replaced by Inquire API: each tab handles its own extension matching. |
| `window.molbuilderTabAutoLoad` per-tab auto-load on `DOMContentLoaded` | Implicit action contradicts the "explicit click" principle of § 7.  Replaced by the user clicking "Load from current selection". |
| `<div id="projects-banner">` "Use this file" banner in each tab | Lived between sidebar and tab; required a separate `lib/projects-selection.js` shim to wire up.  Replaced by tab-owned button + the simpler Inquire API. |
| Browser `<input type="file">` for "load from your computer" | A server-side script can't read the laptop file; the contract was misleading.  See `docs/protocols/web-api.md` § Phase-2 design for the eventual real-upload feature. |
