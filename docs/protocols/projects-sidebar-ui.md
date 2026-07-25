# Projects sidebar — UI + file-content preview contract (aspect E)

**Status**: design reference.  This is **aspect E** of the Projects
Sidebar module: the **user-facing UI** of the sidebar (breadcrumb,
entry list/tree, mutation bar, drawer, visual states, width resize,
visibility rules) **and** the **file-content preview modal**
(`lib/projects/preview.js`).

**This doc is the single source of truth for the sidebar's UI.**  The
master — [`projects-sidebar.md`](projects-sidebar.md) — owns the
mission, the public `window.molbuilder.projects.*` API, the internal
architecture, the byte-I/O + paths contract (§ 5), and the lock model
(§ 8).  This doc owns only *how those pieces are presented and driven
from the DOM*.  When a UI affordance calls the public API, the API
contract is the master's, not this doc's.

**Cross-references**:
* [`projects-sidebar.md`](projects-sidebar.md) — the module MASTER
  (public API, capabilities § 5, subscribe model § 6, lock model § 8).
* [`web-api.md`](web-api.md) § 3 — the wire endpoints the UI drives:
  `read` / `read_range` / `write` (with `expected_mtime`) / `download`
  / `list` / `stat`.
* [`save-flow.md`](save-flow.md) — the **structure** Save panel (aspect
  D).  That is a **structure-aware** save (serialises the model +
  sidecar via `projects.parser.saveMolecule`); it is **NOT** the
  generic file editor documented here.  The preview modal's Save
  writes **raw bytes** to one path; the Save panel writes a whole
  structure + its `.molstruct.json` pair.  Keep them distinct.
* [`selection.md`](selection.md) — preview-vs-commit selection
  semantics (aspect A) behind the single/double-click gestures.

**A note on state authority.**  Per the master's Principle 3, every
visual state below is *derived* from a state mutation that published a
change — no state may appear "because of CSS alone".  The UI is a
function of `(cursor, lock, root)`; it never becomes the source of
truth.

---

## 1. Visual structure

The sidebar is a fixed left-edge panel (`.projects-sidebar`,
server-rendered from `_projects_sidebar.html`, styled by
`projects-sidebar.css`).  Top to bottom:

| Region | Element(s) | Purpose | § |
|---|---|---|---|
| **Header + action bar** | `.ps-header`, `#ps-create-project-btn` / `#ps-create-folder-btn` / `#ps-create-upload-btn`, `#ps-collapse-toggle` | Title, the three mutation buttons, the collapse control | § 5, § 7 |
| **Breadcrumb** | pill chips with `›` separators | Where am I; click a segment to navigate up | § 3 |
| **File-type filter** | `#ps-filter-input`, `#ps-filter-clear` | Hide non-matching files (folders always shown) | § 4.1 |
| **Entry list** | `.ps-list` > `.ps-entry` rows, each with a `⋯` kebab | Directories + files; single-click preview, double-click commit; per-entry actions | § 4, § 5 |
| **Lock banner** | rendered while locked | Reason + Cancel; stays interactive when the rest is faded | master § 8 |
| **Width-resize handle** | `#ps-resize-handle` | Drag the right edge to resize | § 6 |
| **Off-panel affordances** | `#ps-mobile-toggle` / `#ps-mobile-backdrop`, `#ps-collapsed-handle` | Narrow-viewport drawer + desktop collapsed dock | § 7 |
| **Preview modal** | `#ps-preview-modal` | View / edit / save any file's raw content | § 9 |

The DOM is a **server-rendered contract**: consumers query by the IDs
+ classes above; they never assume structure beyond it.  Don't reach
into the sidebar DOM from a tab — use the master's public API.

---

## 2. Visual states

The conceptual UI states the design recognises.  Every state must be
reachable from a known state mutation; no state may appear "because of
CSS alone".

| State | When | What the user sees |
|---|---|---|
| **Idle** | Page load, cursor unset | Breadcrumb at root; project list; "no file selected" |
| **Browsing** | After navigating | Breadcrumb shows path; entries for that dir |
| **File selected** | After clicking a file | Highlight on entry; "Selected: <name>" status |
| **Empty directory** | Dir has no children | Empty list area with a `.is-empty` modifier |
| **Listing error** | Backend returned `{ok:false}` | Inline error row in the list; cursor reset to the attempted path |
| **Locked** | A pipeline holds the lock | Sidebar contents faded + non-interactive; banner with reason + Cancel (master § 8) |
| **No project root** | Init's `apiRoots` returned empty | List replaced with a "no roots configured" message; lock UI still functional |
| **Preview open** | User clicked View on a file | Modal over the page; closes on ESC / backdrop / button (§ 9) |
| **Kebab menu open** | User clicked ⋯ on an entry row | Per-entry action menu (View / Download / Rename / Move / Copy / Delete); auto-dismisses on outside click + ESC + scroll |
| **Dialog open** | Any mutation modal active | `<dialog>` overlays page; primary input focused; ESC / Cancel resolve as null; only one dialog at a time |
| **Dialog error** | Validation or backend rejection | Inline error in dialog; form keeps current value for retry |
| **Resizing** | User dragging the right-edge handle | `--ps-w` updates live; body `cursor:ew-resize`; release persists to localStorage |

---

## 3. Breadcrumb

Each path segment renders as a pill-shaped chip with a `›` separator
between segments.  The root chip carries a small ⌂ glyph; the current
(last) chip uses an accent fill so the user can tell where they are at
a glance.  Non-current chips are keyboard-focusable (`role=link`,
Enter/Space to navigate) and drive `navigateTo` on the target path.

---

## 4. Entry list / tree

`.ps-list` renders one `.ps-entry` per directory child.  Directories
and files are visually distinguished; the row carries `dataset.path`
so click handlers resolve the absolute path without a re-query.

**Interaction** (semantics: [`selection.md`](selection.md)):

* **single-click** a file → *preview/candidate*: `setShared(dir, file)`
  → publishes `onChange`.
* **double-click** a file → *commit*: `publishCommit(dir, file)` →
  publishes `onCommit`; tabs act on this.
* **single-click** a directory → `navigateTo` into it (re-lists +
  re-renders the breadcrumb).
* **per-row `⋯` kebab** → the contextual action menu (§ 5).

`.is-selected` marks the current file; the highlight is applied by the
list's single `onChange` subscriber (master Principle 3 / § 13 M2), not
inline from the click handler.

### 4.1 File-type filter

A free-text filter input between the breadcrumb and the entry list
hides files whose name doesn't match the query.  Folders always stay
visible — they're navigation, not data — so the user can drill into a
sub-folder even when the filter is active.

Match rules:

* **Default**: case-insensitive substring (`"wat"` matches `water.xyz`).
* **Leading-dot shortcut**: `".xyz"` matches files whose name ends in
  `.xyz` only (not `water.xyz.bak`).  Use this for "show me all XYZ
  files in this directory".
* **Empty query**: every file visible (the no-filter state).

DOM:

* `#ps-filter-input` (`<input type="search">`) — the query.
* `#ps-filter-clear` (`<button>`) — × button revealed only when the
  filter is active; click resets state + focuses the input.
* `.ps-list .ps-entry.is-hidden { display: none }` — hidden entries
  stay in the DOM (click handlers + `dataset.path` stay live) so
  clearing the filter doesn't require a re-render.
* `.ps-list.is-filtered-empty::before` — surfaces "(no match)" when
  every entry is hidden by the filter (parallel to the existing
  `.is-empty` "(empty)" affordance).

State persists in `sessionStorage` under
`molbuilder.projects_sidebar_filter`.  Re-applied after every `openDir`
so a previously-active filter doesn't reset on directory change.

JS lives in `lib/projects/list.js`: `_filterMatches`, `_applyFilter`,
`_initFilter`.

---

## 5. Mutation UX (buttons + modal dialogs)

The sidebar's mutation surface is **buttons + modal dialogs**, not
inline forms.  Two trigger points.

### 5.1 Header action bar

Three SEPARATE buttons at the top of the sidebar (revised 2026-06-12;
the earlier v1 single "+" dropdown was replaced because users couldn't
see at a glance what actions were available).  Each click opens its
modal dialog directly:

| Button | Action | Disabled when | API |
|---|---|---|---|
| 🗂 New project | Modal for project name; backend bootstraps the canonical-topic tree | never | `createProject` |
| 📁 New folder | Modal for folder name in current dir | at the `projects/` root | `mkdir` |
| ⬆ Upload | Modal with `<input type="file">` + Upload button | at the `projects/` root | `upload` |

Icons sit above stacked text labels so the row stays compact at narrow
widths.  Stable anchor ids: `#ps-create-project-btn`,
`#ps-create-folder-btn`, `#ps-create-upload-btn`.  The depth-aware
disabled state is driven by an `onChange` subscriber that gates on
`atRoot()`.

### 5.2 Per-entry `⋯` kebab

A button on the right edge of each entry row.  Drops a contextual menu
whose items are eligibility-gated (ineligible items are omitted, not
shown greyed):

| Item | Available for | Result | API |
|---|---|---|---|
| View | files only | Sets `setShared(dir, file)` + opens the preview modal (§ 9) | `readFile` / `readRange` |
| Download | files only | Streams the file (any kind: text, binary, multi-MB) | `GET /api/files/download` (`Content-Disposition: attachment`) |
| Rename… | anything `_isDeletableEntry` allows | Modal for new name; sidecar-pair on `.xyz`/`.pdb` | `rename` |
| Move to… | files only | Tree-picker for destination dir; sidecar-paired | `move` |
| Copy to… | files only | Tree-picker; same-dir copy prompts for a new name | `copy` |
| Delete | anything `_isDeletableEntry` allows | Destructive confirm modal, then delete + refresh; sidecar-paired on `.xyz`/`.pdb` (removes the `.molstruct.json`) | `deleteEntry` |

The "sidecar-pair" notes on Rename / Move / Copy / Delete above mean the
paired `.molstruct.json` follows the `.xyz`/`.pdb` — the rule (which ops,
and when) is specified once in [`projects-sidebar.md`](projects-sidebar.md)
§ 5.4; this table only flags which items are sidecar-aware.

Eligibility check `_isDeletableEntry` is the source of truth for "can
the user mutate this" in the table above: it refuses the `projects/`
root itself and refuses canonical-topic dirs at depth 1 (would orphan
the project layout).

**Directory-only kebabs (two extra menu shapes).** Project dirs and
canonical-topic dirs are excluded from the file kebab above, but each
has its own **single-item** kebab for a whole-tree delete
(`lib/projects/list.js`):

| Entry | Menu | Result | API |
|---|---|---|---|
| **Project dir** (depth 0) | **Delete project…** | Recursively removes the whole project. **Type-the-name confirm** (`window.prompt` — the user must type the project name), then `force:true` delete + refresh. | `deleteEntry(path, {force:true})` |
| **Canonical-topic dir** (depth 1, e.g. `structure/`, `results/`) | **Delete directory…** | Recursively removes the topic dir + its contents. Same type-the-name confirm + `force:true`. | `deleteEntry(path, {force:true})` |

These deliberately bypass `_isDeletableEntry` (which forbids deleting a
project/topic dir via the file kebab) because they are the *explicit*
whole-tree delete path, gated by the stricter type-the-name confirm
rather than the `dialogs.js` default-focus-Cancel modal.

The kebab menu auto-dismisses on outside click + ESC + scroll.

### 5.3 Dialogs

Dialogs live in `lib/projects/dialogs.js` and follow the
single-instance + ESC-as-Cancel pattern from
`modify/structure/save-dialog.js`:

* opening ANY dialog while one is already open cancels the open one
  (resolves it to `null`) and opens the new one — one modal at a time, no
  stacking (the impl always tears down the active dialog and mounts a fresh
  one; there is no "return the existing promise" fast-path).

Destructive flows (overwrite confirm, delete) default-focus on Cancel —
the user has to deliberately travel to the destructive button.

---

## 6. Width resize

The right edge of `.projects-sidebar` carries a 4px-wide drag handle
(`#ps-resize-handle`).  Drag horizontally → `--ps-w` updates live;
release → width persists to
`localStorage.molbuilder.projects_sidebar_width` (px).  Restored BEFORE
layout-sensitive widgets (Plotly, 3Dmol) paint, same reasoning as the
collapsed-state restore (§ 7.2).

Bounds (JS-clamped):

| Min | Max |
|---|---|
| 14rem (224px) | 40rem (640px) |

Double-clicking the handle clears the persisted value (reverts to the
default 18rem).  Handle is CSS-hidden when the sidebar is collapsed or
running in narrow-viewport drawer mode.

---

## 7. Responsive layout — drawer + collapse

### 7.1 Narrow-viewport drawer (≤ 640 px)

**Added 2026-06-02 for task #182.** The sidebar's default desktop
layout (`position: fixed`, left, 18 rem wide, body shifted right by
`padding-left: 18rem`) doesn't fit a phone-width viewport: at 360 px
viewport the body would have to be ≥ 648 px wide and produces a
horizontal scrollbar.

At viewport ≤ 640 px, the sidebar becomes a left-edge drawer:

* **Body**: `padding-left` collapses to `0` (the sidebar is no longer
  part of normal flow).
* **Sidebar**: `transform: translateX(-100%)` slides it off-canvas with
  a 180 ms ease-out transition.  Body class `has-mobile-sidebar-open`
  resets the transform to bring it back as a fixed-position overlay.
* **Hamburger button** (`#ps-mobile-toggle`): fixed at top-left,
  visible only at narrow widths via `display: none` outside the media
  query.  Toggles the body class.  Aria: `aria-controls=
  "projects-sidebar"`, `aria-expanded=` mirrors the class state.
* **Backdrop** (`#ps-mobile-backdrop`): semi-transparent overlay
  visible only when the drawer is open.  Click dismisses.
* **Escape key**: dismisses (standard modal-overlay convention).
* **Resize past breakpoint**: auto-dismisses so rotating from portrait
  to landscape doesn't leave a stale "open" state.

Z-index layering (bottom up):

| Layer | z-index | Why |
|---|---|---|
| Page content | (none, normal flow) | — |
| Backdrop | 85 | Dims the page but not the drawer |
| Drawer sidebar | 90 | Overlays page + backdrop |
| Toggle button | 95 | Stays tappable when drawer is open |
| File-preview modal | 100 | Above the drawer so a modal opened FROM the drawer is not hidden behind it |

A closed modal (`hidden` attr → `display: none`) doesn't participate in
stacking, so the desktop case (sidebar `z-index: 5`, modal `100`) is
unaffected.

JS wiring lives in `lib/projects/projects-sidebar.js::initMobileDrawer`; the
function is a no-op if the optional toggle / backdrop elements are
absent (forward-compat with future templates that drop the
scaffolding).

### 7.2 Desktop hide/show toggle (Phase B.5.4)

On viewports ≥ 640 px the sidebar can be collapsed entirely so the
active tab's workspace gets the full window width.  Three DOM
affordances:

* `#ps-collapse-toggle` — small "◀" button in the sidebar header
  (`.ps-header`); click hides the sidebar.
* `#ps-collapsed-handle` — accent-coloured floating dock tab
  ("Projects" label + chevron) fixed at the page's left edge.  Hidden
  unless the sidebar is collapsed; click brings it back.  Lives outside
  `.projects-sidebar` so it stays visible when the sidebar itself is
  hidden.
* `body.is-projects-sidebar-collapsed` — body-level class.  When set,
  the body's `padding-left` collapses to 0 and `.projects-sidebar`
  slides off-canvas via `translateX(-100%)` + `visibility: hidden`
  (keeps it out of the a11y tree).

State persists in `sessionStorage` under
`molbuilder.projects_sidebar_collapsed`.
`lib/projects/projects-sidebar.js::_restoreCollapsedState` reads it and applies
the body class BEFORE the rest of init runs — so any layout-sensitive
widget (Plotly chart, 3Dmol canvas, CSS-grid auto-fit) measures the
correct geometry on its first paint.

Below the 640 px breakpoint the desktop affordances hide and the mobile
drawer (§ 7.1) takes over so the two systems don't double up.

---

## 8. Visibility model — the `[hidden]` trap

### 8.1 The trap (and the rule we follow because of it)

The browser's `[hidden] { display: none }` rule and any author
`.foo { display: <non-none> }` rule have the **same specificity**.  On
a tie, author CSS wins by cascade order.  An element with `class="foo"`
AND `hidden=""` is rendered VISIBLE despite the attribute.

**Today's rule**: every author `display:` rule on a class whose element
may carry `hidden` MUST be paired with a `.foo[hidden] { display: none }`
guard.  Higher specificity wins the tie.

### 8.2 Design direction

The current pattern is fragile — it requires every CSS contributor to
remember the guard rule and every reviewer to check for it.  The
design's end state is:

* One global helper class — `.is-hidden { display: none !important }`.
* HTML `hidden=` is banned in sidebar templates; replaced by
  `class="is-hidden"` toggled via a tiny `setVisible(el, bool)` helper.
* A CI grep flags any new use of `hidden=` so the bug class can't
  re-enter.

Until that migration completes (master § 13 M7), the guard rule (§ 8.1)
is the contract.

---

## 9. File-content preview modal

`lib/projects/preview.js`.  A **generic, format-blind** view + edit +
save surface for **any** file the sidebar can reach — opened from the
kebab's **View** item, anchored to `#ps-preview-modal`.

> **This is NOT the structure Save panel.**  This modal reads and
> writes **raw bytes** at one path via the file layer.  Saving a
> structure (geometry + `.molstruct.json` sidecar) is a *different*,
> structure-aware flow through `projects.parser.saveMolecule` — see
> [`save-flow.md`](save-flow.md).  Editing a `.molstruct.json` by hand
> here writes bytes verbatim; it does not go through the model.

### 9.1 The editor

A single **CodeMirror 5** instance (lazy-loaded on first open from the
vendored bundle under `/static/vendor/codemirror/`, then reused across
opens).  Replaces the earlier read-only `<pre>` + editable `<textarea>`:

* **Virtual scrolling** caps DOM memory at the visible window
  regardless of file size (a fully-scrolled 500 MB log no longer puts
  500 MB of text in the page).
* **Search** — the footer **Find…** button runs
  `_cm.execCommand("find")` (same command Ctrl-F binds to); the
  vendored search addon gives Ctrl-F over the whole loaded doc.
* **Jump-to-line** — the jump-to-line addon gives Alt-G "go to line".
* **Edit mode** is the same editor with `readOnly` toggled off.

Footer buttons, left to right: **Find… · Edit · Save · Close**.  The
modal fills 80 vh (`height: 80vh` on `.ps-preview-window`) so the flex
column has a definite slot for the absolutely-positioned editor.

### 9.2 Read + size handling

Reads go through the file layer (`readFile` / `readRange`; wire:
[`web-api.md`](web-api.md) § 3):

| Threshold (constant) | Behavior |
|---|---|
| ≤ 1 MB — `VIEW_ONLY_BYTES` | Editable: full edit + select + Ctrl-A path. |
| > 1 MB | **View-only** — Edit button disabled, text selection gated (see below). |
| Bulk single-shot read | Requested with `max_bytes` = the server's 16 MB `read` ceiling (`BULK_READ_MAX_BYTES`), so mid-sized result files load without a 413. |
| Past `BULK_READ_MAX_BYTES` (16 MB — the server's single-shot `read` ceiling) | Loaded in `256 KB` chunks (`PAGE_BYTES`) via `/api/files/read_range`; view-only (Edit disabled) with a "use external editor" hint. Note: paginated view-only begins at the 16 MB bulk ceiling, *not* the 32 MB edit cap — a single-shot editable read is impossible above 16 MB. |
| Non-UTF-8 (400 from `read`) | Edit disabled — the v1 contract is text-only. |

**Why view-only above 1 MB**: keystroke-triggered CodeMirror operations
on multi-MB docs are pathologically slow (Ctrl-A `selectAll` measured
~225 s in headless Chromium on a 2 MB doc).  Ctrl-A / Cmd-A is disabled
via `extraKeys`, and any selection (mouse-drag, programmatic,
shift-arrow) is clamped to `MAX_SELECTION_LINES = 1500` (~100 KB) via
`beforeSelectionChange` for O(1) cost per drag tick.

### 9.3 Edit + save — the mtime-safe overwrite contract

Edit mode saves through the file layer's `writeFile` →
`POST /api/files/write` (wire: [`web-api.md`](web-api.md) § 3.6):

* On read, the response's **`mtime`** is captured into `_state`.
* Save sends that value back as **`expected_mtime`**.
* If the on-disk mtime changed in the meantime, the server returns
  **409**; the modal shows a "file changed on disk; reload to see the
  new content" prompt **instead of silently clobbering** someone else's
  edits.  The user reloads to pick up the new bytes, then re-edits.

This is the same concurrent-edit detection the master specifies for all
writes (master § 4.4, § 5.5, § 11.2).

### 9.4 Whole-file capture (bypasses the editor)

The kebab's **Download** item does **not** go through the editor: it
streams the file via `GET /api/files/download`
(`Content-Disposition: attachment`) with **no size cap and no UTF-8
cap** — the right tool for binary or very large files the editor won't
load.

### 9.5 Close

The modal closes on ESC / backdrop click / the Close button.  It sits
at `z-index: 100` so a modal opened from the narrow-viewport drawer
(§ 7.1) is never hidden behind it.

---

## 10. Change protocol

This doc is aspect E of the Projects Sidebar module; it follows the
master's change protocol ([`projects-sidebar.md`](projects-sidebar.md)
§ 16): update the doc in the SAME commit as the code + tests.  A UI
change that alters the DOM contract (IDs/classes consumers query) MUST
update the tables here; a change to *what the UI calls* (public API
signatures, endpoints) belongs in the master or `web-api.md`, not here —
this doc references those, it does not own them.
